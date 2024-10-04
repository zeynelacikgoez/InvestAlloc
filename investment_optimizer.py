import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, COBYLA, TNC, linprog
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import logging
import concurrent.futures
import sys
import copy
import argparse
import json
import os
from functools import lru_cache
from numba import njit

# ==========================
# Logging-Konfiguration
# ==========================
def configure_logging(level=logging.INFO):
    """
    Konfiguriert das Logging.

    Parameters:
    ----------
    level : int
        Logging-Level (z.B. logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# ==========================
# Hilfsfunktionen
# ==========================
def is_symmetric(matrix, tol=1e-8):
    """
    Überprüft, ob eine Matrix symmetrisch ist.

    Parameters:
    ----------
    matrix : np.array
        Zu überprüfende Matrix.
    tol : float, optional
        Toleranz für den Symmetriecheck. Standard ist 1e-8.

    Returns:
    -------
    bool
        True, wenn symmetrisch, sonst False.
    """
    return np.allclose(matrix, matrix.T, atol=tol)


def validate_inputs(a, b, B, L, U, x0):
    """
    Validiert die Eingabeparameter.

    Parameters:
    ----------
    a : np.array
        Effizienzparameter.
    b : np.array
        Synergieeffekte.
    B : float
        Gesamtbudget.
    L : np.array
        Mindestinvestitionen.
    U : np.array
        Höchste Investitionen.
    x0 : np.array
        Anfangsschätzung.

    Raises:
    ------
    ValueError
        Bei inkonsistenten Eingaben.
    """
    n = len(a)
    b = np.array(b)
    if b.shape != (n, n):
        raise ValueError(f"Synergiematrix b muss quadratisch sein und mit der Länge von a ({n}) übereinstimmen.")
    if not is_symmetric(b):
        raise ValueError("Synergiematrix b muss symmetrisch sein.")
    if not (len(L) == len(U) == len(x0) == n):
        raise ValueError("L, U und x0 müssen die gleiche Länge wie a haben.")
    if np.any(L > U):
        raise ValueError("Für alle Bereiche muss L[i] <= U[i] gelten.")
    if np.sum(L) > B:
        raise ValueError("Die Summe der Mindestinvestitionen überschreitet das Gesamtbudget B.")
    if np.any(x0 < L):
        raise ValueError("Anfangsschätzung x0 unterschreitet mindestens eine Mindestinvestition L.")
    if np.any(x0 > U):
        raise ValueError("Anfangsschätzung x0 überschreitet mindestens eine Höchstinvestition U.")


def get_bounds(L, U):
    """
    Definiert die Investitionsgrenzen für jeden Bereich.

    Parameters:
    ----------
    L : np.array
        Mindestinvestitionen.
    U : np.array
        Höchste Investitionen.

    Returns:
    -------
    list of tuples
        Bounds für jede Variable.
    """
    return [(L[i], U[i]) for i in range(len(L))]


@njit
def compute_synergy(x, b):
    """
    Effiziente Berechnung der Synergieeffekte mit Numba.

    Parameters:
    ----------
    x : np.array
        Investitionen in die Bereiche.
    b : np.array
        Synergieeffekte (quadratische Matrix).

    Returns:
    -------
    float
        Gesamtsynergieeffekt.
    """
    synergy = 0.0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            synergy += b[i, j] * x[i] * x[j]
    return synergy


def objective(x, a, b, epsilon=1e-8):
    """
    Berechnet den negativen Gesamtnutzen basierend auf den Investitionen x.

    Nutzen wird durch den Logarithmus der Investitionen gewichtet mit Effizienzparametern 
    sowie durch Synergieeffekte zwischen den Investitionsbereichen bestimmt.

    Parameters:
    ----------
    x : np.array
        Investitionen in die Bereiche.
    a : np.array
        Effizienzparameter für jeden Bereich.
    b : np.array
        Synergieeffekte zwischen den Bereichen (quadratische Matrix).
    epsilon : float, optional
        Kleine Zahl zur Vermeidung von log(0). Standard ist 1e-8.

    Returns:
    -------
    float
        Negativer Gesamtnutzen (für Minimierung).
    """
    x = np.array(x)
    a = np.array(a)

    # Vermeidung von log(0) durch Hinzufügen einer kleinen Konstante
    utility = np.sum(a * np.log(x + epsilon))
    synergy = compute_synergy(x, b)
    utility += synergy
    logging.debug(f"Nutzen: {utility}")
    return -utility


def objective_with_penalty(x, a, b, B, L, penalty_factors):
    """
    Berechnet den negativen Gesamtnutzen mit quadratischen Strafkosten für Verletzung der Nebenbedingungen.

    Parameters:
    ----------
    x : np.array
        Investitionen in die Bereiche.
    a : np.array
        Effizienzparameter.
    b : np.array
        Synergieeffekte.
    B : float
        Gesamtbudget.
    L : np.array
        Mindestinvestitionen.
    penalty_factors : dict
        Straffaktoren für 'budget' und 'min'.

    Returns:
    -------
    float
        Negativer Gesamtnutzen plus Strafkosten.
    """
    penalty = 0
    budget_excess = np.sum(x) - B
    if budget_excess > 0:
        penalty += (budget_excess ** 2) * penalty_factors['budget']
        logging.debug(f"Quadratische Budgetüberschreitung um {budget_excess} mit Strafkosten {penalty}")
    min_deficit = L - x
    min_deficit = np.where(min_deficit > 0, min_deficit, 0)
    penalty += np.sum((min_deficit ** 2) * penalty_factors['min'])
    logging.debug(f"Quadratische Mindestinvestitionsdefizite: {min_deficit}, Strafkosten insgesamt: {np.sum((min_deficit ** 2) * penalty_factors['min'])}")
    return objective(x, a, b) + penalty


# ==========================
# Ergebnisklasse
# ==========================
class OptimizationResult:
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)


# ==========================
# InvestmentOptimizer Klasse
# ==========================
class InvestmentOptimizer:
    def __init__(self, a, b, B, L, U, x0, investment_labels=None, log_level=logging.INFO):
        """
        Initialisiert das InvestmentOptimizer Objekt.

        Parameters:
        ----------
        a : np.array
            Effizienzparameter.
        b : np.array
            Synergieeffekte.
        B : float
            Gesamtbudget.
        L : np.array
            Mindestinvestitionen.
        U : np.array
            Höchste Investitionen.
        x0 : np.array
            Anfangsschätzung.
        investment_labels : list of str, optional
            Namen der Investitionsbereiche. Standard ist ['Bereich_0', 'Bereich_1', ...].
        log_level : int, optional
            Logging-Level. Standard ist logging.INFO.
        """
        configure_logging(log_level)
        self.a = np.array(a)
        self.b = np.array(b)
        self.B = B
        self.L = np.array(L)
        self.U = np.array(U)
        self.n = len(a)
        self.investment_labels = investment_labels if investment_labels else [f'Bereich_{i}' for i in range(self.n)]
        self.x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
        validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0)

    def update_parameters(self, a=None, b=None, B=None, L=None, U=None, x0=None):
        """
        Aktualisiert die Parameter des Optimizers.

        Parameters:
        ----------
        a : np.array, optional
            Neue Effizienzparameter.
        b : np.array, optional
            Neue Synergieeffekte.
        B : float, optional
            Neues Gesamtbudget.
        L : np.array, optional
            Neue Mindestinvestitionen.
        U : np.array, optional
            Neue Höchste Investitionen.
        x0 : np.array, optional
            Neue Anfangsschätzung.

        Raises:
        ------
        ValueError
            Bei inkonsistenten Eingaben.
        """
        if a is not None:
            self.a = np.array(a)
        if b is not None:
            self.b = np.array(b)
        if B is not None:
            self.B = B
        if L is not None:
            self.L = np.array(L)
        if U is not None:
            self.U = np.array(U)
        if x0 is not None:
            self.x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
        validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0)

    def identify_top_synergies(self, top_n=None):
        """
        Identifiziert die Investitionspaare mit den höchsten Synergieeffekten.

        Parameters:
        ----------
        top_n : int, optional
            Anzahl der Top-Paare, die zurückgegeben werden sollen.

        Returns:
        -------
        list of tuples
            Paare (i, j) sortiert nach absteigenden Synergieeffekten.
        """
        synergies = []
        n = self.b.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                synergies.append(((i, j), self.b[i, j]))
        # Sortieren nach Synergieeffekt
        synergies.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            synergies = synergies[:top_n]
        return synergies

    def optimize(self, method='SLSQP', max_retries=3, initial_penalty_factors=None, workers=None, adaptive_penalty=True, max_penalty=1e6, interactive_plot=False, **kwargs):
        """
        Führt die Optimierung der Investitionsallokationen durch.

        Diese Methode verwendet die angegebene Optimierungsmethode, um die Investitionen
        in verschiedene Bereiche zu optimieren, sodass der Gesamtnutzen maximiert wird,
        unter Einhaltung der Budget- und Investitionsbeschränkungen.

        Parameters:
        ----------
        method : str, optional
            Die Optimierungsmethode, die verwendet werden soll. Unterstützte Methoden sind:
            - 'SLSQP': Sequential Least Squares Programming
            - 'DE': Differential Evolution
            - 'BasinHopping': Basin Hopping
            - 'COBYLA': Constrained Optimization BY Linear Approximations
            - 'TNC': Truncated Newton Conjugate-Gradient
            Weitere Methoden können durch Erweiterung der Optimierungsmethoden hinzugefügt werden.
            Standard ist 'SLSQP'.
        max_retries : int, optional
            Die maximale Anzahl an Wiederholungsversuchen bei Optimierungsfehlern. Standard ist 3.
        initial_penalty_factors : dict, optional
            Ein Wörterbuch mit initialen Straffaktoren für 'budget' und 'min'. Beispiel:
            {'budget': 100, 'min': 100}. Standard ist {'budget': 100, 'min': 100}.
        workers : int, optional
            Die Anzahl der parallelen Worker für Methoden wie Differential Evolution. 
            Standard ist 1.
        adaptive_penalty : bool, optional
            Ob eine adaptive Anpassung der Straffaktoren verwendet werden soll. Standard ist True.
        max_penalty : float, optional
            Maximale Straffaktoren, um zu hohe Werte zu verhindern. Standard ist 1e6.
        interactive_plot : bool, optional
            Ob interaktive Plots verwendet werden sollen. Standard ist False.
        **kwargs : dict
            Zusätzliche Hyperparameter, die an die spezifische Optimierungsmethode weitergegeben werden.

        Returns:
        -------
        OptimizationResult or None
            Das Ergebnis der Optimierung, ähnlich wie bei `scipy.optimize.minimize`.
            Gibt `None` zurück, wenn die Optimierung fehlschlägt.

        Raises:
        ------
        ValueError
            Wenn eine nicht unterstützte Optimierungsmethode angegeben wird.
        """
        if initial_penalty_factors is None:
            penalty_factors = {'budget': 100, 'min': 100}  # Moderate Werte
        else:
            penalty_factors = copy.deepcopy(initial_penalty_factors)
            penalty_factors.setdefault('budget', 100)
            penalty_factors.setdefault('min', 100)

        bounds = get_bounds(self.L, self.U)
        objective_fn = lambda x: objective_with_penalty(
            x, self.a, self.b, self.B, self.L, penalty_factors
        )

        optimization_methods = {
            'SLSQP': lambda: minimize(
                objective_fn,
                self.x0,
                method='SLSQP',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000},
                **kwargs
            ),
            'DE': lambda: differential_evolution(
                objective_fn,
                bounds,
                strategy=kwargs.get('strategy', 'best1bin'),
                maxiter=kwargs.get('maxiter', 1000),
                tol=kwargs.get('tol', 1e-6),
                updating='deferred',
                workers=workers if workers is not None else 1,
                polish=True,
                init='latinhypercube'
            ),
            'BasinHopping': lambda: basinhopping(
                objective_fn,
                self.x0,
                niter=kwargs.get('niter', 100),
                stepsize=kwargs.get('stepsize', 0.5),
                minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds, 'options': {'maxiter': 1000}},
                **kwargs
            ),
            'COBYLA': lambda: minimize(
                objective_fn,
                self.x0,
                method='COBYLA',
                constraints={'type': 'ineq', 'fun': lambda x: self.B - np.sum(x)},
                options={'disp': False, 'maxiter': 1000},
                **kwargs
            ),
            'TNC': lambda: minimize(
                objective_fn,
                self.x0,
                method='TNC',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000},
                **kwargs
            )
            # Weitere Methoden können hier hinzugefügt werden
        }

        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods:
            logging.error(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden sind: {supported_methods}")
            raise ValueError(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden sind: {supported_methods}")

        for attempt in range(1, max_retries + 1):
            logging.info(f"Versuch {attempt} der Optimierung mit Methode {method}...")
            result = None
            try:
                optimization_result = optimization_methods[method]()
                # Standardisieren der Ergebnisse
                if method == 'DE':
                    result = OptimizationResult(
                        x=optimization_result.x,
                        fun=optimization_result.fun,
                        success=optimization_result.success,
                        message=optimization_result.message
                    )
                else:
                    result = OptimizationResult(
                        x=optimization_result.x,
                        fun=optimization_result.fun,
                        success=optimization_result.success,
                        message=optimization_result.message
                    )
            except Exception as e:
                logging.error(f"{method} Optimierungsfehler beim Versuch {attempt}: {e}")
                result = OptimizationResult(
                    x=None,
                    fun=None,
                    success=False,
                    message=str(e)
                )

            if result and getattr(result, 'success', False):
                logging.info(f"Optimierung mit {method} erfolgreich abgeschlossen beim Versuch {attempt}.")
                return result
            else:
                if result:
                    logging.warning(f"Optimierung mit {method} fehlgeschlagen beim Versuch {attempt}: {getattr(result, 'message', 'Kein Ergebnis')}")
                else:
                    logging.warning(f"Optimierung mit {method} fehlgeschlagen beim Versuch {attempt}: Kein Ergebnis")

                # Adaptive Anpassung der Straffaktoren basierend auf der Schwere der Verletzung
                if adaptive_penalty:
                    if result.x is not None:
                        budget_violation = np.sum(result.x) - self.B
                        min_violation = np.sum(np.maximum(self.L - result.x, 0))
                    else:
                        budget_violation = 0
                        min_violation = 0
                    if budget_violation > 0:
                        penalty_factors['budget'] = min(penalty_factors['budget'] * 10, max_penalty)
                    if min_violation > 0:
                        penalty_factors['min'] = min(penalty_factors['min'] * 10, max_penalty)
                    logging.info(f"Adaptive Erhöhung der Straffaktoren: budget={penalty_factors['budget']}, min={penalty_factors['min']}")
                else:
                    # Feste Multiplikation mit Begrenzung
                    penalty_factors['budget'] = min(penalty_factors['budget'] * 10, max_penalty)
                    penalty_factors['min'] = min(penalty_factors['min'] * 10, max_penalty)
                    logging.info(f"Erhöhung der Straffaktoren: budget={penalty_factors['budget']}, min={penalty_factors['min']}")

        logging.error(f"Optimierung mit {method} nach {max_retries} Versuchen fehlgeschlagen.")
        return None

    def sensitivity_analysis(self, B_values, method='SLSQP', **kwargs):
        """
        Führt eine Sensitivitätsanalyse durch, indem das Budget variiert und die optimale Allokation berechnet wird.

        Parameters:
        ----------
        B_values : np.array
            Verschiedene Budgetwerte.
        method : str, optional
            Optimierungsmethode. Standard ist 'SLSQP'.
        **kwargs : dict
            Zusätzliche Parameter an die Optimierungsmethode.

        Returns:
        -------
        tuple
            (Budgets, optimale Allokationen, maximale Nutzen)
        """
        optimal_allocations = []
        max_utilities = []

        for B in B_values:
            logging.info(f"Durchführung der Optimierung für Budget B={B}...")
            # Erstellen eines neuen Optimizers mit aktualisiertem Budget
            optimizer_copy = copy.deepcopy(self)
            optimizer_copy.B = B
            optimizer_copy.x0 = adjust_initial_guess(optimizer_copy.x0, optimizer_copy.L, optimizer_copy.U, B)
            try:
                validate_inputs(optimizer_copy.a, optimizer_copy.b, optimizer_copy.B, optimizer_copy.L, optimizer_copy.U, optimizer_copy.x0)
            except ValueError as e:
                logging.error(f"Validierungsfehler für Budget B={B}: {e}")
                optimal_allocations.append([np.nan] * self.n)
                max_utilities.append(np.nan)
                continue

            result = optimizer_copy.optimize(method=method, **kwargs)
            if result and result.success:
                allocation = result.x
                utility = -result.fun
                optimal_allocations.append(allocation)
                max_utilities.append(utility)
            else:
                logging.warning(f"Optimierung für Budget B={B} fehlgeschlagen oder kein Ergebnis verfügbar.")
                optimal_allocations.append([np.nan] * self.n)
                max_utilities.append(np.nan)

        return B_values, optimal_allocations, max_utilities

    def robustness_analysis(self, num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=None, **kwargs):
        """
        Führt eine Robustheitsanalyse durch, indem die Parameter a und b zufällig variiert werden.

        Parameters:
        ----------
        num_simulations : int, optional
            Anzahl der Simulationen. Standard ist 100.
        method : str, optional
            Optimierungsmethode. Standard ist 'DE'.
        variation_percentage : float, optional
            Prozentuale Variation der Parameter. Standard ist 0.1 (10%).
        parallel : bool, optional
            Ob parallele Verarbeitung genutzt werden soll. Standard ist True.
        num_workers : int, optional
            Anzahl der parallelen Worker. Standard ist None (wird auf Anzahl der CPU-Kerne gesetzt).
        **kwargs : dict
            Zusätzliche Parameter an die Optimierungsmethode.

        Returns:
        -------
        pandas.DataFrame
            Zusammenfassung der Ergebnisse (median, std, min, max, 25%, 75% für jede Investition).
        """
        def single_simulation(sim_index, base_optimizer, variation_percentage):
            logging.info(f"Robustheitsanalyse Simulation {sim_index + 1}/{num_simulations}...")
            # Zufällige Variation der Parameter um ±variation_percentage
            a_sim = base_optimizer.a * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=base_optimizer.a.shape)
            b_sim = base_optimizer.b * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=base_optimizer.b.shape)
            b_sim = np.triu(b_sim, 1) + np.triu(b_sim, 1).T  # Symmetrisch
            b_sim = np.maximum(b_sim, 0)  # Synergieeffekte nicht negativ

            # Erstellen eines neuen Optimizers mit variierten Parametern
            try:
                optimizer_sim = InvestmentOptimizer(
                    a=a_sim,
                    b=b_sim,
                    B=base_optimizer.B,
                    L=base_optimizer.L,
                    U=base_optimizer.U,
                    x0=base_optimizer.x0,  # Keine deepcopy nötig, da x0 wird intern angepasst
                    investment_labels=base_optimizer.investment_labels,
                    log_level=logging.CRITICAL
                )
            except ValueError as e:
                logging.error(f"Validierungsfehler in Simulation {sim_index + 1}: {e}")
                return [np.nan] * base_optimizer.n

            # Optimierung durchführen
            result = optimizer_sim.optimize(method=method, workers=1, **kwargs)
            if result and result.success:
                allocation = result.x
                return allocation
            else:
                return [np.nan] * base_optimizer.n

        results = []
        base_optimizer = copy.deepcopy(self)

        if parallel:
            if num_workers is None:
                num_workers = os.cpu_count() or 1
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(single_simulation, sim, base_optimizer, variation_percentage) for sim in range(num_simulations)]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        else:
            for sim in range(num_simulations):
                results.append(single_simulation(sim, base_optimizer, variation_percentage))

        # Analysieren der Ergebnisse
        df_results = pd.DataFrame(results, columns=self.investment_labels)
        additional_stats = df_results.agg(['median', 'std', 'min', 'max', '25%', '75%']).transpose()
        return additional_stats

    def plot_sensitivity(self, B_values, optimal_allocations, max_utilities, method='SLSQP', interactive=False):
        """
        Plotet die Ergebnisse der Sensitivitätsanalyse.

        Parameters:
        ----------
        B_values : np.array
            Budgetwerte.
        optimal_allocations : list of np.array
            Optimale Investitionsallokationen.
        max_utilities : list of float
            Maximale Nutzen.
        method : str, optional
            Optimierungsmethode. Standard ist 'SLSQP'.
        interactive : bool, optional
            Ob interaktive Plots genutzt werden sollen. Standard ist False.
        """
        if interactive:
            # Plotly nutzen für interaktive Plots
            data_alloc = {label: [alloc[i] if not np.isnan(alloc[i]) else 0 for alloc in optimal_allocations]
                         for i, label in enumerate(self.investment_labels)}
            data_alloc['Budget'] = B_values
            df_alloc = pd.DataFrame(data_alloc)

            data_util = {
                'Budget': B_values,
                'Maximaler Nutzen': max_utilities
            }
            df_util = pd.DataFrame(data_util)

            fig1 = px.line(df_alloc, x='Budget', y=self.investment_labels, title='Optimale Investitionsallokationen')
            fig1.add_traces(px.line(df_util, x='Budget', y='Maximaler Nutzen').data)
            fig1.update_layout(yaxis_title='Investitionsbetrag / Nutzen', legend_title='Investitionsbereiche')
            fig1.show()
        else:
            # Matplotlib nutzen für statische Plots
            fig, ax1 = plt.subplots(figsize=(12, 8))

            colors = sns.color_palette("tab10", n_colors=self.n)

            for i, label in enumerate(self.investment_labels):
                allocations = [alloc[i] if not np.isnan(alloc[i]) else 0 for alloc in optimal_allocations]
                ax1.plot(B_values, allocations, label=label, color=colors[i], alpha=0.7)
            ax1.set_xlabel('Budget B')
            ax1.set_ylabel('Investitionsbetrag x_i')
            ax1.tick_params(axis='y')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(B_values, max_utilities, label='Maximaler Nutzen', color='tab:red', marker='o')
            ax2.set_ylabel('Maximaler Nutzen')
            ax2.tick_params(axis='y')
            ax2.legend(loc='upper right')

            plt.title(f'Optimale Investitionsallokation und Nutzen bei verschiedenen Budgets ({method})')
            plt.show()

    def plot_synergy_heatmap(self):
        """
        Plotet eine Heatmap der Synergieeffekte.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.b, annot=True, xticklabels=self.investment_labels, yticklabels=self.investment_labels, cmap='viridis')
        plt.title('Heatmap der Synergieeffekte')
        plt.show()

    def plot_parameter_sensitivity(self, parameter_values, parameter_name, parameter_index=None, method='SLSQP', interactive=False, **kwargs):
        """
        Plotet die Sensitivität des Nutzens gegenüber einem einzelnen Parameter.

        Parameters:
        ----------
        parameter_values : list or np.array
            Werte des Parameters.
        parameter_name : str
            Name des Parameters ('a' oder 'b').
        parameter_index : tuple or int, optional
            Index des Parameters. Für 'a' ein int, für 'b' ein tuple (i, j).
            Beispiel: 0 für a[0], (0,1) für b[0,1].
        method : str, optional
            Optimierungsmethode. Standard ist 'SLSQP'.
        interactive : bool, optional
            Ob interaktive Plots genutzt werden sollen. Standard ist False.
        **kwargs : dict
            Zusätzliche Parameter an die Optimierungsmethode.
        """
        utilities = []
        allocations = []

        for value in parameter_values:
            optimizer_copy = copy.deepcopy(self)
            if parameter_name == 'a':
                if not isinstance(parameter_index, int):
                    raise ValueError("Für Parameter 'a' muss parameter_index ein int sein.")
                a_new = optimizer_copy.a.copy()
                a_new[parameter_index] = value
                optimizer_copy.update_parameters(a=a_new)
            elif parameter_name == 'b':
                if not (isinstance(parameter_index, tuple) and len(parameter_index) == 2):
                    raise ValueError("Für Parameter 'b' muss parameter_index ein tuple mit zwei Indizes sein, z.B. (0,1).")
                i, j = parameter_index
                b_new = optimizer_copy.b.copy()
                b_new[i, j] = value
                b_new[j, i] = value  # Symmetrisch
                optimizer_copy.update_parameters(b=b_new)
            else:
                logging.error(f"Parametername '{parameter_name}' wird nicht unterstützt für Sensitivitätsanalyse.")
                return

            result = optimizer_copy.optimize(method=method, **kwargs)
            if result and result.success:
                utilities.append(-result.fun)
                allocations.append(result.x)
            else:
                utilities.append(np.nan)
                allocations.append([np.nan] * self.n)

        if interactive:
            # Plotly nutzen für interaktive Plots
            plt.figure(figsize=(10, 6))
            fig = px.line(x=parameter_values, y=utilities, labels={'x': f'{parameter_name}{parameter_index if parameter_index is not None else ""}', 'y': 'Maximaler Nutzen'},
                          title=f'Sensitivität des Nutzens gegenüber {parameter_name}{parameter_index if parameter_index is not None else ""}')
            fig.show()
        else:
            # Matplotlib nutzen für statische Plots
            plt.figure(figsize=(10, 6))
            plt.plot(parameter_values, utilities, marker='o')
            plt.xlabel(f'{parameter_name}{parameter_index if parameter_index is not None else ""}')
            plt.ylabel('Maximaler Nutzen')
            plt.title(f'Sensitivität des Nutzens gegenüber {parameter_name}{parameter_index if parameter_index is not None else ""}')
            plt.grid(True)
            plt.show()

    def plot_interactive_sensitivity(self, B_values, optimal_allocations, max_utilities, method='SLSQP'):
        """
        Erstellt ein interaktives Dashboard für die Sensitivitätsanalyse.

        Parameters:
        ----------
        B_values : np.array
            Budgetwerte.
        optimal_allocations : list of np.array
            Optimale Investitionsallokationen.
        max_utilities : list of float
            Maximale Nutzen.
        method : str, optional
            Optimierungsmethode. Standard ist 'SLSQP'.
        """
        # Erstellen eines DataFrames
        data = {'Budget': B_values}
        for i, label in enumerate(self.investment_labels):
            data[label] = [alloc[i] if not np.isnan(alloc[i]) else 0 for alloc in optimal_allocations]
        data['Maximaler Nutzen'] = max_utilities
        df = pd.DataFrame(data)

        # Plot für Investitionsallokationen
        fig1 = px.line(df, x='Budget', y=self.investment_labels, title='Optimale Investitionsallokationen')
        fig1.show()

        # Plot für maximalen Nutzen
        fig2 = px.scatter(df, x='Budget', y='Maximaler Nutzen', title='Maximaler Nutzen bei verschiedenen Budgets')
        fig2.show()

    def plot_robustness_analysis(self, df_results):
        """
        Plotet die Verteilung der Investitionsallokationen aus der Robustheitsanalyse.

        Parameters:
        ----------
        df_results : pandas.DataFrame
            Ergebnisse der Robustheitsanalyse.
        """
        # Boxplot
        df_melted = df_results.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Bereich', y='Investition', data=df_melted)
        plt.xlabel('Investitionsbereich')
        plt.ylabel('Investitionsbetrag')
        plt.title('Verteilung der Investitionsallokationen aus der Robustheitsanalyse')
        plt.show()

        # Histogramme mit KDE
        g = sns.FacetGrid(df_melted, col="Bereich", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.histplot, "Investition", kde=True, bins=20, color='skyblue')
        g.fig.suptitle('Histogramme der Investitionsallokationen', y=1.02)
        plt.show()

        # Scatter-Plots zur Identifikation von Korrelationen zwischen Bereichen
        sns.pairplot(df_results.dropna())
        plt.suptitle('Scatter-Plots der Investitionsallokationen', y=1.02)
        plt.show()

    def plot_additional_synergy_heatmap(self):
        """
        Plotet eine zusätzliche Heatmap der Synergieeffekte (optional, falls benötigt).
        """
        self.plot_synergy_heatmap()


# ==========================
# Hilfsfunktionen für die Anpassung der Anfangsschätzung
# ==========================
def adjust_initial_guess(x0, L, U, B):
    """
    Passt die Anfangsschätzung an, um die Nebenbedingungen zu erfüllen, mittels linearer Programmierung.

    Parameters:
    ----------
    x0 : np.array
        Ursprüngliche Anfangsschätzung.
    L : np.array
        Mindestinvestitionen.
    U : np.array
        Höchste Investitionen.
    B : float
        Gesamtbudget.

    Returns:
    -------
    np.array
        Angepasste Anfangsschätzung.
    """
    n = len(x0)
    # Ziel: Minimierung der Abweichung von x0
    c = np.ones(n)
    A_eq = [np.ones(n)]
    b_eq = [B]
    bounds = [(L[i], U[i]) for i in range(n)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        return res.x
    else:
        logging.warning("Lineare Programmierung zur Anpassung der Anfangsschätzung fehlgeschlagen. Verwende Heuristik.")
        # Fallback: ursprüngliche Heuristik
        x0 = np.clip(x0, L, U)
        total = np.sum(x0)
        if total > B:
            scaling_factor = B / total
            x0 = x0 * scaling_factor
        elif total < B:
            deficit = B - np.sum(x0)
            while deficit > 1e-6:
                increment = deficit / len(x0)
                new_x0 = np.minimum(x0 + increment, U)
                actual_increment = np.sum(new_x0 - x0)
                x0 = new_x0
                deficit = B - np.sum(x0)
                if actual_increment == 0:
                    break
        return x0


# ==========================
# OptimizationResult Klasse
# ==========================
class OptimizationResult:
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)


# ==========================
# Hauptfunktion
# ==========================
def parse_arguments():
    """
    Parst die Kommandozeilenargumente.

    Returns:
    -------
    argparse.Namespace
        Geparste Argumente.
    """
    parser = argparse.ArgumentParser(description='Investment Optimizer')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive_plot', action='store_true', help='Verwende interaktive Plots')
    return parser.parse_args()


def main():
    """
    Hauptfunktion zur Ausführung der Investitionsoptimierung.
    """
    args = parse_arguments()
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(len(config['a']))])
        a = np.array(config['a'])
        b = np.array(config['b'])
        B = config['B']
        L = np.array(config['L'])
        U = np.array(config['U'])
        x0 = np.array(config['x0'])
    else:
        # Standard-Beispielparameter
        investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        n = len(investment_labels)
        a = np.array([1, 2, 3, 4])  # Effizienzparameter für die Bereiche
        b = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0]
        ])
        B = 10.0  # Gesamtbudget in Euro
        L = np.array([1] * n)  # Mindestinvestitionen
        U = np.array([5] * n)  # Höchste Investitionen
        x0 = np.array([2, 2, 2, 4])  # Anfangsschätzung

    # Initialisieren des Optimizers mit DEBUG-Logging (kann angepasst werden)
    optimizer = InvestmentOptimizer(a, b, B, L, U, x0, investment_labels=investment_labels, log_level=logging.DEBUG)

    # Optimierung mit SLSQP
    result_slsqp = optimizer.optimize(method='SLSQP', interactive_plot=args.interactive_plot)

    # Optimierung mit Differential Evolution (globale Optimierung)
    result_de = optimizer.optimize(method='DE', workers=2, interactive_plot=args.interactive_plot)  # Beispiel: 2 Worker

    # Ergebnisse anzeigen
    print("Optimierung mit SLSQP:")
    if result_slsqp and result_slsqp.success:
        print("Optimale Allokation:", result_slsqp.x)
        print("Maximaler Nutzen:", -result_slsqp.fun)
    else:
        print("Optimierung fehlgeschlagen oder kein Ergebnis verfügbar.")

    print("\nOptimierung mit Differential Evolution:")
    if result_de and result_de.success:
        print("Optimale Allokation:", result_de.x)
        print("Maximaler Nutzen:", -result_de.fun)
    else:
        print("Optimierung fehlgeschlagen oder kein Ergebnis verfügbar.")

    # Sensitivitätsanalyse: Variation des Budgets
    B_values = np.linspace(5, 20, 16)  # Budget von 5 bis 20 in Schritten von 1
    B_sens, allocations_sens, utilities_sens = optimizer.sensitivity_analysis(
        B_values, method='SLSQP'
    )
    optimizer.plot_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP', interactive=args.interactive_plot)

    # Optional: Interaktive Sensitivitätsdiagramme
    # optimizer.plot_interactive_sensitivity(B_values, allocations_sens, utilities_sens, method='SLSQP')

    # Identifizierung der wichtigsten Synergieeffekte
    top_synergies = optimizer.identify_top_synergies(top_n=6)
    print("\nWichtigste Synergieeffekte (sortiert):")
    for pair, value in top_synergies:
        print(f"Bereiche {investment_labels[pair[0]]} & {investment_labels[pair[1]]}: Synergieeffekt = {value}")

    # Robustheitsanalyse
    df_robustness = optimizer.robustness_analysis(num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=4)
    print("\nRobustheitsanalyse (Statistik der Investitionsallokationen):")
    print(df_robustness)

    # Visualisierung der Robustheitsanalyse
    optimizer.plot_robustness_analysis(df_robustness)

    # Zusätzliche Visualisierung: Heatmap der Synergien
    optimizer.plot_synergy_heatmap()

    # Beispiel für eine Sensitivitätsanalyse eines einzelnen Parameters
    parameter_values = np.linspace(1, 5, 10)
    # Sensitivität des ersten Effizienzparameters a[0]
    optimizer.plot_parameter_sensitivity(parameter_values, 'a', parameter_index=0, method='SLSQP', interactive=args.interactive_plot)
    # Sensitivität des Synergieelements b[0,1]
    optimizer.plot_parameter_sensitivity(parameter_values, 'b', parameter_index=(0,1), method='SLSQP', interactive=args.interactive_plot)

    # Weitere Visualisierungen können hier hinzugefügt werden


# ==========================
# Ausführung des Skripts
# ==========================
if __name__ == "__main__":
    main()
