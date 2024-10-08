import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, NonlinearConstraint
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

def validate_inputs(a, b, B, L, U, x0, epsilon_a=1e-6):
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
    epsilon_a : float, optional
        Mindestwert für Effizienzparameter. Standard ist 1e-6.

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
    if np.sum(U) < B:
        raise ValueError("Die Summe der Höchstinvestitionen U unterschreitet das Gesamtbudget B. Das Problem ist unlösbar.")
    if np.any(x0 < L):
        raise ValueError("Anfangsschätzung x0 unterschreitet mindestens eine Mindestinvestition L.")
    if np.any(x0 > U):
        raise ValueError("Anfangsschätzung x0 überschreitet mindestens eine Höchstinvestition U.")
    if np.any(L <= 0):
        raise ValueError("Alle Mindestinvestitionen L müssen größer als Null sein.")
    if np.any(a <= epsilon_a):
        raise ValueError(f"Alle Effizienzparameter a müssen größer als {epsilon_a} sein.")
    if np.any(b < 0):
        raise ValueError("Alle Synergieeffekte b müssen größer oder gleich Null sein.")

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

def compute_synergy(x, b):
    """
    Berechnet die Synergieeffekte zwischen den Investitionsbereichen.

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
    # Effizientere Berechnung mittels quadratischer Form
    return 0.5 * np.dot(x, np.dot(b, x))

def validate_initial_guess(x, L, U, B, tol=1e-6):
    """
    Validiert, ob die Anfangsschätzung die Nebenbedingungen erfüllt.

    Parameters:
    ----------
    x : np.array
        Anfangsschätzung.
    L : np.array
        Mindestinvestitionen.
    U : np.array
        Höchste Investitionen.
    B : float
        Gesamtbudget.
    tol : float, optional
        Toleranz für die Validierung. Standard ist 1e-6.

    Returns:
    -------
    bool
        True, wenn alle Bedingungen erfüllt sind, sonst False.
    """
    return (
        np.isclose(np.sum(x), B, atol=tol, rtol=tol) and
        np.all(x >= L - tol) and
        np.all(x <= U + tol)
    )

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

    Raises:
    ------
    ValueError
        Wenn Anpassung nicht möglich ist.
    """
    n = len(x0)

    # Überprüfen der Machbarkeit
    if np.sum(L) > B:
        raise ValueError("Die Summe der Mindestinvestitionen überschreitet das Gesamtbudget B. Das Problem ist unlösbar.")
    if np.sum(U) < B:
        raise ValueError("Die Summe der Höchstinvestitionen U unterschreitet das Gesamtbudget B. Das Problem ist unlösbar.")

    # Ziel: Minimierung der Abweichung von x0
    res = minimize(
        lambda x: np.sum((x - x0) ** 2),
        x0,
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - B}],
        bounds=get_bounds(L, U),
        method='SLSQP'
    )

    if res.success and validate_initial_guess(res.x, L, U, B):
        return res.x
    else:
        logging.warning("Lineare Programmierung zur Anpassung der Anfangsschätzung fehlgeschlagen. Verwende Heuristik.")
        # Fallback: Ursprungsschätzung proportional anpassen
        x0 = np.clip(x0, L, U)
        current_sum = np.sum(x0)
        deficit = B - current_sum
        if not np.isclose(current_sum, B, atol=1e-6, rtol=1e-6):
            if deficit > 0:
                # Erhöhen Sie die Investitionen proportional zu den verbleibenden Kapazitäten
                capacity = U - x0
                total_capacity = np.sum(capacity)
                if total_capacity > 1e-8:  # Strengere Prüfung
                    x0 += capacity * (deficit / total_capacity)
                else:
                    logging.error("Keine Kapazität zur Anpassung der Anfangsschätzung vorhanden.")
                    raise ValueError("Anpassung der Anfangsschätzung nicht möglich: Keine Kapazität zur Erhöhung.")
            else:
                # Reduzieren Sie die Investitionen proportional zu den überschüssigen Beträgen
                excess = x0 - L
                total_excess = np.sum(excess)
                if total_excess > 1e-8:  # Strengere Prüfung
                    x0 -= excess * (-deficit / total_excess)
                else:
                    logging.error("Keine überschüssigen Investitionen zur Anpassung vorhanden.")
                    raise ValueError("Anpassung der Anfangsschätzung nicht möglich: Keine überschüssigen Investitionen.")
            x0 = np.clip(x0, L, U)
            # Finaler Check
            if not validate_initial_guess(x0, L, U, B):
                logging.error("Angepasste Anfangsschätzung erfüllt die Nebenbedingungen nicht.")
                raise ValueError("Angepasste Anfangsschätzung erfüllt die Nebenbedingungen nicht.")
        return x0

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
# Externe Simulation-Funktion
# ==========================
def single_simulation(sim_index, a, b, B, L, U, x0, investment_labels, method, variation_percentage, **kwargs):
    """
    Führt eine einzelne Simulationsrunde für die Robustheitsanalyse durch.

    Parameters:
    ----------
    sim_index : int
        Index der Simulation.
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
    investment_labels : list of str
        Namen der Investitionsbereiche.
    method : str
        Optimierungsmethode.
    variation_percentage : float
        Prozentuale Variation der Parameter.
    **kwargs : dict
        Zusätzliche Parameter an die Optimierungsmethode.

    Returns:
    -------
    list or np.array
        Optimale Investitionsallokation oder [np.nan]*n bei Fehlern.
    """
    try:
        logging.info(f"Robustheitsanalyse Simulation {sim_index + 1} gestartet.")
        # Zufällige Variation der Parameter um ±variation_percentage
        a_sim = a * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=a.shape)
        # Dynamischer Mindestwert basierend auf ursprünglichem a
        epsilon_a = 1e-6
        a_sim = np.maximum(a_sim, epsilon_a)  # Verhindern kleiner oder negativer Effizienzparameter

        b_sim = b * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=b.shape)
        b_sim = np.maximum(b_sim, 0)  # Verhindern negativer Synergieeffekte
        # Symmetrisch machen unter Berücksichtigung der Diagonale
        b_sim = np.triu(b_sim, 1) + np.triu(b_sim, 1).T
        np.fill_diagonal(b_sim, 0)

        # Überprüfung der Synergieeffekte
        if not is_symmetric(b_sim):
            logging.warning(f"Synergiematrix in Simulation {sim_index + 1} ist nicht symmetrisch. Korrigiere Symmetrie.")
            b_sim = np.triu(b_sim, 1) + np.triu(b_sim, 1).T

        # Erstellen eines neuen Optimizers mit variierten Parametern
        try:
            optimizer_sim = InvestmentOptimizer(
                a=a_sim,
                b=b_sim,
                B=B,
                L=L,
                U=U,
                x0=x0,
                investment_labels=investment_labels,
                log_level=logging.CRITICAL  # Minimieren der Log-Ausgabe in Simulationen
            )
        except ValueError as e:
            logging.error(f"Validierungsfehler in Simulation {sim_index + 1}: {e}")
            return [np.nan] * len(a)

        # Optimierung durchführen
        result = optimizer_sim.optimize(method=method, workers=1, **kwargs)
        if result and result.success:
            logging.info(f"Robustheitsanalyse Simulation {sim_index + 1} erfolgreich abgeschlossen.")
            return result.x.tolist()
        else:
            logging.warning(f"Robustheitsanalyse Simulation {sim_index + 1} fehlgeschlagen oder kein Ergebnis verfügbar.")
            return [np.nan] * len(a)
    except Exception as e:
        logging.error(f"Unerwarteter Fehler in Simulation {sim_index + 1}: {e}")
        return [np.nan] * len(a)

# ==========================
# InvestmentOptimizer Klasse
# ==========================
class InvestmentOptimizer:
    def __init__(self, a, b, B, L, U, x0, investment_labels=None, log_level=logging.INFO, c=None):
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
        c : np.array, optional
            Parameter für quadratische Nutzenfunktion (abnehmende Grenzerträge).
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
        self.c = np.array(c) if c is not None else None  # Optionaler Parameter für quadratische Nutzenfunktion
        validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0)

    def update_parameters(self, a=None, b=None, B=None, L=None, U=None, x0=None, c=None):
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
            Neue Höchsteinvestitionen.
        x0 : np.array, optional
            Neue Anfangsschätzung.
        c : np.array, optional
            Neuer Parameter für quadratische Nutzenfunktion.
        
        Raises:
        ------
        ValueError
            Wenn inkonsistente Eingaben vorliegen.
        """
        if a is not None:
            self.a = np.array(a)
        if b is not None:
            b = np.array(b)
            if not is_symmetric(b):
                raise ValueError("Synergiematrix b muss symmetrisch sein.")
            if np.any(b < 0):
                raise ValueError("Alle Synergieeffekte b müssen größer oder gleich Null sein.")
            self.b = b
        if B is not None:
            self.B = B
        if L is not None:
            self.L = np.array(L)
        if U is not None:
            self.U = np.array(U)
        if x0 is not None:
            self.x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
        if c is not None:
            self.c = np.array(c)
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

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        """
        Berechnet den negativen Gesamtnutzen mit Straftermen für Budgetabweichungen.

        Parameters:
        ----------
        x : np.array
            Investitionen in die Bereiche.
        penalty_coeff : float, optional
            Strafkoeffizient für Budgetabweichungen. Standard ist 1e8.

        Returns:
        -------
        float
            Negativer Gesamtnutzen mit Straftermen.
        """
        # Berechnung des individuellen Nutzens
        if self.c is not None:
            utility = np.sum(self.a * x - 0.5 * self.c * x**2)
        else:
            utility = np.sum(self.a * np.log(x))
        
        # Berechnung der Synergieeffekte
        synergy = compute_synergy(x, self.b)
        
        # Gesamtnutzen
        total_utility = utility + synergy

        # Strafterm für Budgetabweichung (optional, wenn nicht strikt als Gleichung verwendet)
        budget_diff = np.sum(x) - self.B
        penalty_budget = penalty_coeff * (budget_diff ** 2)

        # Negative Nutzen für Minimierung
        return -total_utility + penalty_budget

    def optimize(self, method='SLSQP', max_retries=3, workers=None, **kwargs):
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
            - 'TNC': Truncated Newton Conjugate-Gradient
            Weitere Methoden können durch Erweiterung der Optimierungsmethoden hinzugefügt werden.
            Standard ist 'SLSQP'.
        max_retries : int, optional
            Die maximale Anzahl an Wiederholungsversuchen bei Optimierungsfehlern. Standard ist 3.
        workers : int, optional
            Die Anzahl der parallelen Worker für Methoden wie Differential Evolution. 
            Standard ist 1.
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
        bounds = get_bounds(self.L, self.U)

        optimization_methods = {
            'SLSQP': lambda: minimize(
                self.objective_with_penalty,
                self.x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - self.B}  # Summe(x) == B
                ],
                options={'disp': False, 'maxiter': 1000},
                args=(1e8,)  # penalty_coeff
            ),
            'DE': lambda: differential_evolution(
                self.objective_with_penalty,
                bounds,
                strategy=kwargs.get('strategy', 'best1bin'),
                maxiter=kwargs.get('maxiter', 1000),
                tol=kwargs.get('tol', 1e-8),
                updating='deferred',
                workers=workers if workers is not None else 1,
                polish=True,
                init='latinhypercube',
                args=(1e8,)
            ),
            'BasinHopping': lambda: basinhopping(
                self.objective_with_penalty,
                self.x0,
                niter=kwargs.get('niter', 100),
                stepsize=kwargs.get('stepsize', 0.5),
                minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds, 'constraints': [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - self.B}
                ], 'options': {'maxiter': 1000, 'disp': False}},
                stepsize=kwargs.get('stepsize', 0.5),
                T=kwargs.get('T', 1.0),
                niter_success=kwargs.get('niter_success', 10)
            ),
            'TNC': lambda: minimize(
                self.objective_with_penalty,
                self.x0,
                method='TNC',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000},
                args=(1e8,)
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
                if method == 'BasinHopping':
                    # Bei BasinHopping ist das Ergebnis anders strukturiert
                    x_opt = optimization_result.x
                    fun_opt = optimization_result.fun
                    success = optimization_result.lowest_optimization_result.success
                    message = optimization_result.lowest_optimization_result.message
                elif method == 'DE':
                    x_opt = optimization_result.x
                    fun_opt = optimization_result.fun
                    success = optimization_result.success
                    message = optimization_result.message if hasattr(optimization_result, 'message') else 'Optimierung abgeschlossen.'
                else:
                    x_opt = optimization_result.x
                    fun_opt = optimization_result.fun
                    success = optimization_result.success
                    message = optimization_result.message if hasattr(optimization_result, 'message') else 'Optimierung abgeschlossen.'

                # Überprüfung der Nebenbedingungen mit strengeren Toleranzen
                constraints_satisfied = (
                    np.isclose(np.sum(x_opt), self.B, atol=1e-8, rtol=1e-8) and
                    np.all(x_opt >= self.L - 1e-8) and
                    np.all(x_opt <= self.U + 1e-8)
                )

                if success and constraints_satisfied:
                    logging.info(f"Optimierung mit {method} erfolgreich abgeschlossen beim Versuch {attempt}.")
                    result = OptimizationResult(
                        x=x_opt,
                        fun=fun_opt,
                        success=success,
                        message=message
                    )
                    return result
                else:
                    logging.warning(f"Optimierung mit {method} fehlgeschlagen beim Versuch {attempt}: {message}")
            except Exception as e:
                logging.error(f"{method} Optimierungsfehler beim Versuch {attempt}: {e}")
                result = OptimizationResult(
                    x=None,
                    fun=None,
                    success=False,
                    message=str(e)
                )

        logging.error(f"Optimierung mit {method} nach {max_retries} Versuchen fehlgeschlagen.")
        return None

    def sensitivity_analysis(self, B_values, method='SLSQP', tol=1e-6, **kwargs):
        """
        Führt eine Sensitivitätsanalyse durch, indem das Budget variiert und die optimale Allokation berechnet wird.

        Parameters:
        ----------
        B_values : np.array
            Verschiedene Budgetwerte.
        method : str, optional
            Optimierungsmethode. Standard ist 'SLSQP'.
        tol : float, optional
            Toleranz für die Validierung. Standard ist 1e-6.
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
            try:
                # Anpassen des Budgets und Anfangsschätzung
                optimizer_copy = InvestmentOptimizer(
                    a=self.a,
                    b=self.b,
                    B=B,
                    L=self.L,
                    U=self.U,
                    x0=self.x0,
                    investment_labels=self.investment_labels,
                    log_level=logging.CRITICAL  # Minimieren der Log-Ausgabe in Sensitivitätsanalyse
                )
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
        results = []
        a = self.a
        b = self.b
        B = self.B
        L = self.L
        U = self.U
        x0 = self.x0
        investment_labels = self.investment_labels

        if parallel:
            if num_workers is None:
                num_workers = min(4, os.cpu_count() or 1)  # Begrenzung auf maximal 4 Worker
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            single_simulation, sim, a, b, B, L, U, x0,
                            investment_labels, method, variation_percentage, **kwargs
                        ) for sim in range(num_simulations)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logging.error(f"Fehler in Simulation: {e}")
                            results.append([np.nan] * len(a))
            except Exception as e:
                logging.error(f"Fehler bei der parallelen Verarbeitung: {e}")
                logging.info("Wechsle zu sequenzieller Verarbeitung.")
                for sim in range(num_simulations):
                    results.append(single_simulation(sim, a, b, B, L, U, x0, investment_labels, method, variation_percentage, **kwargs))
        else:
            for sim in range(num_simulations):
                results.append(single_simulation(sim, a, b, B, L, U, x0, investment_labels, method, variation_percentage, **kwargs))

        # Analysieren der Ergebnisse
        df_results = pd.DataFrame(results, columns=investment_labels)
        # Verwendung von describe mit angepassten Perzentilen
        additional_stats = df_results.describe(percentiles=[0.25, 0.75]).transpose()
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
            Ob interaktive Plotly-Plots erstellt werden sollen. Standard ist False (Matplotlib).
        """
        try:
            if interactive:
                # Plotly nutzen
                df_alloc = pd.DataFrame(optimal_allocations, columns=self.investment_labels)
                df_alloc['Budget'] = B_values
                df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': max_utilities})

                fig1 = px.line(df_alloc, x='Budget', y=self.investment_labels, title='Optimale Investitionsallokationen')
                fig1.show()

                fig2 = px.line(df_util, x='Budget', y='Maximaler Nutzen', title='Maximaler Nutzen bei verschiedenen Budgets')
                fig2.show()
            else:
                # Matplotlib nutzen
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
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Sensitivitätsanalyse: {e}")

    def plot_synergy_heatmap(self):
        """
        Plotet eine Heatmap der Synergieeffekte.
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.b, annot=True, xticklabels=self.investment_labels, yticklabels=self.investment_labels, cmap='viridis')
            plt.title('Heatmap der Synergieeffekte')
            plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Synergie-Heatmap: {e}")

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
            Ob interaktive Plotly-Plots erstellt werden sollen. Standard ist False (Matplotlib).
        **kwargs : dict
            Zusätzliche Parameter an die Optimierungsmethode.
        """
        utilities = []
        allocations = []

        for value in parameter_values:
            try:
                optimizer_copy = InvestmentOptimizer(
                    a=self.a.copy(),
                    b=self.b.copy(),
                    B=self.B,
                    L=self.L.copy(),
                    U=self.U.copy(),
                    x0=self.x0.copy(),
                    investment_labels=self.investment_labels,
                    log_level=logging.CRITICAL  # Minimieren der Log-Ausgabe
                )
                if parameter_name == 'a':
                    if not isinstance(parameter_index, int):
                        raise ValueError("Für Parameter 'a' muss parameter_index ein int sein.")
                    epsilon_a = 1e-6
                    if value <= epsilon_a:
                        raise ValueError(f"Effizienzparameter a muss größer als {epsilon_a} sein.")
                    a_new = optimizer_copy.a.copy()
                    a_new[parameter_index] = value
                    optimizer_copy.update_parameters(a=a_new)
                elif parameter_name == 'b':
                    if not (isinstance(parameter_index, tuple) and len(parameter_index) == 2):
                        raise ValueError("Für Parameter 'b' muss parameter_index ein tuple mit zwei Indizes sein, z.B. (0,1).")
                    if value < 0:
                        raise ValueError("Synergieeffekte b müssen größer oder gleich Null sein.")
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
            except Exception as e:
                logging.error(f"Fehler bei der Sensitivitätsanalyse für Wert {value}: {e}")
                utilities.append(np.nan)
                allocations.append([np.nan] * self.n)

        try:
            if interactive:
                # Plotly nutzen
                if parameter_name == 'a':
                    xlabel = f'a[{parameter_index}]'
                elif parameter_name == 'b':
                    xlabel = f'b[{parameter_index[0]},{parameter_index[1]}]'
                else:
                    xlabel = parameter_name

                df_util = pd.DataFrame({'Parameter Value': parameter_values, 'Maximaler Nutzen': utilities})
                fig = px.line(df_util, x='Parameter Value', y='Maximaler Nutzen',
                              title=f'Sensitivität des Nutzens gegenüber {xlabel}')
                fig.show()
            else:
                # Matplotlib nutzen
                if parameter_name == 'a':
                    xlabel = f'a[{parameter_index}]'
                elif parameter_name == 'b':
                    xlabel = f'b[{parameter_index[0]},{parameter_index[1]}]'
                else:
                    xlabel = parameter_name

                plt.figure(figsize=(10, 6))
                plt.plot(parameter_values, utilities, marker='o')
                plt.xlabel(xlabel)
                plt.ylabel('Maximaler Nutzen')
                plt.title(f'Sensitivität des Nutzens gegenüber {xlabel}')
                plt.grid(True)
                plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Parameter-Sensitivität: {e}")

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
        try:
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
        except Exception as e:
            logging.error(f"Fehler beim Erstellen des interaktiven Sensitivitäts-Dashboards: {e}")

    def plot_robustness_analysis(self, df_results):
        """
        Plotet die Verteilung der Investitionsallokationen aus der Robustheitsanalyse.

        Parameters:
        ----------
        df_results : pandas.DataFrame
            Ergebnisse der Robustheitsanalyse.
        """
        try:
            # Entfernen von NaN-Werten und Informieren über Anzahl der fehlgeschlagenen Simulationen
            num_failed = df_results.isna().any(axis=1).sum()
            if num_failed > 0:
                logging.warning(f"{num_failed} Simulationen sind fehlgeschlagen und werden in den Plots ausgeschlossen.")
            df_clean = df_results.dropna()
            if df_clean.empty:
                logging.warning("Keine gültigen Daten zum Plotten vorhanden.")
                return

            # Boxplot
            df_melted = df_clean.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
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
            sns.pairplot(df_clean)
            plt.suptitle('Scatter-Plots der Investitionsallokationen', y=1.02)
            plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Robustheitsanalyse: {e}")

    def plot_additional_synergy_heatmap(self):
        """
        Plotet eine zusätzliche Heatmap der Synergieeffekte (optional, falls benötigt).
        """
        self.plot_synergy_heatmap()

# ==========================
# Hauptfunktion
# ==========================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Investment Optimizer')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Verwenden Sie interaktive Plotly-Plots')
    # Weitere Argumente können hinzugefügt werden
    return parser.parse_args()

def main():
    """
    Hauptfunktion zur Ausführung der Investitionsoptimierung.
    """
    args = parse_arguments()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Überprüfung, ob alle notwendigen Schlüssel vorhanden sind
            required_keys = ['a', 'b', 'B', 'L', 'U', 'x0']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Schlüssel '{key}' fehlt in der Konfigurationsdatei.")
            investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(len(config['a']))])
            a = np.array(config['a'])
            b = np.array(config['b'])
            B = config['B']
            L = np.array(config['L'])
            U = np.array(config['U'])
            x0 = np.array(config['x0'])
            c = np.array(config['c']) if 'c' in config else None  # Optionaler Parameter für quadratische Nutzenfunktion
        except KeyError as e:
            logging.error(f"Fehlender Schlüssel in der Konfigurationsdatei: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logging.error(f"Fehler beim Parsen der JSON-Konfigurationsdatei: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfigurationsdatei: {e}")
            sys.exit(1)
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
        c = np.array([0.1, 0.1, 0.1, 0.1])  # Beispielwerte für quadratische Nutzenfunktion

    # Initialisieren des Optimizers mit DEBUG-Logging (kann angepasst werden)
    try:
        optimizer = InvestmentOptimizer(a, b, B, L, U, x0, investment_labels=investment_labels, log_level=logging.DEBUG, c=c)
    except ValueError as e:
        logging.error(f"Initialisierungsfehler: {e}")
        sys.exit(1)

    # Optimierung mit SLSQP
    result_slsqp = optimizer.optimize(method='SLSQP')

    # Optimierung mit Differential Evolution (globale Optimierung)
    result_de = optimizer.optimize(method='DE', workers=2)  # Beispiel: 2 Worker

    # Ergebnisse anzeigen
    print("Optimierung mit SLSQP:")
    if result_slsqp is not None and result_slsqp.success:
        print("Optimale Allokation:", result_slsqp.x)
        print("Maximaler Nutzen:", -result_slsqp.fun)
    else:
        print("Optimierung fehlgeschlagen oder kein Ergebnis verfügbar.")

    print("\nOptimierung mit Differential Evolution:")
    if result_de is not None and result_de.success:
        print("Optimale Allokation:", result_de.x)
        print("Maximaler Nutzen:", -result_de.fun)
    else:
        print("Optimierung fehlgeschlagen oder kein Ergebnis verfügbar.")

    # Sensitivitätsanalyse: Variation des Budgets
    B_values = np.arange(5, 21, 1)  # Budget von 5 bis 20 in Schritten von 1
    B_sens, allocations_sens, utilities_sens = optimizer.sensitivity_analysis(
        B_values, method='SLSQP'
    )
    optimizer.plot_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP', interactive=args.interactive)

    # Optional: Interaktive Sensitivitätsdiagramme
    # optimizer.plot_interactive_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP')

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
    optimizer.plot_parameter_sensitivity(parameter_values, 'a', parameter_index=0, method='SLSQP', interactive=args.interactive)
    # Sensitivität des Synergieelements b[0,1]
    optimizer.plot_parameter_sensitivity(parameter_values, 'b', parameter_index=(0, 1), method='SLSQP', interactive=args.interactive)

    # Weitere Visualisierungen können hier hinzugefügt werden

# ==========================
# Ausführung des Skripts
# ==========================
if __name__ == "__main__":
    main()
