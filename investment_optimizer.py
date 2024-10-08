import sys
import logging
import concurrent.futures
import copy
import argparse
import json
import os

# Überprüfen der erforderlichen Pakete
required_packages = [
    'numpy', 'scipy', 'matplotlib', 'pandas',
    'seaborn', 'plotly', 'packaging'
]

missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"Fehlende Pakete: {', '.join(missing_packages)}. Bitte installieren Sie sie mit pip.")
    sys.exit(1)

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
try:
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

from packaging import version
import scipy

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

def validate_inputs(a, b, B, L, U, x0, c=None, epsilon_a=1e-6):
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
    if c is not None:
        c = np.array(c)
        if c.shape != a.shape:
            raise ValueError("Parameter 'c' muss die gleiche Form wie 'a' haben.")
        if np.any(c < 0):
            raise ValueError("Alle Werte in 'c' müssen größer oder gleich Null sein.")

def get_bounds(L, U):
    return [(max(L[i], 1e-8), U[i]) for i in range(len(L))]  # Problem 8: Untergrenzen angepasst

def compute_synergy(x, b):
    return 0.5 * np.dot(x, np.dot(b, x))

def adjust_initial_guess(x0, L, U, B):
    n = len(x0)
    if np.sum(L) > B or np.sum(U) < B:
        raise ValueError("Das Problem ist unlösbar aufgrund der Investitionsgrenzen.")
    res = minimize(
        lambda x: np.sum((x - x0) ** 2),
        x0,
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - B}],
        bounds=get_bounds(L, U),
        method='SLSQP'
    )
    if res.success and np.isclose(np.sum(res.x), B) and np.all(res.x >= L) and np.all(res.x <= U):
        return res.x
    x0 = np.clip(x0, L, U)
    current_sum = np.sum(x0)
    deficit = B - current_sum
    if deficit > 0:
        capacity = U - x0
        total_capacity = np.sum(capacity)
        if total_capacity > 1e-8:
            x0 += capacity * (deficit / total_capacity)
        else:
            # Gleichmäßige Verteilung des Defizits unter Berücksichtigung der Grenzen
            addition = deficit / n
            x0 += addition
    else:
        excess = x0 - L
        total_excess = np.sum(excess)
        if total_excess > 1e-8:
            x0 -= excess * (-deficit / total_excess)
        else:
            # Gleichmäßige Verteilung des Defizits
            x0 -= (-deficit / n)
    x0 = np.clip(x0, L, U)
    if not np.isclose(np.sum(x0), B, atol=1e-6):  # Problem 4: Toleranz erhöht
        x0 *= B / np.sum(x0)
        x0 = np.clip(x0, L, U)
    if np.isclose(np.sum(x0), B, atol=1e-6) and np.all(x0 >= L) and np.all(x0 <= U):
        return x0
    raise ValueError("Anpassung der Anfangsschätzung nicht möglich.")

class OptimizationResult:
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)

def single_simulation(sim_index, a, b, B, L, U, x0, investment_labels, method, variation_percentage, optimizer_class):
    try:
        a_sim = a * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=a.shape)
        a_sim = np.maximum(a_sim, 1e-6)
        b_sim = b * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=b.shape)
        b_sim = np.triu(b_sim, 1) + np.triu(b_sim, 1).T
        optimizer_sim = optimizer_class(a=a_sim, b=b_sim, B=B, L=L, U=U, x0=x0,
                                       investment_labels=investment_labels, log_level=logging.CRITICAL, c=None)
        result = optimizer_sim.optimize(method=method)
        if result and result.success:
            return result.x.tolist()
        else:
            return [np.nan] * len(a)
    except Exception as e:
        logging.error(f"Simulation {sim_index} fehlgeschlagen: {e}", exc_info=True)  # Problem 5: Detaillierte Ausnahme
        return [np.nan] * len(a)

class InvestmentOptimizer:
    def __init__(self, a, b, B, L, U, x0, investment_labels=None, log_level=logging.INFO, c=None):
        configure_logging(log_level)
        try:
            self.a = np.array(a)
            self.b = np.array(b)
            self.B = B
            self.L = np.array(L)
            self.U = np.array(U)
            self.n = len(a)
            self.investment_labels = investment_labels if investment_labels else [f'Bereich_{i}' for i in range(self.n)]
            self.x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
            self.c = np.array(c) if c is not None else None
            validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0, self.c)  # Problem 9: Validierung von c
        except Exception as e:
            logging.error(f"Fehler bei der Initialisierung: {e}", exc_info=True)
            raise

    def update_parameters(self, a=None, b=None, B=None, L=None, U=None, x0=None, c=None):
        try:
            parameters_updated = False
            if a is not None:
                self.a = np.array(a)
                parameters_updated = True
            if b is not None:
                b = np.array(b)
                if not is_symmetric(b):
                    raise ValueError("Synergiematrix b muss symmetrisch sein.")
                if np.any(b < 0):
                    raise ValueError("Alle Synergieeffekte b müssen größer oder gleich Null sein.")
                self.b = b
                parameters_updated = True
            if B is not None:
                self.B = B
                parameters_updated = True
            if L is not None:
                self.L = np.array(L)
                parameters_updated = True
            if U is not None:
                self.U = np.array(U)
                parameters_updated = True
            if x0 is not None:
                self.x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
            elif parameters_updated:
                # x0 neu anpassen, wenn L, U oder B geändert wurden
                self.x0 = adjust_initial_guess(self.x0, self.L, self.U, self.B)
            if c is not None:
                self.c = np.array(c)
            validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0, self.c)
        except Exception as e:
            logging.error(f"Fehler beim Aktualisieren der Parameter: {e}", exc_info=True)
            raise

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        synergy = compute_synergy(x, self.b)
        if self.c is not None:
            utility = np.sum(self.a * x - 0.5 * self.c * x**2)
        else:
            utility = np.sum(self.a * np.log(x))  # Problem 8: np.maximum entfernt
        total_utility = utility + synergy
        budget_diff = np.sum(x) - self.B
        penalty_budget = penalty_coeff * (budget_diff ** 2)
        return -total_utility + penalty_budget

    def objective_without_penalty(self, x):
        synergy = compute_synergy(x, self.b)
        if self.c is not None:
            utility = np.sum(self.a * x - 0.5 * self.c * x**2)
        else:
            utility = np.sum(self.a * np.log(x))
        total_utility = utility + synergy
        return -total_utility  # Problem 1: Separate Zielfunktion ohne Strafterm

    def identify_top_synergies_correct(self, top_n=6):  # Problem 2: Angepasste Funktion
        try:
            b_copy = self.b.copy()
            np.fill_diagonal(b_copy, 0)
            # Extrahieren der oberen Dreiecksmatrix ohne Diagonale
            triu_indices = np.triu_indices(self.n, k=1)
            synergy_values = b_copy[triu_indices]
            # Finden der Indizes der größten Synergien
            top_indices = np.argpartition(synergy_values, -top_n)[-top_n:]
            top_synergies = []
            for idx in top_indices:
                i = triu_indices[0][idx]
                j = triu_indices[1][idx]
                top_synergies.append(((i, j), b_copy[i, j]))
            # Sortieren der Top-Synergien
            top_synergies.sort(key=lambda x: x[1], reverse=True)
            return top_synergies
        except Exception as e:
            logging.error(f"Fehler beim Identifizieren der Top-Synergien: {e}", exc_info=True)
            return []

    def optimize(self, method='SLSQP', max_retries=3, workers=None, **kwargs):
        bounds = get_bounds(self.L, self.U)
        scipy_version = version.parse(scipy.__version__)
        de_workers_supported = scipy_version >= version.parse("1.4.0")  # Anpassung je nach benötigter Version

        # Korrektur für den updating-Parameter
        if method == 'DE':
            updating = 'deferred' if workers is not None and workers != 1 and de_workers_supported else 'immediate'

        # Problem 1: Dynamische Zielfunktion basierend auf der Methode
        use_penalty = method in ['DE', 'TNC']  # Methoden, die keine Constraints unterstützen

        optimization_methods = {
            'SLSQP': lambda: minimize(
                self.objective_without_penalty if not use_penalty else self.objective_with_penalty,
                self.x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - self.B}
                ] if not use_penalty else [],
                options={'disp': False, 'maxiter': 1000},
                args=() if not use_penalty else (1e8,)
            ),
            'DE': lambda: differential_evolution(
                self.objective_with_penalty,
                bounds,
                strategy=kwargs.get('strategy', 'best1bin'),
                maxiter=kwargs.get('maxiter', 1000),
                tol=kwargs.get('tol', 1e-8),
                updating=updating,  # Korrektur angewendet
                workers=workers if workers is not None and de_workers_supported else 1,
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
        }
        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods:
            logging.error(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden sind: {supported_methods}")
            raise ValueError(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden sind: {supported_methods}")
        for attempt in range(1, max_retries + 1):
            try:
                optimization_result = optimization_methods[method]()
                if method == 'BasinHopping':
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
                constraints_satisfied = (
                    np.isclose(np.sum(x_opt), self.B, atol=1e-6, rtol=1e-6) and  # Problem 4: Toleranz angepasst
                    np.all(x_opt >= self.L - 1e-8) and
                    np.all(x_opt <= self.U + 1e-8)
                )
                if success and constraints_satisfied:
                    return OptimizationResult(
                        x=x_opt,
                        fun=fun_opt,
                        success=success,
                        message=message
                    )
            except Exception as e:
                logging.error(f"Optimierungsversuch {attempt} mit Methode {method} fehlgeschlagen: {e}", exc_info=True)
        return None

    def sensitivity_analysis(self, B_values, method='SLSQP', tol=1e-6, **kwargs):
        allocations = []
        max_utilities = []
        for B in B_values:
            try:
                optimizer_copy = InvestmentOptimizer(
                    a=self.a,
                    b=self.b,
                    B=B,
                    L=self.L,
                    U=self.U,
                    x0=self.x0 * (B / self.B),  # Problem 6: x0 angepasst
                    investment_labels=self.investment_labels,
                    log_level=logging.CRITICAL,
                    c=self.c
                )
                result = optimizer_copy.optimize(method=method, **kwargs)
                if result and result.success:
                    allocations.append(result.x)
                    max_utilities.append(-result.fun)
                else:
                    allocations.append([np.nan] * self.n)
                    max_utilities.append(np.nan)
            except Exception as e:
                logging.error(f"Sensitivitätsanalyse für Budget {B} fehlgeschlagen: {e}", exc_info=True)
                allocations.append([np.nan] * self.n)
                max_utilities.append(np.nan)
        return B_values, allocations, max_utilities

    def robustness_analysis(self, num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=None):
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
                num_workers = min(4, os.cpu_count() or 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  # Problem 10: ThreadPoolExecutor verwendet
                futures = [
                    executor.submit(
                        single_simulation, sim, a, b, B, L, U, x0,
                        investment_labels, method, variation_percentage, InvestmentOptimizer
                    ) for sim in range(num_simulations)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Robustheitsanalyse Simulation fehlgeschlagen: {e}", exc_info=True)
                        results.append([np.nan] * len(a))
        else:
            for sim in range(num_simulations):
                try:
                    result = single_simulation(sim, a, b, B, L, U, x0, investment_labels, method, variation_percentage, InvestmentOptimizer)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Robustheitsanalyse Simulation fehlgeschlagen: {e}", exc_info=True)
                    results.append([np.nan] * len(a))
        df_results = pd.DataFrame(results, columns=investment_labels)
        additional_stats = df_results.describe(percentiles=[0.25, 0.75]).transpose()
        return additional_stats, df_results

    def multi_criteria_optimization(self, alpha, beta, gamma, risk_func, sustainability_func, method='SLSQP'):
        def objective(x):
            synergy = compute_synergy(x, self.b)
            if self.c is not None:
                utility = np.sum(self.a * x - 0.5 * self.c * x**2)
            else:
                utility = np.sum(self.a * np.log(x))
            total_utility = utility + synergy
            risk = beta * risk_func(x)
            sustainability = gamma * sustainability_func(x)
            return -(alpha * total_utility - risk - sustainability)
        bounds = get_bounds(self.L, self.U)
        try:
            result = minimize(
                objective,
                self.x0,
                method=method,
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - self.B}],
                options={'disp': False, 'maxiter': 1000}
            )
            if result.success:
                return result
            else:
                logging.error(f"Multikriterielle Optimierung fehlgeschlagen: {result.message}")
                return None
        except Exception as e:
            logging.error(f"Multikriterielle Optimierung fehlgeschlagen: {e}", exc_info=True)
            return None

    def plot_sensitivity(self, B_values, allocations, utilities, method='SLSQP', interactive=False):
        try:
            if interactive:
                if plotly_available:
                    df_alloc = pd.DataFrame(allocations, columns=self.investment_labels)
                    df_alloc['Budget'] = B_values
                    df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})
                    fig1 = px.line(df_alloc, x='Budget', y=self.investment_labels, title='Optimale Investitionsallokationen')
                    fig1.show()
                    fig2 = px.line(df_util, x='Budget', y='Maximaler Nutzen', title='Maximaler Nutzen bei verschiedenen Budgets')
                    fig2.show()
                else:
                    logging.warning("Plotly ist nicht installiert. Wechsel zu statischen Plots.")
                    interactive = False  # Fallback auf statische Plots
            if not interactive:
                df_alloc = pd.DataFrame(allocations, columns=self.investment_labels)
                df_alloc.fillna(0, inplace=True)
                df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})
                fig, ax1 = plt.subplots(figsize=(12, 8))
                colors = sns.color_palette("tab10", n_colors=self.n)
                for i, label in enumerate(self.investment_labels):
                    ax1.plot(B_values, df_alloc[label], label=label, color=colors[i], alpha=0.7)
                ax1.set_xlabel('Budget B')
                ax1.set_ylabel('Investitionsbetrag x_i')
                ax1.legend(loc='upper left')
                ax1.grid(True)
                ax2 = ax1.twinx()
                ax2.plot(B_values, df_util['Maximaler Nutzen'], label='Maximaler Nutzen', color='tab:red', marker='o')
                ax2.set_ylabel('Maximaler Nutzen')
                ax2.legend(loc='upper right')
                plt.title(f'Optimale Investitionsallokation und Nutzen bei verschiedenen Budgets ({method})')
                plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Sensitivitätsanalyse: {e}", exc_info=True)

    def plot_robustness_analysis(self, df_results):
        try:
            num_failed = df_results.isna().any(axis=1).sum()
            if num_failed > 0:
                logging.warning(f"{num_failed} Simulationen sind fehlgeschlagen und werden in den Plots ausgeschlossen.")
            df_clean = df_results.dropna()
            if df_clean.empty:
                logging.warning("Keine gültigen Daten zum Plotten vorhanden.")
                return
            df_melted = df_clean.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Bereich', y='Investition', data=df_melted)
            plt.xlabel('Investitionsbereich')
            plt.ylabel('Investitionsbetrag')
            plt.title('Verteilung der Investitionsallokationen aus der Robustheitsanalyse')
            plt.show()
            g = sns.FacetGrid(df_melted, col="Bereich", col_wrap=4, sharex=False, sharey=False)
            g.map(sns.histplot, "Investition", kde=True, bins=20, color='skyblue')
            g.fig.suptitle('Histogramme der Investitionsallokationen', y=1.02)
            plt.show()
            g_pair = sns.pairplot(df_clean)
            g_pair.fig.suptitle('Scatter-Plots der Investitionsallokationen', y=1.02)
            plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Robustheitsanalyse: {e}", exc_info=True)

    def plot_synergy_heatmap(self):
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.b, annot=True, xticklabels=self.investment_labels, yticklabels=self.investment_labels, cmap='viridis')
            plt.title('Heatmap der Synergieeffekte')
            plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Synergie-Heatmap: {e}", exc_info=True)

    def plot_parameter_sensitivity(self, parameter_values, parameter_name, parameter_index=None, method='SLSQP', interactive=False):
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
                    log_level=logging.CRITICAL,
                    c=self.c
                )
                if parameter_name == 'a':
                    if not isinstance(parameter_index, int):
                        raise ValueError("Für Parameter 'a' muss parameter_index ein int sein.")
                    if value <= 1e-6:
                        raise ValueError("Effizienzparameter a muss größer als 1e-6 sein.")
                    a_new = optimizer_copy.a.copy()
                    a_new[parameter_index] = value
                    optimizer_copy.update_parameters(a=a_new)
                elif parameter_name == 'b':
                    if not (isinstance(parameter_index, tuple) and len(parameter_index) == 2):
                        raise ValueError("Für Parameter 'b' muss parameter_index ein tuple mit zwei Indizes sein.")
                    if value < 0:
                        raise ValueError("Synergieeffekte b müssen größer oder gleich Null sein.")
                    i, j = parameter_index
                    b_new = optimizer_copy.b.copy()
                    b_new[i, j] = value
                    b_new[j, i] = value
                    optimizer_copy.update_parameters(b=b_new)
                else:
                    raise ValueError("Unbekannter Parametername.")
                result = optimizer_copy.optimize(method=method)
                if result and result.success:
                    utilities.append(-result.fun)
                    allocations.append(result.x)
                else:
                    utilities.append(np.nan)
                    allocations.append([np.nan] * self.n)
            except Exception as e:
                logging.error(f"Sensitivitätsanalyse für {parameter_name} mit Wert {value} fehlgeschlagen: {e}", exc_info=True)
                utilities.append(np.nan)
                allocations.append([np.nan] * self.n)
        try:
            xlabel = ""
            if parameter_name == 'a':
                xlabel = f'a[{parameter_index}]'
            elif parameter_name == 'b':
                xlabel = f'b[{parameter_index[0]},{parameter_index[1]}]'
            else:
                xlabel = parameter_name
            if interactive:
                if plotly_available:
                    df_util = pd.DataFrame({'Parameter Value': parameter_values, 'Maximaler Nutzen': utilities})
                    fig = px.line(df_util, x='Parameter Value', y='Maximaler Nutzen',
                                  title=f'Sensitivität des Nutzens gegenüber {xlabel}')
                    fig.show()
                else:
                    logging.warning("Plotly ist nicht installiert. Wechsel zu statischen Plots.")
                    interactive = False  # Fallback auf statische Plots
            if not interactive:
                plt.figure(figsize=(10, 6))
                plt.plot(parameter_values, utilities, marker='o')
                plt.xlabel(xlabel)
                plt.ylabel('Maximaler Nutzen')
                plt.title(f'Sensitivität des Nutzens gegenüber {xlabel}')
                plt.grid(True)
                plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Parametersensitivität: {e}", exc_info=True)

    def identify_top_synergies(self, top_n=6):
        return self.identify_top_synergies_correct(top_n=top_n)  # Problem 2: Angepasste Funktion verwenden

def parse_arguments():
    parser = argparse.ArgumentParser(description='Investment Optimizer')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Verwenden Sie interaktive Plotly-Plots')
    return parser.parse_args()

def main():
    args = parse_arguments()
    configure_logging(logging.DEBUG)  # Log-Level auf DEBUG gesetzt
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Validierungen hinzufügen (Problem 7)
            required_keys = ['a', 'b', 'B', 'L', 'U', 'x0']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Schlüssel '{key}' fehlt in der Konfigurationsdatei.")
            # Weitere Validierungen
            if not isinstance(config['a'], list) or not all(isinstance(i, (int, float)) for i in config['a']):
                raise ValueError("Parameter 'a' muss eine Liste von Zahlen sein.")
            if not isinstance(config['b'], list) or not all(isinstance(row, list) for row in config['b']):
                raise ValueError("Parameter 'b' muss eine 2D-Liste sein.")
            if not isinstance(config['L'], list) or not all(isinstance(i, (int, float)) for i in config['L']):
                raise ValueError("Parameter 'L' muss eine Liste von Zahlen sein.")
            if not isinstance(config['U'], list) or not all(isinstance(i, (int, float)) for i in config['U']):
                raise ValueError("Parameter 'U' muss eine Liste von Zahlen sein.")
            if not isinstance(config['x0'], list) or not all(isinstance(i, (int, float)) for i in config['x0']):
                raise ValueError("Parameter 'x0' muss eine Liste von Zahlen sein.")
            investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(len(config['a']))])
            a = np.array(config['a'])
            b = np.array(config['b'])
            B = config['B']
            L = np.array(config['L'])
            U = np.array(config['U'])
            x0 = np.array(config['x0'])
            c = np.array(config['c']) if 'c' in config else None
        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfiguration: {e}", exc_info=True)
            sys.exit(1)
    else:
        investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        n = len(investment_labels)
        a = np.array([1, 2, 3, 4])
        b = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0]
        ])
        B = 10.0
        L = np.array([1] * n)
        U = np.array([5] * n)
        x0 = np.array([2, 2, 2, 4])
        c = np.array([0.1, 0.1, 0.1, 0.1])
    try:
        optimizer = InvestmentOptimizer(a, b, B, L, U, x0, investment_labels=investment_labels, log_level=logging.DEBUG, c=c)
    except Exception:
        sys.exit(1)

    # Optimierung mit SLSQP
    result_slsqp = optimizer.optimize(method='SLSQP')
    # Optimierung mit Differential Evolution
    result_de = optimizer.optimize(method='DE', workers=2)

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

    # Sensitivitätsanalyse für verschiedene Budgets
    B_values = np.arange(5, 21, 1)
    B_sens, allocations_sens, utilities_sens = optimizer.sensitivity_analysis(
        B_values, method='SLSQP'
    )
    optimizer.plot_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP', interactive=args.interactive)

    # Identifikation der Top-Synergien
    top_synergies = optimizer.identify_top_synergies(top_n=6)
    print("\nWichtigste Synergieeffekte (sortiert):")
    for pair, value in top_synergies:
        print(f"Bereiche {investment_labels[pair[0]]} & {investment_labels[pair[1]]}: Synergieeffekt = {value}")

    # Robustheitsanalyse
    df_robustness_stats, df_robustness = optimizer.robustness_analysis(num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=4)
    print("\nRobustheitsanalyse (Statistik der Investitionsallokationen):")
    print(df_robustness_stats)
    optimizer.plot_robustness_analysis(df_robustness)

    # Multikriterielle Optimierung
    def risk_func(x):
        return np.var(x)

    def sustainability_func(x):
        return np.sum(x**2)

    result_mc = optimizer.multi_criteria_optimization(alpha=0.5, beta=0.3, gamma=0.2, risk_func=risk_func, sustainability_func=sustainability_func, method='SLSQP')
    if result_mc is not None and result_mc.success:
        print("\nMultikriterielle Optimierung:")
        print("Optimale Lösung:", result_mc.x)
    else:
        print("\nMultikriterielle Optimierung fehlgeschlagen oder kein Ergebnis verfügbar.")

    # Parametersensitivitätsanalysen
    optimizer.plot_parameter_sensitivity(np.linspace(1, 5, 10), 'a', parameter_index=0, method='SLSQP', interactive=args.interactive)
    optimizer.plot_parameter_sensitivity(np.linspace(0.05, 0.15, 10), 'b', parameter_index=(0,1), method='SLSQP', interactive=args.interactive)

if __name__ == "__main__":
    main()
