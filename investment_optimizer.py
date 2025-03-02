#!/usr/bin/env python
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

# Konstante, um Probleme (z.B. mit np.log) zu vermeiden:
MIN_INVESTMENT = 1e-8

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

def validate_inputs(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, c=None, epsilon_a=1e-6):
    n = len(roi_factors)
    synergy_matrix = np.array(synergy_matrix)
    if synergy_matrix.shape != (n, n):
        raise ValueError(f"Synergie-Matrix muss quadratisch sein und zur Länge der ROI-Faktoren ({n}) passen.")
    if not is_symmetric(synergy_matrix):
        raise ValueError("Synergie-Matrix muss symmetrisch sein.")
    if not (len(lower_bounds) == len(upper_bounds) == len(initial_allocation) == n):
        raise ValueError("lower_bounds, upper_bounds und initial_allocation müssen die gleiche Länge haben wie roi_factors.")
    if np.any(np.array(lower_bounds) > np.array(upper_bounds)):
        raise ValueError("Für jeden Bereich muss lower_bound <= upper_bound gelten.")
    if np.sum(lower_bounds) > total_budget:
        raise ValueError("Die Summe der Mindestinvestitionen überschreitet das Gesamtbudget.")
    if np.sum(upper_bounds) < total_budget:
        raise ValueError("Die Summe der Höchstinvestitionen ist kleiner als das Gesamtbudget.")
    if np.any(initial_allocation < lower_bounds):
        raise ValueError("Initial allocation unterschreitet mindestens eine Mindestinvestition.")
    if np.any(initial_allocation > upper_bounds):
        raise ValueError("Initial allocation überschreitet mindestens eine Höchstinvestition.")
    if np.any(np.array(lower_bounds) <= 0):
        raise ValueError("Alle Mindestinvestitionen müssen größer als 0 sein.")
    if np.any(np.array(roi_factors) <= epsilon_a):
        raise ValueError(f"Alle ROI-Faktoren müssen größer als {epsilon_a} sein.")
    if np.any(synergy_matrix < 0):
        raise ValueError("Alle Synergieeffekte müssen größer oder gleich Null sein.")
    if c is not None:
        c = np.array(c)
        if c.shape != (n,):
            raise ValueError("Optionaler Parameter 'c' muss die gleiche Form wie roi_factors haben.")
        if np.any(c < 0):
            raise ValueError("Alle Werte in 'c' müssen größer oder gleich Null sein.")

def get_bounds(lower_bounds, upper_bounds):
    return [(max(lower_bounds[i], MIN_INVESTMENT), upper_bounds[i]) for i in range(len(lower_bounds))]

def compute_synergy(x, synergy_matrix):
    return 0.5 * np.dot(x, np.dot(synergy_matrix, x))

def adjust_initial_guess(initial_allocation, lower_bounds, upper_bounds, total_budget, tol=1e-6):
    x0 = np.clip(initial_allocation, lower_bounds, upper_bounds)
    current_sum = np.sum(x0)
    max_iter = 100
    iter_count = 0
    while not np.isclose(current_sum, total_budget, atol=tol) and iter_count < max_iter:
        scale = total_budget / current_sum
        x0 = np.clip(x0 * scale, lower_bounds, upper_bounds)
        current_sum = np.sum(x0)
        iter_count += 1
    if not np.isclose(current_sum, total_budget, atol=tol):
        raise ValueError("Anpassung der Anfangsschätzung nicht möglich.")
    return x0

def validate_optional_param_c(c, reference_shape):
    c = np.array(c)
    if c.shape != reference_shape:
        raise ValueError("Parameter 'c' muss die gleiche Form wie die ROI-Faktoren haben.")
    if np.any(c < 0):
        raise ValueError("Alle Werte in 'c' müssen größer oder gleich Null sein.")
    return c

class OptimizationResult:
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)

def single_simulation(sim_index, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels, method, variation_percentage, optimizer_class):
    try:
        roi_sim = roi_factors * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=len(roi_factors))
        roi_sim = np.maximum(roi_sim, 1e-6)
        synergy_sim = synergy_matrix * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=synergy_matrix.shape)
        synergy_sim = np.triu(synergy_sim, 1) + np.triu(synergy_sim, 1).T
        optimizer_sim = optimizer_class(
            roi_sim, synergy_sim, total_budget,
            lower_bounds, upper_bounds, initial_allocation,
            investment_labels=investment_labels, log_level=logging.CRITICAL, c=None
        )
        result = optimizer_sim.optimize(method=method)
        if result and result.success:
            return result.x.tolist()
        else:
            return [np.nan] * len(roi_factors)
    except Exception as e:
        logging.error(f"Simulation {sim_index} fehlgeschlagen: {e}", exc_info=True)
        return [np.nan] * len(roi_factors)

class InvestmentOptimizer:
    """
    Optimiert die Investitionsallokation, wobei ROI-Faktoren und Synergieeffekte
    unter Berücksichtigung von Budget- und Investitionsgrenzen maximalen Nutzen erzielen sollen.
    """
    def __init__(self, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels=None, log_level=logging.INFO, c=None):
        configure_logging(log_level)
        try:
            self.roi_factors = np.array(roi_factors)
            self.synergy_matrix = np.array(synergy_matrix)
            self.total_budget = total_budget
            self.lower_bounds = np.array(lower_bounds)
            self.upper_bounds = np.array(upper_bounds)
            self.n = len(roi_factors)
            self.investment_labels = investment_labels if investment_labels else [f'Bereich_{i}' for i in range(self.n)]
            self.initial_allocation = adjust_initial_guess(initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
            self.c = np.array(c) if c is not None else None
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, self.c)
        except Exception as e:
            logging.error(f"Fehler bei der Initialisierung: {e}", exc_info=True)
            raise

    def update_parameters(self, roi_factors=None, synergy_matrix=None, total_budget=None, lower_bounds=None, upper_bounds=None, initial_allocation=None, c=None):
        try:
            parameters_updated = False
            if roi_factors is not None:
                self.roi_factors = np.array(roi_factors)
                parameters_updated = True
            if synergy_matrix is not None:
                synergy_matrix = np.array(synergy_matrix)
                if not is_symmetric(synergy_matrix):
                    raise ValueError("Synergie-Matrix muss symmetrisch sein.")
                if np.any(synergy_matrix < 0):
                    raise ValueError("Alle Synergieeffekte müssen größer oder gleich Null sein.")
                self.synergy_matrix = synergy_matrix
                parameters_updated = True
            if total_budget is not None:
                self.total_budget = total_budget
                parameters_updated = True
            if lower_bounds is not None:
                self.lower_bounds = np.array(lower_bounds)
                parameters_updated = True
            if upper_bounds is not None:
                self.upper_bounds = np.array(upper_bounds)
                parameters_updated = True
            if initial_allocation is not None:
                self.initial_allocation = adjust_initial_guess(initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
            elif parameters_updated:
                self.initial_allocation = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
            if c is not None:
                self.c = np.array(c)
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, self.c)
        except Exception as e:
            logging.error(f"Fehler beim Aktualisieren der Parameter: {e}", exc_info=True)
            raise

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        synergy = compute_synergy(x, self.synergy_matrix)
        if self.c is not None:
            utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
        else:
            utility = np.sum(self.roi_factors * np.log(x))
        total_utility = utility + synergy
        budget_diff = np.sum(x) - self.total_budget
        penalty_budget = penalty_coeff * (budget_diff ** 2)
        return -total_utility + penalty_budget

    def objective_without_penalty(self, x):
        synergy = compute_synergy(x, self.synergy_matrix)
        if self.c is not None:
            utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
        else:
            utility = np.sum(self.roi_factors * np.log(x))
        total_utility = utility + synergy
        return -total_utility

    def identify_top_synergies_correct(self, top_n=6):
        try:
            synergy_copy = self.synergy_matrix.copy()
            np.fill_diagonal(synergy_copy, 0)
            triu_indices = np.triu_indices(self.n, k=1)
            synergy_values = synergy_copy[triu_indices]
            top_indices = np.argpartition(synergy_values, -top_n)[-top_n:]
            top_synergies = []
            for idx in top_indices:
                i = triu_indices[0][idx]
                j = triu_indices[1][idx]
                top_synergies.append(((i, j), synergy_copy[i, j]))
            top_synergies.sort(key=lambda x: x[1], reverse=True)
            return top_synergies
        except Exception as e:
            logging.error(f"Fehler beim Identifizieren der Top-Synergien: {e}", exc_info=True)
            return []

    def optimize(self, method='SLSQP', max_retries=3, workers=None, **kwargs):
        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        scipy_version = version.parse(scipy.__version__)
        de_workers_supported = scipy_version >= version.parse("1.4.0")
        if method == 'DE':
            updating = 'deferred' if workers is not None and workers != 1 and de_workers_supported else 'immediate'
        use_penalty = method in ['DE', 'TNC']
        optimization_methods = {
            'SLSQP': lambda: minimize(
                self.objective_with_penalty if use_penalty else self.objective_without_penalty,
                self.initial_allocation,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}] if not use_penalty else [],
                options={'disp': False, 'maxiter': 1000},
                args=() if not use_penalty else (1e8,)
            ),
            'DE': lambda: differential_evolution(
                self.objective_with_penalty,
                bounds,
                strategy=kwargs.get('strategy', 'best1bin'),
                maxiter=kwargs.get('maxiter', 1000),
                tol=kwargs.get('tol', 1e-8),
                updating=updating,
                workers=workers if workers is not None and de_workers_supported else 1,
                polish=True,
                init='latinhypercube',
                args=(1e8,)
            ),
            'BasinHopping': lambda: basinhopping(
                self.objective_with_penalty,
                self.initial_allocation,
                niter=kwargs.get('niter', 100),
                stepsize=kwargs.get('stepsize', 0.5),
                minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds, 'constraints': [{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}],
                                  'options': {'maxiter': 1000, 'disp': False}},
                T=kwargs.get('T', 1.0),
                niter_success=kwargs.get('niter_success', 10)
            ),
            'TNC': lambda: minimize(
                self.objective_with_penalty,
                self.initial_allocation,
                method='TNC',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000},
                args=(1e8,)
            )
        }
        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods:
            logging.error(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden: {supported_methods}")
            raise ValueError(f"Nicht unterstützte Optimierungsmethode: {method}.")
        for attempt in range(1, max_retries + 1):
            try:
                opt_result = optimization_methods[method]()
                if method == 'BasinHopping':
                    x_opt = opt_result.x
                    fun_opt = opt_result.fun
                    success = opt_result.lowest_optimization_result.success
                    message = opt_result.lowest_optimization_result.message
                else:
                    x_opt = opt_result.x
                    fun_opt = opt_result.fun
                    success = opt_result.success
                    message = opt_result.message if hasattr(opt_result, 'message') else 'Optimierung abgeschlossen.'
                constraints_satisfied = (
                    np.isclose(np.sum(x_opt), self.total_budget, atol=1e-6) and
                    np.all(x_opt >= self.lower_bounds - 1e-8) and
                    np.all(x_opt <= self.upper_bounds + 1e-8)
                )
                if success and constraints_satisfied:
                    return OptimizationResult(x=x_opt, fun=fun_opt, success=success, message=message)
            except Exception as e:
                logging.error(f"Optimierungsversuch {attempt} mit Methode {method} fehlgeschlagen: {e}", exc_info=True)
        return None

    def sensitivity_analysis(self, B_values, method='SLSQP', tol=1e-6, **kwargs):
        allocations = []
        max_utilities = []
        for B_new in B_values:
            try:
                new_x0 = self.initial_allocation * (B_new / self.total_budget)
                new_x0 = adjust_initial_guess(new_x0, self.lower_bounds, self.upper_bounds, B_new)
                optimizer_copy = InvestmentOptimizer(
                    self.roi_factors,
                    self.synergy_matrix,
                    B_new,
                    self.lower_bounds,
                    self.upper_bounds,
                    new_x0,
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
                logging.error(f"Sensitivitätsanalyse für Budget {B_new} fehlgeschlagen: {e}", exc_info=True)
                allocations.append([np.nan] * self.n)
                max_utilities.append(np.nan)
        return B_values, allocations, max_utilities

    def robustness_analysis(self, num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=None):
        results = []
        if parallel:
            if num_workers is None:
                num_workers = min(4, os.cpu_count() or 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        single_simulation,
                        sim, self.roi_factors, self.synergy_matrix, self.total_budget,
                        self.lower_bounds, self.upper_bounds, self.initial_allocation,
                        self.investment_labels, method, variation_percentage, InvestmentOptimizer
                    ) for sim in range(num_simulations)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Robustheitsanalyse Simulation fehlgeschlagen: {e}", exc_info=True)
                        results.append([np.nan] * self.n)
        else:
            for sim in range(num_simulations):
                try:
                    result = single_simulation(sim, self.roi_factors, self.synergy_matrix, self.total_budget,
                                               self.lower_bounds, self.upper_bounds, self.initial_allocation,
                                               self.investment_labels, method, variation_percentage, InvestmentOptimizer)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Robustheitsanalyse Simulation fehlgeschlagen: {e}", exc_info=True)
                    results.append([np.nan] * self.n)
        df_results = pd.DataFrame(results, columns=self.investment_labels)
        additional_stats = df_results.describe(percentiles=[0.25, 0.75]).transpose()
        return additional_stats, df_results

    def multi_criteria_optimization(self, alpha, beta, gamma, risk_func, sustainability_func, method='SLSQP'):
        def objective(x):
            synergy = compute_synergy(x, self.synergy_matrix)
            if self.c is not None:
                utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
            else:
                utility = np.sum(self.roi_factors * np.log(x))
            total_utility = utility + synergy
            risk = beta * risk_func(x)
            sustainability = gamma * sustainability_func(x)
            return -(alpha * total_utility - risk - sustainability)
        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        try:
            result = minimize(
                objective,
                self.initial_allocation,
                method=method,
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}],
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
            if interactive and plotly_available:
                df_alloc = pd.DataFrame(allocations, columns=self.investment_labels)
                df_alloc['Budget'] = B_values
                df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})
                fig1 = px.line(df_alloc, x='Budget', y=self.investment_labels, title='Optimale Investitionsallokationen')
                fig1.show()
                fig2 = px.line(df_util, x='Budget', y='Maximaler Nutzen', title='Maximaler Nutzen bei verschiedenen Budgets')
                fig2.show()
            else:
                df_alloc = pd.DataFrame(allocations, columns=self.investment_labels)
                df_alloc.fillna(0, inplace=True)
                df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})
                fig, ax1 = plt.subplots(figsize=(12, 8))
                colors = sns.color_palette("tab10", n_colors=self.n)
                for i, label in enumerate(self.investment_labels):
                    ax1.plot(B_values, df_alloc[label], label=label, color=colors[i], alpha=0.7)
                ax1.set_xlabel('Budget')
                ax1.set_ylabel('Investitionsbetrag')
                ax1.legend(loc='upper left')
                ax1.grid(True)
                ax2 = ax1.twinx()
                ax2.plot(B_values, df_util['Maximaler Nutzen'], label='Maximaler Nutzen', color='tab:red', marker='o')
                ax2.set_ylabel('Maximaler Nutzen')
                ax2.legend(loc='upper right')
                plt.title(f'Optimale Investitionsallokation und Nutzen ({method})')
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
            sns.heatmap(self.synergy_matrix, annot=True, xticklabels=self.investment_labels, yticklabels=self.investment_labels, cmap='viridis')
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
                    self.roi_factors.copy(),
                    self.synergy_matrix.copy(),
                    self.total_budget,
                    self.lower_bounds.copy(),
                    self.upper_bounds.copy(),
                    self.initial_allocation.copy(),
                    investment_labels=self.investment_labels,
                    log_level=logging.CRITICAL,
                    c=self.c
                )
                if parameter_name == 'roi_factors':
                    if not isinstance(parameter_index, int):
                        raise ValueError("Für 'roi_factors' muss parameter_index ein int sein.")
                    if value <= 1e-6:
                        raise ValueError("ROI-Faktor muss größer als 1e-6 sein.")
                    new_roi = optimizer_copy.roi_factors.copy()
                    new_roi[parameter_index] = value
                    optimizer_copy.update_parameters(roi_factors=new_roi)
                elif parameter_name == 'synergy_matrix':
                    if not (isinstance(parameter_index, tuple) and len(parameter_index) == 2):
                        raise ValueError("Für 'synergy_matrix' muss parameter_index ein Tuple mit zwei Indizes sein.")
                    if value < 0:
                        raise ValueError("Synergieeffekte müssen größer oder gleich Null sein.")
                    i, j = parameter_index
                    new_synergy = optimizer_copy.synergy_matrix.copy()
                    new_synergy[i, j] = value
                    new_synergy[j, i] = value
                    optimizer_copy.update_parameters(synergy_matrix=new_synergy)
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
            xlabel = f'{parameter_name}[{parameter_index}]' if parameter_index is not None else parameter_name
            if interactive and plotly_available:
                df_util = pd.DataFrame({'Parameter Value': parameter_values, 'Maximaler Nutzen': utilities})
                fig = px.line(df_util, x='Parameter Value', y='Maximaler Nutzen', title=f'Sensitivität des Nutzens gegenüber {xlabel}')
                fig.show()
            else:
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
        return self.identify_top_synergies_correct(top_n=top_n)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Investment Optimizer für Startups')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Interaktive Plotly-Plots verwenden')
    return parser.parse_args()

def optimize_for_startup(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation=None):
    if initial_allocation is None:
        initial_allocation = np.full(len(roi_factors), total_budget/len(roi_factors))
    optimizer = InvestmentOptimizer(
        roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
        log_level=logging.WARNING
    )
    result = optimizer.optimize(method='SLSQP')
    if result is not None and result.success:
        allocation_dict = {f"Bereich_{i}": round(val, 2) for i, val in enumerate(result.x)}
        return {'allocation': allocation_dict, 'max_utility': -result.fun}
    else:
        return {'allocation': None, 'max_utility': None}

def main():
    args = parse_arguments()
    configure_logging(logging.DEBUG)
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            required_keys = ['roi_factors', 'synergy_matrix', 'total_budget', 'lower_bounds', 'upper_bounds', 'initial_allocation']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Schlüssel '{key}' fehlt in der Konfigurationsdatei.")
            investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(len(config['roi_factors']))])
            roi_factors = np.array(config['roi_factors'])
            synergy_matrix = np.array(config['synergy_matrix'])
            total_budget = config['total_budget']
            lower_bounds = np.array(config['lower_bounds'])
            upper_bounds = np.array(config['upper_bounds'])
            initial_allocation = np.array(config['initial_allocation'])
            c = np.array(config['c']) if 'c' in config else None
        except Exception as e:
            logging.error(f"Fehler beim Laden der Konfiguration: {e}", exc_info=True)
            sys.exit(1)
    else:
        investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        roi_factors = np.array([1, 2, 3, 4])
        synergy_matrix = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0]
        ])
        total_budget = 10.0
        lower_bounds = np.array([1, 1, 1, 1])
        upper_bounds = np.array([5, 5, 5, 5])
        initial_allocation = np.array([2, 2, 2, 4])
        c = np.array([0.1, 0.1, 0.1, 0.1])
    try:
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels=investment_labels, log_level=logging.DEBUG, c=c)
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
    B_sens, allocations_sens, utilities_sens = optimizer.sensitivity_analysis(B_values, method='SLSQP')
    optimizer.plot_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP', interactive=args.interactive)

    # Identifikation der Top-Synergien
    top_synergies = optimizer.identify_top_synergies(top_n=6)
    print("\nWichtigste Synergieeffekte (sortiert):")
    for pair, value in top_synergies:
        print(f"Bereiche {investment_labels[pair[0]]} & {investment_labels[pair[1]]}: Synergieeffekt = {value}")

    # Robustheitsanalyse
    df_robust_stats, df_robust = optimizer.robustness_analysis(num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=4)
    print("\nRobustheitsanalyse (Statistik der Investitionsallokationen):")
    print(df_robust_stats)
    optimizer.plot_robustness_analysis(df_robust)

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
    optimizer.plot_parameter_sensitivity(np.linspace(1, 5, 10), 'roi_factors', parameter_index=0, method='SLSQP', interactive=args.interactive)
    optimizer.plot_parameter_sensitivity(np.linspace(0.05, 0.15, 10), 'synergy_matrix', parameter_index=(0, 1), method='SLSQP', interactive=args.interactive)

if __name__ == "__main__":
    main()
