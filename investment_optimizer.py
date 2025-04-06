#!/usr/bin/env python
import sys
import logging
import concurrent.futures
import copy
import argparse
import json
import os
import itertools
import numpy as np
from scipy.optimize import (minimize, differential_evolution, basinhopping, Bounds,
                            NonlinearConstraint, LinearConstraint)
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

# --- Konstanten ---
# Numerische Präzision und Toleranzen
FLOAT_PRECISION = 1e-9 # Allgemeine kleine Zahl
BUDGET_PRECISION = 1e-6 # Toleranz für Budget-Constraint-Prüfung
BOUNDS_PRECISION = 1e-7 # Toleranz für Bounds-Prüfung
MIN_INVESTMENT = 1e-8  # Mindestwert, um log(0) etc. zu vermeiden

# Standardwerte für Nutzenfunktion und Analysen (können durch Config überschrieben werden)
DEFAULT_UTILITY_FUNCTION = 'log'
DEFAULT_SENSITIVITY_NUM_STEPS = 15
DEFAULT_SENSITIVITY_BUDGET_MIN_FACTOR = 0.7
DEFAULT_SENSITIVITY_BUDGET_MAX_FACTOR = 1.5
DEFAULT_ROBUSTNESS_NUM_SIMULATIONS = 200
DEFAULT_ROBUSTNESS_VARIATION_PERCENTAGE = 0.15
DEFAULT_ROBUSTNESS_NUM_WORKERS_FACTOR = 4 # Wird mit min(os.cpu_count(), factor) verwendet
DEFAULT_PARAM_SENSITIVITY_NUM_STEPS = 11
DEFAULT_PARAM_SENSITIVITY_MIN_FACTOR = 0.5
DEFAULT_PARAM_SENSITIVITY_MAX_FACTOR = 1.5
DEFAULT_TOP_N_SYNERGIES = 6
DEFAULT_PARETO_NUM_SAMPLES = 50

# Überprüfen der erforderlichen Pakete bei Skriptstart
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
    print(f"Fehlende Pakete: {', '.join(missing_packages)}. Bitte installieren Sie sie mit pip install -r requirements.txt.")
    # Exit nur, wenn Kernpakete fehlen
    if any(p in missing_packages for p in ['numpy', 'scipy', 'pandas', 'matplotlib', 'packaging']):
        sys.exit(1)
    elif 'plotly' in missing_packages:
        plotly_available = False # Deaktiviere nur Plotly

# --- Logging Konfiguration ---
def configure_logging(level=logging.INFO):
    """Konfiguriert das Logging-System."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Hilfsfunktionen ---
def is_symmetric(matrix, tol=FLOAT_PRECISION):
    """Überprüft, ob eine Matrix symmetrisch ist."""
    return np.allclose(matrix, matrix.T, atol=tol)

def validate_inputs(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                    utility_function_type=DEFAULT_UTILITY_FUNCTION, c=None,
                    s_curve_L=None, s_curve_k=None, s_curve_x0=None,
                    epsilon_a=MIN_INVESTMENT):
    """Validiert die Eingabeparameter für den Optimizer, berücksichtigt den Typ der Nutzenfunktion."""
    n = len(roi_factors) # Basisdimension bleibt roi_factors
    if len(lower_bounds) != n or len(upper_bounds) != n or len(initial_allocation) != n:
         raise ValueError("roi_factors, lower_bounds, upper_bounds und initial_allocation müssen die gleiche Länge haben.")

    synergy_matrix = np.array(synergy_matrix)
    if synergy_matrix.shape != (n, n):
        raise ValueError(f"Synergie-Matrix muss quadratisch ({n}x{n}) sein.")
    if not is_symmetric(synergy_matrix):
        raise ValueError("Synergie-Matrix muss symmetrisch sein.")
    if np.any(synergy_matrix < 0):
        raise ValueError("Alle Synergieeffekte müssen größer oder gleich Null sein.")

    if np.any(np.array(lower_bounds) > np.array(upper_bounds) + FLOAT_PRECISION):
        raise ValueError("Für jeden Bereich muss lower_bound <= upper_bound gelten.")
    if np.sum(lower_bounds) > total_budget + BUDGET_PRECISION:
        raise ValueError(f"Die Summe der Mindestinvestitionen ({np.sum(lower_bounds)}) überschreitet das Gesamtbudget ({total_budget}).")
    if np.sum(upper_bounds) < total_budget - BUDGET_PRECISION:
        raise ValueError(f"Die Summe der Höchstinvestitionen ({np.sum(upper_bounds)}) ist kleiner als das Gesamtbudget ({total_budget}).")

    if np.any(initial_allocation < np.array(lower_bounds) - BOUNDS_PRECISION):
        raise ValueError("Initial allocation unterschreitet mindestens eine Mindestinvestition.")
    if np.any(initial_allocation > np.array(upper_bounds) + BOUNDS_PRECISION):
        raise ValueError("Initial allocation überschreitet mindestens eine Höchstinvestition.")
    if np.any(np.array(lower_bounds) < 0):
        logging.warning(f"Einige Mindestinvestitionen sind < 0. Sie werden auf 0 oder MIN_INVESTMENT ({MIN_INVESTMENT}) angehoben, wo nötig.")

    # --- Validierung basierend auf Nutzenfunktion ---
    valid_utility_types = ['log', 'quadratic', 's_curve']
    if utility_function_type not in valid_utility_types:
        raise ValueError(f"Unbekannter utility_function_type: '{utility_function_type}'. Muss einer von {valid_utility_types} sein.")

    if utility_function_type == 'log':
        if np.any(np.array(roi_factors) <= epsilon_a):
            raise ValueError(f"Alle ROI-Faktoren müssen größer als {epsilon_a} sein, wenn utility_function_type='log' ist.")
        if c is not None: logging.warning("Parameter 'c' wird ignoriert (utility_function_type='log').")
        if s_curve_L is not None or s_curve_k is not None or s_curve_x0 is not None: logging.warning("S-Kurven-Parameter ignoriert (utility_function_type='log').")

    elif utility_function_type == 'quadratic':
        if c is None: raise ValueError("Parameter 'c' muss angegeben werden (utility_function_type='quadratic').")
        c = np.array(c)
        if c.shape != (n,): raise ValueError("Parameter 'c' muss die gleiche Form wie roi_factors haben.")
        if np.any(c < 0): raise ValueError("Alle Werte in 'c' müssen >= 0 sein.")
        if s_curve_L is not None or s_curve_k is not None or s_curve_x0 is not None: logging.warning("S-Kurven-Parameter ignoriert (utility_function_type='quadratic').")

    elif utility_function_type == 's_curve':
        logging.info("utility_function_type='s_curve': 'roi_factors' wird ignoriert, S-Kurven-Parameter verwendet.")
        if s_curve_L is None or s_curve_k is None or s_curve_x0 is None:
            raise ValueError("s_curve_L, s_curve_k und s_curve_x0 müssen angegeben werden (utility_function_type='s_curve').")
        s_curve_L, s_curve_k, s_curve_x0 = np.array(s_curve_L), np.array(s_curve_k), np.array(s_curve_x0)
        if not (s_curve_L.shape == (n,) and s_curve_k.shape == (n,) and s_curve_x0.shape == (n,)):
            raise ValueError("s_curve_L/k/x0 müssen die gleiche Form wie roi_factors haben.")
        if np.any(s_curve_L <= 0): raise ValueError("Alle Werte in 's_curve_L' müssen positiv sein.")
        if np.any(s_curve_k <= 0): raise ValueError("Alle Werte in 's_curve_k' müssen positiv sein.")
        if c is not None: logging.warning("Parameter 'c' wird ignoriert (utility_function_type='s_curve').")

def get_bounds(lower_bounds, upper_bounds):
    """Erstellt die Bounds-Liste (Tupel) für ältere scipy.optimize-Methoden."""
    safe_lower_bounds = np.maximum(lower_bounds, 0)
    # Stelle sicher, dass lower <= upper gilt, auch nach Anwendung von MIN_INVESTMENT
    return [(max(safe_lower_bounds[i], MIN_INVESTMENT), max(upper_bounds[i], max(safe_lower_bounds[i], MIN_INVESTMENT))) for i in range(len(lower_bounds))]


def compute_synergy(x, synergy_matrix):
    """Berechnet den Synergie-Term."""
    return 0.5 * np.dot(x, np.dot(synergy_matrix, x))

def adjust_initial_guess(initial_allocation, lower_bounds, upper_bounds, total_budget, tol=BUDGET_PRECISION):
    """Passt die initiale Schätzung an, um Bounds und Budget-Summe (approximativ) zu erfüllen."""
    x0 = np.array(initial_allocation, dtype=float)
    n = len(x0)
    lb = np.maximum(np.array(lower_bounds, dtype=float), 0) # >= 0 für Logik hier
    ub = np.array(upper_bounds, dtype=float)

    # Schritt 1: Clipping
    x0 = np.clip(x0, lb, ub)

    # Schritt 2: Iterative Anpassung an Budget
    current_sum = np.sum(x0)
    max_iter = 100
    iter_count = 0
    if np.isclose(current_sum, total_budget, atol=tol): return x0

    while not np.isclose(current_sum, total_budget, atol=tol) and iter_count < max_iter:
        diff = total_budget - current_sum
        can_increase = (x0 < ub - BOUNDS_PRECISION)
        can_decrease = (x0 > lb + BOUNDS_PRECISION)

        if diff > 0: # Erhöhen
            active_indices = np.where(can_increase)[0]
            if len(active_indices) == 0: break
            weights = x0[active_indices] + FLOAT_PRECISION
            sum_weights = np.sum(weights)
            adjustment = diff / len(active_indices) if sum_weights < FLOAT_PRECISION else diff * weights / sum_weights
            x0[active_indices] += adjustment
        elif diff < 0: # Verringern
            active_indices = np.where(can_decrease)[0]
            if len(active_indices) == 0: break
            weights = x0[active_indices] + FLOAT_PRECISION
            sum_weights = np.sum(weights)
            adjustment = diff / len(active_indices) if sum_weights < FLOAT_PRECISION else diff * weights / sum_weights
            x0[active_indices] += adjustment
        else: break

        x0 = np.clip(x0, lb, ub) # Erneut clippen nach Anpassung
        current_sum = np.sum(x0)
        iter_count += 1

    # Schritt 3: Fallback, wenn Budget nicht erreicht wurde
    if not np.isclose(current_sum, total_budget, atol=tol * 10):
        logging.warning(f"Anpassung der Startallokation erreichte nicht exakt das Budget ({current_sum:.4f} vs {total_budget:.4f}). Verwende Fallback.")
        x0 = lb.copy()
        remaining_budget = total_budget - np.sum(x0)
        if remaining_budget < -BUDGET_PRECISION:
             logging.error("Summe Lower Bounds > Budget in adjust_initial_guess Fallback.")
             x0 = np.clip(initial_allocation, lb, ub); x0 *= (total_budget / np.sum(x0)) if np.sum(x0)>FLOAT_PRECISION else 1; return np.clip(x0, lb, ub)
        elif remaining_budget > FLOAT_PRECISION:
            allocatable_range = ub - lb
            positive_range_indices = np.where(allocatable_range > FLOAT_PRECISION)[0]
            if len(positive_range_indices) > 0:
                 allocatable_sum = np.sum(allocatable_range[positive_range_indices])
                 if allocatable_sum > FLOAT_PRECISION:
                     proportions = allocatable_range[positive_range_indices] / allocatable_sum
                     x0[positive_range_indices] += remaining_budget * proportions
                 else: # Falls Range 0, aber Indices existieren (lb=ub), verteile gleichmäßig
                     x0[positive_range_indices] += remaining_budget / len(positive_range_indices)

        x0 = np.clip(x0, lb, ub)
        current_sum = np.sum(x0)
        if not np.isclose(current_sum, total_budget, atol=tol * 100):
             logging.error(f"Anpassung der Startallokation fehlgeschlagen. Budgetsumme {current_sum} weicht stark von {total_budget} ab.")
             return np.clip(initial_allocation, lb, ub) # Letzter Versuch

    return x0

class OptimizationResult:
    """Container für Optimierungsergebnisse."""
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x; self.fun = fun; self.success = success; self.message = message
        for key, value in kwargs.items(): setattr(self, key, value)

def single_simulation(sim_index, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels, method, variation_percentage, optimizer_class,
                      utility_function_type, c=None, s_curve_L=None, s_curve_k=None, s_curve_x0=None):
    """Führt eine einzelne Simulation für die Robustheitsanalyse durch."""
    try:
        # --- Parameter Variation ---
        roi_sim, c_sim = None, None
        s_curve_L_sim, s_curve_k_sim, s_curve_x0_sim = None, None, None
        n_local = len(roi_factors) # Dimension

        if utility_function_type == 'log' or utility_function_type == 'quadratic':
             roi_sim = roi_factors * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=n_local)
             if utility_function_type == 'log': roi_sim = np.maximum(roi_sim, MIN_INVESTMENT)
             if utility_function_type == 'quadratic' and c is not None:
                 c_sim = c * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=c.shape); c_sim = np.maximum(c_sim, 0)
        elif utility_function_type == 's_curve':
             if s_curve_L is not None: s_curve_L_sim = s_curve_L * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=s_curve_L.shape); s_curve_L_sim = np.maximum(s_curve_L_sim, FLOAT_PRECISION)
             if s_curve_k is not None: s_curve_k_sim = s_curve_k * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=s_curve_k.shape); s_curve_k_sim = np.maximum(s_curve_k_sim, FLOAT_PRECISION)
             if s_curve_x0 is not None:
                 x0_var = np.abs(s_curve_x0 * variation_percentage) + FLOAT_PRECISION; s_curve_x0_sim = s_curve_x0 + np.random.uniform(-x0_var, x0_var, size=s_curve_x0.shape)
        else: # Fallback oder Fehler
             roi_sim = roi_factors # Keine Variation, wenn Typ unbekannt

        synergy_sim = synergy_matrix * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=synergy_matrix.shape)
        synergy_sim = np.maximum(synergy_sim, 0); synergy_sim = (synergy_sim + synergy_sim.T) / 2; np.fill_diagonal(synergy_sim, 0)

        # --- Optimizer erstellen ---
        optimizer_sim = optimizer_class(
            roi_factors=roi_sim if roi_sim is not None else roi_factors, # Übergib variierte oder Original-ROI
            synergy_matrix=synergy_sim, total_budget=total_budget,
            lower_bounds=lower_bounds, upper_bounds=upper_bounds, initial_allocation=initial_allocation,
            investment_labels=investment_labels, log_level=logging.CRITICAL, utility_function_type=utility_function_type,
            c=c_sim, s_curve_L=s_curve_L_sim, s_curve_k=s_curve_k_sim, s_curve_x0=s_curve_x0_sim
        )
        result = optimizer_sim.optimize(method=method)
        return result.x.tolist() if result and result.success else [np.nan] * n_local
    except Exception as e:
        logging.error(f"Simulation {sim_index} fehlgeschlagen: {e}", exc_info=False)
        return [np.nan] * len(roi_factors) # Verwende ursprüngliche Länge für Konsistenz

class InvestmentOptimizer:
    """
    Optimiert die Investitionsallokation.
    Unterstützt Nutzenfunktionen: 'log', 'quadratic', 's_curve'.
    """
    def __init__(self, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                 investment_labels=None, log_level=logging.INFO,
                 utility_function_type=DEFAULT_UTILITY_FUNCTION, c=None,
                 s_curve_L=None, s_curve_k=None, s_curve_x0=None):
        """Initialisiert den InvestmentOptimizer."""
        configure_logging(log_level)
        try:
            self.roi_factors = np.array(roi_factors)
            self.synergy_matrix = np.array(synergy_matrix)
            self.total_budget = total_budget
            self.lower_bounds = np.array(lower_bounds)
            self.upper_bounds = np.array(upper_bounds)
            self.n = len(roi_factors)
            self.investment_labels = investment_labels if investment_labels else [f'Bereich_{i}' for i in range(self.n)]
            if len(self.investment_labels) != self.n: raise ValueError("Label/ROI Längen mismatch.")

            self.utility_function_type = utility_function_type
            self.c = np.array(c) if c is not None else None
            self.s_curve_L = np.array(s_curve_L) if s_curve_L is not None else None
            self.s_curve_k = np.array(s_curve_k) if s_curve_k is not None else None
            self.s_curve_x0 = np.array(s_curve_x0) if s_curve_x0 is not None else None

            # Validierung (Basis + Utility-spezifisch)
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds,
                            np.clip(initial_allocation, self.lower_bounds, self.upper_bounds), # Use clipped for initial check
                            self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0)

            self.initial_allocation = adjust_initial_guess(initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)

            # Finale Validierung mit angepasster Allokation
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation,
                            self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0)

            logging.info(f"InvestmentOptimizer initialisiert (Nutzenfunktion: {self.utility_function_type}).")
        except Exception as e:
            logging.error(f"Fehler bei der Initialisierung: {e}", exc_info=True); raise

    def update_parameters(self, roi_factors=None, synergy_matrix=None, total_budget=None,
                          lower_bounds=None, upper_bounds=None, initial_allocation=None,
                          utility_function_type=None, c=None,
                          s_curve_L=None, s_curve_k=None, s_curve_x0=None):
        """Aktualisiert die Parameter des Optimizers und validiert sie erneut."""
        try:
            parameters_updated = False; utility_params_changed = False
            # --- Update Basisparameter ---
            if roi_factors is not None:
                 new_roi = np.array(roi_factors)
                 if len(new_roi) != self.n: raise ValueError("Dimension von roi_factors kann nicht geändert werden.")
                 self.roi_factors = new_roi; parameters_updated = True; logging.debug("roi_factors aktualisiert.")
            if synergy_matrix is not None:
                synergy_matrix=np.array(synergy_matrix)
                if not is_symmetric(synergy_matrix) or np.any(synergy_matrix < 0) or synergy_matrix.shape!=(self.n, self.n): raise ValueError("Ungültige synergy_matrix.")
                self.synergy_matrix = synergy_matrix; parameters_updated = True; logging.debug("synergy_matrix aktualisiert.")
            if total_budget is not None: self.total_budget = total_budget; parameters_updated = True; logging.debug("total_budget aktualisiert.")
            if lower_bounds is not None:
                new_lb = np.array(lower_bounds)
                if len(new_lb) != self.n: raise ValueError("Dimension von lower_bounds kann nicht geändert werden.")
                self.lower_bounds = new_lb; parameters_updated = True; logging.debug("lower_bounds aktualisiert.")
            if upper_bounds is not None:
                new_ub = np.array(upper_bounds)
                if len(new_ub) != self.n: raise ValueError("Dimension von upper_bounds kann nicht geändert werden.")
                self.upper_bounds = new_ub; parameters_updated = True; logging.debug("upper_bounds aktualisiert.")

            # --- Update Nutzenfunktion und Parameter ---
            if utility_function_type is not None:
                if utility_function_type not in ['log', 'quadratic', 's_curve']: raise ValueError(f"Unbekannter utility_function_type: {utility_function_type}")
                if self.utility_function_type != utility_function_type: self.utility_function_type = utility_function_type; parameters_updated = True; utility_params_changed = True; logging.debug(f"utility_function_type -> {self.utility_function_type}")
            # Handling von 'c' - Prüfung ob Argument explizit übergeben wurde
            if 'c' in locals():
                 if c is not None: # Wenn c nicht None ist (neuer Wert oder Array)
                      new_c = np.array(c)
                      if len(new_c) != self.n: raise ValueError("Dimension von c passt nicht.")
                      self.c = new_c; parameters_updated = True; utility_params_changed = True; logging.debug("Parameter c aktualisiert/gesetzt.")
                 elif self.c is not None: # Wenn c explizit None ist und vorher nicht None war
                      self.c = None; parameters_updated = True; utility_params_changed = True; logging.debug("Parameter c entfernt.")
            # Handling von S-Kurven Parametern
            if 's_curve_L' in locals():
                 new_L = np.array(s_curve_L) if s_curve_L is not None else None
                 if new_L is not None and len(new_L) != self.n: raise ValueError("Dimension von s_curve_L passt nicht.")
                 self.s_curve_L = new_L; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_L aktualisiert.")
            if 's_curve_k' in locals():
                 new_k = np.array(s_curve_k) if s_curve_k is not None else None
                 if new_k is not None and len(new_k) != self.n: raise ValueError("Dimension von s_curve_k passt nicht.")
                 self.s_curve_k = new_k; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_k aktualisiert.")
            if 's_curve_x0' in locals():
                 new_x0 = np.array(s_curve_x0) if s_curve_x0 is not None else None
                 if new_x0 is not None and len(new_x0) != self.n: raise ValueError("Dimension von s_curve_x0 passt nicht.")
                 self.s_curve_x0 = new_x0; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_x0 aktualisiert.")

            # --- Initial Allocation anpassen ---
            current_initial_allocation = self.initial_allocation if initial_allocation is None else np.array(initial_allocation)
            if initial_allocation is not None:
                if len(current_initial_allocation) != self.n: raise ValueError("Länge der neuen initial_allocation passt nicht.")
                self.initial_allocation = adjust_initial_guess(current_initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget); parameters_updated = True; logging.debug("initial_allocation aktualisiert/angepasst.")
            elif parameters_updated: # Wenn andere Parameter geändert wurden, passe alte Allokation an
                self.initial_allocation = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget); logging.debug("Bestehende initial_allocation angepasst.")

            # --- Finale Validierung ---
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation,
                            self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0)
            logging.info(f"Parameter aktualisiert (Nutzenfunktion: {self.utility_function_type}).")
        except Exception as e:
            logging.error(f"Fehler beim Aktualisieren der Parameter: {e}", exc_info=True); raise

    def _calculate_utility(self, x):
        """Berechnet den Nutzen-Term basierend auf self.utility_function_type."""
        try:
            if self.utility_function_type == 'log':
                return np.sum(self.roi_factors * np.log(np.maximum(x, MIN_INVESTMENT)))
            elif self.utility_function_type == 'quadratic':
                if self.c is None: raise ValueError("'c' fehlt für quadratic.")
                return np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
            elif self.utility_function_type == 's_curve':
                if self.s_curve_L is None or self.s_curve_k is None or self.s_curve_x0 is None: raise ValueError("S-Kurven Parameter fehlen.")
                # Clip exponent to avoid overflow/underflow
                exponent_term = np.clip(-self.s_curve_k * (np.maximum(x, 0) - self.s_curve_x0), -700, 700) # Ensure x>=0 for stability
                return np.sum(self.s_curve_L / (1 + np.exp(exponent_term)))
            else: raise ValueError(f"Unbekannter Typ '{self.utility_function_type}'.")
        except Exception as e:
             logging.error(f"Fehler in _calculate_utility (Typ: {self.utility_function_type}) für x={x}: {e}", exc_info=False)
             return -np.inf # Return negative infinity to signal error to minimizer

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        """Zielfunktion (zu minimieren) mit Budget-Penalty."""
        utility = self._calculate_utility(x)
        if np.isinf(utility): return np.inf # Error in utility -> return infinity
        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy
        budget_diff = np.sum(x) - self.total_budget
        penalty_budget = penalty_coeff * (budget_diff ** 2)
        result = -total_utility + penalty_budget
        # Return large finite number if calculation resulted in inf/nan
        return result if np.isfinite(result) else np.finfo(float).max / 10

    def objective_without_penalty(self, x):
        """Zielfunktion (zu minimieren) ohne explizite Penalties."""
        utility = self._calculate_utility(x)
        if np.isinf(utility): return np.inf # Error in utility -> return infinity
        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy
        result = -total_utility
        # Return large finite number if calculation resulted in inf/nan
        return result if np.isfinite(result) else np.finfo(float).max / 10

    def identify_top_synergies_correct(self, top_n=DEFAULT_TOP_N_SYNERGIES):
        """Identifiziert die Top N Synergieeffekte."""
        if top_n <= 0 or self.n < 2: return []
        try:
            synergy_copy = self.synergy_matrix.copy(); np.fill_diagonal(synergy_copy, -np.inf)
            triu_indices = np.triu_indices(self.n, k=1); synergy_values = synergy_copy[triu_indices]
            num_synergies = len(synergy_values); actual_top_n = min(top_n, num_synergies)
            if actual_top_n == 0: return []
            sorted_indices = np.argsort(synergy_values); top_indices = sorted_indices[-actual_top_n:]
            top_synergies = []
            for flat_idx in reversed(top_indices):
                row, col = triu_indices[0][flat_idx], triu_indices[1][flat_idx]
                value = synergy_copy[row, col]
                if np.isfinite(value): top_synergies.append(((row, col), value))
            return top_synergies
        except Exception as e:
            logging.error(f"Fehler bei Top-Synergien: {e}", exc_info=True); return []

    def optimize(self, method='SLSQP', max_retries=3, workers=None, penalty_coeff=1e8, **kwargs):
        """Führt die Optimierung mit der gewählten Methode durch."""
        scipy_bounds = Bounds(lb=np.maximum(self.lower_bounds, 0), ub=self.upper_bounds)
        tuple_bounds = get_bounds(self.lower_bounds, self.upper_bounds)

        # Define constraints (use NonLinear for broader compatibility with minimize)
        constraints_for_minimize = NonlinearConstraint(fun=lambda x: np.sum(x), lb=self.total_budget, ub=self.total_budget)
        # DE prefers LinearConstraint if available
        constraints_for_de = LinearConstraint(np.ones((1, self.n)), lb=self.total_budget, ub=self.total_budget)

        scipy_version = version.parse(scipy.__version__)
        de_constraints_supported = scipy_version >= version.parse("1.7.0")
        de_workers_supported = scipy_version >= version.parse("1.4.0")
        actual_workers = workers if workers is not None and workers > 0 and de_workers_supported else 1
        de_updating = 'deferred' if actual_workers > 1 and de_workers_supported else 'immediate'

        use_penalty_objective = method in ['TNC', 'L-BFGS-B', 'Nelder-Mead']
        constraints_arg = None # Constraints object to pass to the optimizer
        if method == 'DE':
            if de_constraints_supported: use_penalty_objective = False; constraints_arg = constraints_for_de; logging.debug("DE: Native Constraints.")
            else: use_penalty_objective = True; logging.warning("DE: Fallback auf Penalty (alte SciPy?).")
        elif method in ['SLSQP', 'trust-constr']: use_penalty_objective = False; constraints_arg = constraints_for_minimize
        elif method == 'BasinHopping': use_penalty_objective = False # Handled by local minimizer

        objective_func = self.objective_with_penalty if use_penalty_objective else self.objective_without_penalty
        obj_args = (penalty_coeff,) if use_penalty_objective else ()

        # Define options, merge with kwargs
        def get_options(method_key, base_options, kwargs):
            opts = base_options.copy()
            opts.update(kwargs.get('options', {}))
            return opts

        slsqp_options = get_options('slsqp', {'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION}, kwargs)
        de_opts_base = {'strategy': 'best1bin', 'maxiter': 1000, 'popsize': 15, 'tol': 0.01,'mutation': (0.5, 1), 'recombination': 0.7,
                      'updating': de_updating, 'workers': actual_workers, 'polish': not de_constraints_supported and de_constraints_supported is not None , 'init': 'latinhypercube'}
        de_opts_base.update({k:v for k,v in kwargs.items() if k not in ['options', 'constraints']}) # DE takes kwargs directly
        basinhopping_opts_base = {'niter': 100, 'stepsize': 0.5, 'T': 1.0, 'disp': False, 'niter_success': 10}
        basinhopping_opts_base['minimizer_kwargs'] = { 'method': 'SLSQP', 'bounds': tuple_bounds, 'constraints': constraints_for_minimize,
                                                 'options': {'maxiter': 200, 'ftol': FLOAT_PRECISION, 'disp': False}, 'args': () }
        basinhopping_opts_base.update({k:v for k,v in kwargs.items() if k not in ['options', 'minimizer_kwargs']})
        tnc_options = get_options('tnc', {'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION, 'gtol': FLOAT_PRECISION*10}, kwargs)
        lbfgsb_options = get_options('lbfgsb', {'disp': False, 'maxiter': 15000, 'ftol': 2.22e-09, 'gtol': 1e-5}, kwargs)
        trustconstr_options = get_options('trustconstr', {'disp': False, 'maxiter': 1000, 'gtol': FLOAT_PRECISION, 'xtol': FLOAT_PRECISION}, kwargs)
        neldermead_options = get_options('neldermead', {'disp': False, 'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4}, kwargs)


        # Lambda definitions using current state
        optimization_methods = {
            'SLSQP': lambda x0: minimize(objective_func, x0, method='SLSQP', bounds=tuple_bounds, constraints=constraints_arg, options=slsqp_options, args=obj_args),
            'DE': lambda x0: differential_evolution(objective_func, tuple_bounds, constraints=constraints_arg if de_constraints_supported else (), args=obj_args if use_penalty_objective else (), **de_opts_base),
            'BasinHopping': lambda x0: basinhopping(objective_func, x0, args=(), **basinhopping_opts_base), # uses args from minimizer_kwargs
            'TNC': lambda x0: minimize(objective_func, x0, method='TNC', bounds=tuple_bounds, options=tnc_options, args=obj_args),
            'L-BFGS-B': lambda x0: minimize(objective_func, x0, method='L-BFGS-B', bounds=tuple_bounds, options=lbfgsb_options, args=obj_args),
            'trust-constr': lambda x0: minimize(objective_func, x0, method='trust-constr', bounds=scipy_bounds, constraints=constraints_arg, options=trustconstr_options, args=obj_args),
            'Nelder-Mead': lambda x0: minimize(objective_func, x0, method='Nelder-Mead', bounds=tuple_bounds, options=neldermead_options, args=obj_args),
        }

        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods: raise ValueError(f"Methode {method} nicht unterstützt. Optionen: {supported_methods}")

        logging.info(f"Starte Optimierung mit Methode: {method}")
        if method == 'DE': logging.info(f"DE Details: Native Constraints = {de_constraints_supported}, Workers = {actual_workers}, Polish = {de_opts_base['polish']}")

        current_initial_allocation = self.initial_allocation.copy()
        for attempt in range(1, max_retries + 1):
            try:
                logging.debug(f"Optimierungsversuch {attempt}/{max_retries} mit Methode {method}...")
                current_method_func = optimization_methods[method]
                opt_result = current_method_func(current_initial_allocation) # Pass current start point

                # --- Ergebnisextraktion ---
                if method == 'BasinHopping':
                    x_opt = opt_result.x; fun_opt_val = opt_result.fun # BH fun is from the local minimizer (should be without penalty)
                    success = opt_result.lowest_optimization_result.success if hasattr(opt_result.lowest_optimization_result, 'success') else False
                    message = opt_result.message;
                    if isinstance(message, (list, tuple)): message = " ".join(map(str, message))
                else:
                    x_opt = opt_result.x
                    fun_opt_val = self.objective_without_penalty(x_opt) if use_penalty_objective else opt_result.fun # Recalculate if penalty was used
                    success = opt_result.success
                    message = opt_result.message if hasattr(opt_result, 'message') else 'Optimierung abgeschlossen.'

                # --- Constraint-Prüfung ---
                budget_ok = np.isclose(np.sum(x_opt), self.total_budget, atol=BUDGET_PRECISION)
                lower_bounds_ok = np.all(x_opt >= self.lower_bounds - BOUNDS_PRECISION)
                upper_bounds_ok = np.all(x_opt <= self.upper_bounds + BOUNDS_PRECISION)
                constraints_satisfied = budget_ok and lower_bounds_ok and upper_bounds_ok

                logging.debug(f"Versuch {attempt}: Erfolg={success}, Budget OK={budget_ok} (Sum={np.sum(x_opt):.6f}), Bounds OK={lower_bounds_ok and upper_bounds_ok}, Nachricht='{message}'")

                if success and constraints_satisfied:
                    logging.info(f"Optimierung mit {method} erfolgreich in Versuch {attempt}.")
                    return OptimizationResult(x=x_opt, fun=fun_opt_val, success=success, message=message) # fun is always without penalty here
                elif success and not constraints_satisfied: logging.warning(f"Optimierung {method} nominell erfolgreich, aber Constraints verletzt. Retry...")
                elif not success: logging.warning(f"Optimierung {method} nicht erfolgreich (Nachricht: {message}). Retry...")

            except Exception as e: logging.error(f"Optimierungsversuch {attempt} ({method}) fehlgeschlagen: {e}", exc_info=False)

            if attempt < max_retries: # Prepare for next attempt
                 jitter = np.random.normal(0, self.total_budget * 0.01, self.n)
                 current_initial_allocation = adjust_initial_guess(self.initial_allocation + jitter, self.lower_bounds, self.upper_bounds, self.total_budget)
                 logging.debug("Startpunkt leicht verändert für nächsten Versuch.")

        logging.error(f"Optimierung mit Methode {method} nach {max_retries} Versuchen fehlgeschlagen.")
        return None

    def sensitivity_analysis(self, B_values, method='SLSQP', tol=BUDGET_PRECISION, **kwargs):
        """Führt eine Sensitivitätsanalyse für verschiedene Budgets durch."""
        allocations, max_utilities = [], []
        # Store original state completely
        original_state = {
            'budget': self.total_budget, 'initial': self.initial_allocation.copy(),
            'utility_type': self.utility_function_type, 'c': self.c.copy() if self.c is not None else None,
            'L': self.s_curve_L.copy() if self.s_curve_L is not None else None,
            'k': self.s_curve_k.copy() if self.s_curve_k is not None else None,
            'x0': self.s_curve_x0.copy() if self.s_curve_x0 is not None else None
        }
        logging.info(f"Starte Sensitivitätsanalyse für Budgets [{min(B_values):.2f}-{max(B_values):.2f}] ({method}).")
        last_successful_alloc = original_state['initial']
        for B_new in B_values:
            current_sum = np.sum(last_successful_alloc); new_x0_guess = last_successful_alloc * (B_new / current_sum) if current_sum > FLOAT_PRECISION else np.full(self.n, B_new / self.n)
            adjusted_x0 = adjust_initial_guess(new_x0_guess, self.lower_bounds, self.upper_bounds, B_new)
            try:
                 # Update only budget and initial allocation for this run
                 self.update_parameters(total_budget=B_new, initial_allocation=adjusted_x0)
                 result = self.optimize(method=method, **kwargs)
                 if result and result.success:
                     last_successful_alloc = result.x; allocations.append(result.x); max_utilities.append(-result.fun)
                 else:
                     logging.warning(f"Sensitivitätsanalyse für Budget {B_new:.2f} fehlgeschlagen."); allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
            except Exception as e:
                 logging.error(f"Fehler Sensitivitätsanalyse B={B_new:.2f}: {e}", exc_info=True); allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
            finally:
                 # Reset parameters for the next loop iteration or end (safer)
                 self.update_parameters(total_budget=original_state['budget'], initial_allocation=original_state['initial'], utility_function_type=original_state['utility_type'],
                                        c=original_state['c'], s_curve_L=original_state['L'], s_curve_k=original_state['k'], s_curve_x0=original_state['x0'])
        logging.info("Sensitivitätsanalyse abgeschlossen.")
        return B_values, allocations, max_utilities

    def robustness_analysis(self, num_simulations=DEFAULT_ROBUSTNESS_NUM_SIMULATIONS, method='DE', variation_percentage=DEFAULT_ROBUSTNESS_VARIATION_PERCENTAGE, parallel=True, num_workers=None):
        """ Führt eine Robustheitsanalyse durch. """
        if num_workers is None: cpu_count = os.cpu_count() or 1; num_workers = min(cpu_count, DEFAULT_ROBUSTNESS_NUM_WORKERS_FACTOR)
        logging.info(f"Starte Robustheitsanalyse ({num_simulations} Sims, +/-{variation_percentage*100:.1f}%, Utility: {self.utility_function_type}, Method: {method}, Parallel: {parallel}, Workers: {num_workers if parallel else 'N/A'}).")
        # Pass all necessary parameters for the utility function calculation
        args_template = (self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, self.investment_labels, method, variation_percentage, InvestmentOptimizer,
                         self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0)
        results = []
        log_interval = max(1, num_simulations // 10); completed_count = 0
        executor_class = concurrent.futures.ThreadPoolExecutor if parallel and num_simulations > 1 else None

        if executor_class:
            with executor_class(max_workers=num_workers) as executor:
                futures = [executor.submit(single_simulation, i, *args_template) for i in range(num_simulations)]
                for future in concurrent.futures.as_completed(futures):
                    try: results.append(future.result());
                    except Exception as e: logging.error(f"Fehler bei Robustheitssimulation Ergebnis: {e}", exc_info=False); results.append([np.nan] * self.n)
                    completed_count += 1
                    if completed_count % log_interval == 0: logging.debug(f"Robustheitsanalyse: {completed_count}/{num_simulations} abgeschlossen.")
        else: # Sequentiell
            for sim in range(num_simulations):
                 results.append(single_simulation(sim, *args_template)); completed_count += 1
                 if completed_count % log_interval == 0: logging.debug(f"Robustheitsanalyse: {completed_count}/{num_simulations} abgeschlossen.")

        df_results = pd.DataFrame(results, columns=self.investment_labels); num_failed = df_results.isna().any(axis=1).sum(); num_successful = len(df_results) - num_failed
        logging.info(f"Robustheitsanalyse abgeschlossen. {num_successful}/{num_simulations} erfolgreich.")
        if num_failed > 0: logging.warning(f"{num_failed} Simulationen lieferten kein gültiges Ergebnis (NaN).")
        df_clean = df_results.dropna()
        if not df_clean.empty:
             percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]; stats = df_clean.describe(percentiles=percentiles).transpose()
             mean_val = df_clean.mean()
             stats['cv'] = (df_clean.std() / mean_val).replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero
             stats['range'] = df_clean.max() - df_clean.min(); stats['iqr'] = stats['75%'] - stats['25%']
        else: logging.warning("Keine erfolgreichen Simulationen für Statistik."); stats = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'cv', 'range', 'iqr'], index=self.investment_labels)
        return stats, df_results

    def multi_criteria_optimization(self, alpha, beta, gamma, risk_func, sustainability_func, method='SLSQP'):
        """Führt eine multikriterielle Optimierung für EINE gegebene Gewichtung durch."""
        if not (callable(risk_func) and callable(sustainability_func)): raise TypeError("risk_func/sustainability_func müssen callable sein.")
        def multi_objective(x):
            neg_utility = self.objective_without_penalty(x)
            if np.isinf(neg_utility) or np.isnan(neg_utility): return np.finfo(float).max / 10 # Hoher Wert bei Fehler
            try: risk = risk_func(x); risk = risk if np.isfinite(risk) else np.inf
            except Exception as e: logging.warning(f"Fehler risk_func: {e}"); risk = np.inf
            try: sustainability = sustainability_func(x); sustainability = sustainability if np.isfinite(sustainability) else -np.inf
            except Exception as e: logging.warning(f"Fehler sust_func: {e}"); sustainability = -np.inf
            combined_value = alpha * neg_utility + beta * risk - gamma * sustainability
            return combined_value if np.isfinite(combined_value) else np.finfo(float).max / 10
        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        constraints = NonlinearConstraint(fun=lambda x: np.sum(x), lb=self.total_budget, ub=self.total_budget)
        logging.debug(f"Starte multikrit. Optimierung für Gewichte: U={alpha}, R={beta}, S={gamma}")
        try:
            result = minimize(multi_objective, self.initial_allocation, method=method, bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION})
            if result.success:
                final_x = result.x; final_utility = -self.objective_without_penalty(final_x)
                final_risk = risk_func(final_x) if callable(risk_func) else np.nan; final_sustainability = sustainability_func(final_x) if callable(sustainability_func) else np.nan
                return OptimizationResult(x=final_x, fun=result.fun, success=result.success, message=result.message, utility=final_utility, risk=final_risk, sustainability=final_sustainability)
            else: logging.debug(f"Multikrit. Optimierung fehlgeschlagen (U={alpha},R={beta},S={gamma}): {result.message}"); return None
        except Exception as e: logging.error(f"Fehler multikrit. Optimierung (U={alpha},R={beta},S={gamma}): {e}", exc_info=False); return None

    def generate_weights(self, num_objectives, num_samples):
        """Generiert Gewichtungsvektoren, die zu 1 summieren."""
        # ... (Code bleibt gleich) ...
        weights = []
        if num_objectives == 2:
            for i in range(num_samples + 1): w1 = i / num_samples; weights.append(np.array([w1, 1.0 - w1]))
        elif num_objectives == 3:
            steps = int(round(num_samples**(1/2))); steps = max(1, steps)
            for i in range(steps + 1):
                for j in range(steps - i + 1): weights.append(np.array([i, j, steps - i - j]) / steps)
            for w_bound in [[1,0,0], [0,1,0], [0,0,1]]: # Ensure boundary weights
                 if not any(np.allclose(w, w_bound) for w in weights): weights.append(np.array(w_bound))
            weights = np.unique(np.round(weights, decimals=5), axis=0)
            logging.info(f"Generiere {len(weights)} eindeutige Gewichtungsvektoren für 3 Ziele.")
        else: weights = [np.full(num_objectives, 1.0 / num_objectives)]
        return [w for w in weights]

    def is_dominated(self, solution_objectives, other_objectives_list):
        """Prüft, ob 'solution_objectives' von irgendeiner Lösung in 'other_objectives_list' dominiert wird."""
        # ... (Code bleibt gleich) ...
        sol = np.array(solution_objectives)
        for other in other_objectives_list:
            oth = np.array(other)
            if np.all(oth >= sol) and np.any(oth > sol): return True
        return False

    def filter_non_dominated(self, solutions_objectives):
        """Filtert eine Liste von Zielwert-Vektoren und gibt nur die nicht-dominierten zurück."""
        # ... (Code bleibt gleich, mit Korrektur für Vergleich) ...
        if not solutions_objectives: return []
        non_dominated_indices = []
        num_solutions = len(solutions_objectives)
        for i in range(num_solutions):
             is_dominated_by_any = False
             for j in range(num_solutions):
                 if i == j: continue
                 # Check if sol_j dominates sol_i
                 if np.all(np.array(solutions_objectives[j]) >= np.array(solutions_objectives[i])) and \
                    np.any(np.array(solutions_objectives[j]) > np.array(solutions_objectives[i])):
                     is_dominated_by_any = True
                     break
             if not is_dominated_by_any:
                 non_dominated_indices.append(i)
        non_dominated_set = [solutions_objectives[i] for i in non_dominated_indices]
        unique_solutions, seen = [], set()
        for sol in non_dominated_set:
             obj_tuple = tuple(np.round(sol, decimals=5))
             if obj_tuple not in seen: unique_solutions.append(sol); seen.add(obj_tuple)
        logging.info(f"Filterung: {len(solutions_objectives)} -> {len(non_dominated_set)} nicht dominiert -> {len(unique_solutions)} eindeutig nicht dominiert.")
        return unique_solutions

    def find_pareto_approximation(self, risk_func, sustainability_func, num_weight_samples=DEFAULT_PARETO_NUM_SAMPLES, method='SLSQP'):
        """Approximiert die Pareto-Front durch Iteration über verschiedene Gewichtungen."""
        # ... (Code bleibt gleich, mit Korrektur bei Ergebnis-Matching) ...
        logging.info(f"Starte Pareto-Front-Approximation ({num_weight_samples} Samples, Methode: {method}).")
        weights_list = self.generate_weights(3, num_weight_samples)
        results, successful_runs = [], 0
        for weights in weights_list:
            alpha, beta, gamma = weights
            result = self.multi_criteria_optimization(alpha, beta, gamma, risk_func, sustainability_func, method)
            if result and result.success:
                results.append({'allocation': result.x, 'utility': result.utility, 'risk': result.risk, 'sustainability': result.sustainability, 'alpha': alpha, 'beta': beta, 'gamma': gamma}); successful_runs += 1
        logging.info(f"Pareto-Approximation: {successful_runs}/{len(weights_list)} Läufe erfolgreich.")
        if not results: logging.warning("Keine erfolgreichen Pareto-Läufe."); return pd.DataFrame()
        objective_values = [(r['utility'], -r['risk'], r['sustainability']) for r in results if all(np.isfinite(v) for v in [r['utility'], r['risk'], r['sustainability']))]
        valid_indices = [i for i, r in enumerate(results) if all(np.isfinite(v) for v in [r['utility'], r['risk'], r['sustainability']))]
        if not objective_values: logging.warning("Keine gültigen Zielwerte für Pareto-Filterung."); return pd.DataFrame()
        non_dominated_objectives = self.filter_non_dominated(objective_values)
        final_results = []
        # Finde die Original-Ergebnisse für die nicht-dominierten Ziele (robuster Vergleich)
        used_original_indices = set()
        tol = 1e-5
        for nd_obj in non_dominated_objectives:
            best_match_original_idx = -1
            min_dist = np.inf
            current_match_objective_idx = -1

            for objective_idx, obj_val in enumerate(objective_values):
                 original_idx = valid_indices[objective_idx]
                 if original_idx in used_original_indices: continue

                 dist = np.linalg.norm(np.array(nd_obj) - np.array(obj_val))
                 if dist < min_dist:
                     # Prüfen ob dist "nah genug" ist und besser als bisher
                     if dist < tol:
                          min_dist = dist
                          best_match_original_idx = original_idx
                          current_match_objective_idx = objective_idx # Merke Index in objective_values/valid_indices

            if best_match_original_idx != -1:
                 final_results.append(results[best_match_original_idx])
                 used_original_indices.add(best_match_original_idx)
                 # Markiere auch den Index in objective_values als verwendet, falls nötig
                 # (wird implizit durch used_original_indices abgedeckt)


        if not final_results: logging.warning("Keine nicht-dominierten Lösungen nach Filterung."); return pd.DataFrame()
        logging.info(f"Pareto-Approximation: {len(final_results)} nicht-dominierte Lösungen gefunden.")
        df_results = pd.DataFrame(final_results).sort_values(by='utility', ascending=False).reset_index(drop=True)
        return df_results


    # --- Plotting Methoden ---
    # plot_sensitivity, plot_robustness_analysis, plot_synergy_heatmap, plot_parameter_sensitivity, plot_pareto_approximation
    # bleiben wie in der letzten Version (mit plt.close() etc.)
    def plot_sensitivity(self, B_values, allocations, utilities, method='SLSQP', interactive=False):
        """Plottet die Ergebnisse der Sensitivitätsanalyse."""
        if not isinstance(B_values, (list, np.ndarray)) or len(B_values) == 0 or \
           not isinstance(allocations, (list, np.ndarray)) or len(allocations) == 0 or \
           not isinstance(utilities, (list, np.ndarray)) or len(utilities) == 0:
             logging.warning("Keine Daten für Sensitivitätsplot."); return
        df_alloc = pd.DataFrame(allocations, columns=self.investment_labels); df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})
        valid_indices = df_util['Maximaler Nutzen'].notna() & (~df_alloc.isna().any(axis=1))
        if not valid_indices.any(): logging.warning("Keine gültigen Datenpunkte für Sensitivitätsplot."); return
        B_values_clean=df_util['Budget'][valid_indices].values; allocations_clean=df_alloc[valid_indices].values; utilities_clean=df_util['Maximaler Nutzen'][valid_indices].values
        if allocations_clean.ndim == 1: allocations_clean = allocations_clean.reshape(1, -1)
        if len(B_values_clean) < 2: logging.warning("Weniger als 2 Punkte für Sensitivitätsplot."); return
        logging.info("Erstelle Plot für Sensitivitätsanalyse...")
        fig=None
        try:
            if interactive and plotly_available:
                 df_alloc_plt = pd.DataFrame(allocations_clean, columns=self.investment_labels); df_alloc_plt['Budget'] = B_values_clean
                 df_util_plt = pd.DataFrame({'Budget': B_values_clean, 'Maximaler Nutzen': utilities_clean})
                 fig1 = px.line(df_alloc_plt, x='Budget', y=self.investment_labels, title=f'Allokationen vs. Budget ({method})', labels={'value': 'Betrag', 'variable': 'Bereich'}); fig1.show()
                 fig2 = px.line(df_util_plt, x='Budget', y='Maximaler Nutzen', markers=True, title=f'Nutzen vs. Budget ({method})'); fig2.show()
            else:
                 fig, ax1 = plt.subplots(figsize=(12, 7)); colors = plt.cm.viridis(np.linspace(0, 1, self.n))
                 for i, label in enumerate(self.investment_labels): ax1.plot(B_values_clean, allocations_clean[:, i], label=label, color=colors[i], marker='o', linestyle='-', markersize=4, alpha=0.8)
                 ax1.set_xlabel('Gesamtbudget'); ax1.set_ylabel('Investitionsbetrag', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.legend(loc='upper left', title='Bereiche'); ax1.grid(True, linestyle='--', alpha=0.6)
                 ax2 = ax1.twinx(); ax2.plot(B_values_clean, utilities_clean, label='Maximaler Nutzen', color='tab:red', marker='x', linestyle='--', markersize=6); ax2.set_ylabel('Maximaler Nutzen', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red'); ax2.legend(loc='upper right')
                 plt.title(f'Sensitivitätsanalyse: Allokation & Nutzen vs. Budget ({method})', pad=20); fig.tight_layout(); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Sensitivität: {e}", exc_info=True)
        finally:
             if fig and not (interactive and plotly_available): plt.close(fig)

    def plot_robustness_analysis(self, df_results):
        """Plottet die Ergebnisse der Robustheitsanalyse."""
        if df_results is None or df_results.empty: logging.warning("Keine Daten für Robustheitsplot."); return
        df_clean = df_results.dropna(); num_successful_sims = len(df_clean)
        if num_successful_sims == 0: logging.warning("Keine gültigen Daten für Robustheitsplot."); return
        logging.info(f"Erstelle Plots für Robustheitsanalyse ({num_successful_sims} gültige Sims)...")
        try:
            df_melted = df_clean.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
            fig1, ax1 = plt.subplots(figsize=(max(8, self.n * 1.2), 6))
            sns.boxplot(x='Bereich', y='Investition', data=df_melted, palette='viridis', ax=ax1)
            ax1.set_xlabel('Investitionsbereich'); ax1.set_ylabel('Simulierter Investitionsbetrag'); ax1.set_title(f'Verteilung der Allokationen (Robustheit, n={num_successful_sims})')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right'); fig1.tight_layout(); plt.show(); plt.close(fig1)

            num_cols = min(4, self.n)
            try:
                 g = sns.FacetGrid(df_melted, col="Bereich", col_wrap=num_cols, sharex=False, sharey=False, height=3, aspect=1.2)
                 g.map(sns.histplot, "Investition", kde=True, bins=15, color='skyblue', stat='density'); g.fig.suptitle(f'Histogramme der Allokationen (Robustheit, n={num_successful_sims})', y=1.03)
                 g.set_titles("{col_name}"); plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show(); plt.close(g.fig)
            except Exception as e_facet: logging.warning(f"Fehler FacetGrid: {e_facet}.")

            if self.n <= 5:
                 logging.debug("Erstelle Pairplot für Robustheit.")
                 try:
                     g_pair = sns.pairplot(df_clean, kind='scatter', diag_kind='kde', plot_kws={'alpha':0.5, 's':20}, palette='viridis'); g_pair.fig.suptitle(f'Paarweise Beziehungen (Robustheit, n={num_successful_sims})', y=1.02)
                     plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show(); plt.close(g_pair.fig)
                 except Exception as e_pair: logging.warning(f"Fehler Pairplot: {e_pair}.")
            elif self.n > 5: logging.info("Pairplot übersprungen (>5 Bereiche).")
        except Exception as e: logging.error(f"Fehler beim Plotten der Robustheit: {e}", exc_info=True)
        finally: plt.close('all')

    def plot_synergy_heatmap(self):
        """Plottet eine Heatmap der Synergie-Matrix."""
        if self.synergy_matrix is None or self.n < 2: logging.warning("Keine Synergie-Matrix zum Plotten."); return
        logging.info("Erstelle Heatmap der Synergieeffekte...")
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, self.n * 0.8), max(5, self.n * 0.7)))
            sns.heatmap(self.synergy_matrix, annot=True, fmt=".2f", xticklabels=self.investment_labels, yticklabels=self.investment_labels, cmap='viridis', linewidths=.5, linecolor='black', cbar_kws={'label': 'Synergiestärke'}, ax=ax)
            ax.set_title('Heatmap der Synergieeffekte'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor'); plt.setp(ax.get_yticklabels(), rotation=0); fig.tight_layout(); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Synergie-Heatmap: {e}", exc_info=True)
        finally:
             if fig: plt.close(fig)

    def plot_parameter_sensitivity(self, parameter_values, parameter_name, parameter_index=None, method='SLSQP', interactive=False):
        """ Plottet die Sensitivität des Nutzens gegenüber Parameteränderungen. """
        if parameter_values is None or len(parameter_values) == 0: logging.warning(f"Keine Werte für '{parameter_name}'."); return
        original_state = {'roi': self.roi_factors.copy(), 'syn': self.synergy_matrix.copy(), 'initial': self.initial_allocation.copy(),
                          'utility_type': self.utility_function_type, 'c': self.c.copy() if self.c is not None else None,
                          'L': self.s_curve_L.copy() if self.s_curve_L is not None else None, 'k': self.s_curve_k.copy() if self.s_curve_k is not None else None,
                          'x0': self.s_curve_x0.copy() if self.s_curve_x0 is not None else None }
        utilities, allocations = [], []; parameter_relevant = True; xlabel = f"{parameter_name}[{parameter_index}]"
        # --- Bestimme xlabel und Relevanz ---
        if parameter_name == 'roi_factors':
             if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 'roi_factors'."); return
             xlabel = f'ROI Faktor für "{self.investment_labels[parameter_index]}"'
             if self.utility_function_type == 's_curve': logging.warning("ROI nicht relevant für S-Kurve."); parameter_relevant = False
        elif parameter_name == 'synergy_matrix':
             if not (isinstance(parameter_index, tuple) and len(parameter_index)==2 and 0<=parameter_index[0]<self.n and 0<=parameter_index[1]<self.n and parameter_index[0]!=parameter_index[1]): logging.error("Ungültiger Index für 'synergy_matrix'."); return
             i, j = parameter_index; i, j = min(i,j), max(i,j); parameter_index = (i,j); xlabel = f'Synergie "{self.investment_labels[i]}" & "{self.investment_labels[j]}"'
        elif parameter_name == 'c':
             if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 'c'."); return
             xlabel = f'Parameter c für "{self.investment_labels[parameter_index]}"'
             if self.utility_function_type != 'quadratic': logging.warning("'c' nur relevant für 'quadratic'."); parameter_relevant = False
        # TODO: Add sensitivity for s_curve_L, s_curve_k, s_curve_x0 if needed
        else: logging.error(f"Unbekannter Parametername: {parameter_name}"); return
        if not parameter_relevant: return
        logging.info(f"Starte Parametersensitivität für {xlabel}...")
        # --- Iteriere über Werte ---
        for value in parameter_values:
            logging.debug(f"Teste {xlabel} = {value:.4f}")
            temp_state = {'roi': original_state['roi'].copy(), 'syn': original_state['syn'].copy(), 'c': original_state['c'].copy() if original_state['c'] is not None else None,
                          'L': original_state['L'].copy() if original_state['L'] is not None else None, 'k': original_state['k'].copy() if original_state['k'] is not None else None,
                          'x0': original_state['x0'].copy() if original_state['x0'] is not None else None}
            try:
                 if parameter_name == 'roi_factors':
                     if self.utility_function_type == 'log' and value <= MIN_INVESTMENT: logging.warning(f"Wert {value:.4f} zu klein für log."); utilities.append(np.nan); allocations.append([np.nan]*self.n); continue
                     temp_state['roi'][parameter_index] = value
                 elif parameter_name == 'synergy_matrix':
                      if value < 0: logging.warning(f"Negativer Wert {value:.4f} für Synergie."); utilities.append(np.nan); allocations.append([np.nan]*self.n); continue
                      i, j = parameter_index; temp_state['syn'][i, j] = temp_state['syn'][j, i] = value
                 elif parameter_name == 'c':
                      if value < 0: logging.warning(f"Negativer Wert {value:.4f} für 'c'."); utilities.append(np.nan); allocations.append([np.nan]*self.n); continue
                      if temp_state['c'] is not None: temp_state['c'][parameter_index] = value
                 # Update optimizer temporarily
                 self.update_parameters(roi_factors=temp_state['roi'], synergy_matrix=temp_state['syn'], initial_allocation=original_state['initial'], utility_function_type=original_state['utility_type'],
                                        c=temp_state['c'], s_curve_L=temp_state['L'], s_curve_k=temp_state['k'], s_curve_x0=temp_state['x0'])
                 result = self.optimize(method=method)
                 if result and result.success: utilities.append(-result.fun); allocations.append(result.x)
                 else: logging.warning(f"Optimierung fehlgeschlagen für {xlabel} = {value:.4f}"); utilities.append(np.nan); allocations.append([np.nan]*self.n)
            except Exception as e: logging.error(f"Fehler bei Parametersensitivität {xlabel}={value:.4f}: {e}", exc_info=True); utilities.append(np.nan); allocations.append([np.nan]*self.n)
        # --- Wichtig: Parameter auf Originalwerte zurücksetzen ---
        self.update_parameters(roi_factors=original_state['roi'], synergy_matrix=original_state['syn'], initial_allocation=original_state['initial'], utility_function_type=original_state['utility_type'],
                               c=original_state['c'], s_curve_L=original_state['L'], s_curve_k=original_state['k'], s_curve_x0=original_state['x0'])
        # --- Ergebnisse plotten ---
        df_param_sens = pd.DataFrame({'Parameterwert': parameter_values, 'Maximaler Nutzen': utilities}); valid_indices = df_param_sens['Maximaler Nutzen'].notna()
        if not valid_indices.any(): logging.warning(f"Keine gültigen Ergebnisse für Parametersensitivität {xlabel}."); return
        if valid_indices.sum() < 2: logging.warning(f"Weniger als 2 Punkte für Parametersensitivität {xlabel}."); return
        param_values_clean = df_param_sens['Parameterwert'][valid_indices].values; utilities_clean = df_param_sens['Maximaler Nutzen'][valid_indices].values
        logging.info(f"Erstelle Plot für Parametersensitivität von {xlabel}...")
        fig = None
        try:
            if interactive and plotly_available:
                df_plot = pd.DataFrame({'Parameterwert': param_values_clean, 'Maximaler Nutzen': utilities_clean})
                fig_plotly = px.line(df_plot, x='Parameterwert', y='Maximaler Nutzen', markers=True, title=f'Sensitivität des Nutzens vs <br>{xlabel}'); fig_plotly.update_layout(xaxis_title=xlabel); fig_plotly.show()
            else:
                fig, ax = plt.subplots(figsize=(10, 6)); ax.plot(param_values_clean, utilities_clean, marker='o', linestyle='-'); ax.set_xlabel(xlabel); ax.set_ylabel('Maximaler Nutzen'); ax.set_title(f'Sensitivität des Nutzens vs\n{xlabel}'); ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Parametersensitivität für {xlabel}: {e}", exc_info=True)
        finally:
             if fig and not (interactive and plotly_available): plt.close(fig)

    def plot_pareto_approximation(self, df_pareto, interactive=False):
        """ Plottet die approximierte Pareto-Front. """
        if df_pareto is None or df_pareto.empty: logging.warning("Keine Daten für Pareto-Plot."); return
        required_cols = ['utility', 'risk', 'sustainability']
        if not all(col in df_pareto.columns for col in required_cols): logging.error(f"Pareto DataFrame fehlen Spalten: {required_cols}."); return
        logging.info(f"Erstelle Plot für approximierte Pareto-Front ({len(df_pareto)} Punkte)...")
        utility, risk, sustainability = df_pareto['utility'].values, df_pareto['risk'].values, df_pareto['sustainability'].values
        fig = None
        try:
            if interactive and plotly_available:
                 df_plot = df_pareto.copy()
                 color_col = 'sustainability'; hover_data = ['allocation', 'utility', 'risk', 'sustainability']
                 if all(col in df_plot for col in ['alpha', 'beta', 'gamma']):
                      df_plot['weights'] = df_plot.apply(lambda r: f"U:{r.alpha:.2f}|R:{r.beta:.2f}|S:{r.gamma:.2f}", axis=1); color_col = 'weights'; hover_data.append('weights')
                 fig_plotly = px.scatter_3d(df_plot, x='utility', y='risk', z='sustainability', color=color_col, hover_data=hover_data, title='Approximierte Pareto-Front')
                 fig_plotly.update_layout(scene = dict(xaxis_title='Utility (Max)', yaxis_title='Risk (Min)', zaxis_title='Sustainability (Max)')); fig_plotly.show()
            else:
                 fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False); cmap = plt.cm.viridis
                 try: # Normalize Sustainability for color, handle single value case
                      sus_min, sus_max = sustainability.min(), sustainability.max()
                      norm = plt.Normalize(sus_min, sus_max) if sus_min != sus_max else None
                 except: norm = None
                 c_values = sustainability if norm else 'blue' # Use blue if normalization fails
                 scatter1 = axs[0].scatter(utility, risk, c=c_values, cmap=cmap, norm=norm, alpha=0.7)
                 axs[0].set_xlabel('Utility (Max)'); axs[0].set_ylabel('Risk (Min)'); axs[0].set_title('Utility vs. Risk'); axs[0].grid(True, linestyle='--', alpha=0.6)
                 if norm: fig.colorbar(scatter1, ax=axs[0], label='Sustainability')

                 try: risk_min, risk_max = risk.min(), risk.max(); norm_r = plt.Normalize(risk_min, risk_max) if risk_min != risk_max else None
                 except: norm_r = None
                 c_values_r = risk if norm_r else 'blue'
                 scatter2 = axs[1].scatter(utility, sustainability, c=c_values_r, cmap=cmap, norm=norm_r, alpha=0.7)
                 axs[1].set_xlabel('Utility (Max)'); axs[1].set_ylabel('Sustainability (Max)'); axs[1].set_title('Utility vs. Sustainability'); axs[1].grid(True, linestyle='--', alpha=0.6)
                 if norm_r: fig.colorbar(scatter2, ax=axs[1], label='Risk')

                 try: util_min, util_max = utility.min(), utility.max(); norm_u = plt.Normalize(util_min, util_max) if util_min != util_max else None
                 except: norm_u = None
                 c_values_u = utility if norm_u else 'blue'
                 scatter3 = axs[2].scatter(risk, sustainability, c=c_values_u, cmap=cmap, norm=norm_u, alpha=0.7)
                 axs[2].set_xlabel('Risk (Min)'); axs[2].set_ylabel('Sustainability (Max)'); axs[2].set_title('Risk vs. Sustainability'); axs[2].grid(True, linestyle='--', alpha=0.6)
                 if norm_u: fig.colorbar(scatter3, ax=axs[2], label='Utility')

                 plt.suptitle('Approximierte Pareto-Front (2D Projektionen)', fontsize=16); fig.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Pareto-Approximation: {e}", exc_info=True)
        finally:
             if fig and not (interactive and plotly_available): plt.close(fig)
             elif not fig and not (interactive and plotly_available): plt.close('all')

# --- Hilfsfunktionen und Main ---
def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description='Investment Optimizer')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Interaktive Plotly-Plots verwenden')
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging-Level')
    return parser.parse_args()

def optimize_for_startup(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation=None,
                         utility_function_type='log', c=None, s_curve_L=None, s_curve_k=None, s_curve_x0=None):
    """Vereinfachte Funktion zur schnellen Optimierung (z.B. für API)."""
    n = len(roi_factors);
    if initial_allocation is None: initial_allocation = np.full(n, total_budget / n) if n > 0 else np.array([])
    try:
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                                        log_level=logging.WARNING, utility_function_type=utility_function_type, c=c,
                                        s_curve_L=s_curve_L, s_curve_k=s_curve_k, s_curve_x0=s_curve_x0)
        result = optimizer.optimize(method='SLSQP')
        if result and result.success:
            alloc_dict = {f"{optimizer.investment_labels[i]}": round(val, 4) for i, val in enumerate(result.x)}
            return {'allocation': alloc_dict, 'max_utility': -result.fun, 'success': True}
        else: message = result.message if result else "Optimierung fehlgeschlagen"; return {'allocation': None, 'max_utility': None, 'success': False, 'message': message}
    except Exception as e: logging.error(f"Fehler in optimize_for_startup: {e}", exc_info=False); return {'allocation': None, 'max_utility': None, 'success': False, 'message': str(e)}

def main():
    """Hauptfunktion zur Ausführung des Optimierers."""
    args = parse_arguments()
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    configure_logging(log_level)
    config = {}
    try: # Lade Config oder verwende Defaults
        if args.config:
            logging.info(f"Lade Konfiguration aus: {args.config}")
            with open(args.config, 'r') as f: config = json.load(f)
            required_keys = ['roi_factors', 'synergy_matrix', 'total_budget', 'lower_bounds', 'upper_bounds', 'initial_allocation']
            if any(key not in config for key in required_keys): raise KeyError(f"Schlüssel {required_keys} fehlen.")
            # Extract parameters safely using .get()
            roi_factors_cfg = config.get('roi_factors')
            n_cfg = len(roi_factors_cfg) if roi_factors_cfg else 0
            investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(n_cfg)])
            roi_factors = np.array(roi_factors_cfg); synergy_matrix = np.array(config.get('synergy_matrix'))
            total_budget = config.get('total_budget'); lower_bounds = np.array(config.get('lower_bounds')); upper_bounds = np.array(config.get('upper_bounds'))
            initial_allocation = np.array(config.get('initial_allocation'))
            utility_function_type = config.get('utility_function_type', DEFAULT_UTILITY_FUNCTION)
            c = np.array(config['c']) if config.get('c') is not None else None # Use get for safe access
            s_curve_L = np.array(config.get('s_curve_L')) if config.get('s_curve_L') is not None else None
            s_curve_k = np.array(config.get('s_curve_k')) if config.get('s_curve_k') is not None else None
            s_curve_x0 = np.array(config.get('s_curve_x0')) if config.get('s_curve_x0') is not None else None
        else:
            logging.info("Keine Konfigurationsdatei, verwende Standardwerte."); n_areas = 4
            investment_labels=['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']; roi_factors=np.array([1.5, 2.0, 2.5, 1.8])
            synergy_matrix=np.array([[0.,.1,.15,.05],[.1,0.,.2,.1],[.15,.2,0.,.25],[.05,.1,.25,0.]])
            total_budget=100.; lower_bounds=np.array([10.]*n_areas); upper_bounds=np.array([50.]*n_areas)
            initial_allocation=np.array([total_budget/n_areas]*n_areas); utility_function_type=DEFAULT_UTILITY_FUNCTION
            c=None; s_curve_L, s_curve_k, s_curve_x0 = None, None, None
            config = {} # Empty config dict for analysis defaults
    except Exception as e: logging.error(f"Fehler beim Laden/Verarbeiten der Konfiguration: {e}", exc_info=True); sys.exit(1)

    try: # Initialisiere Optimizer
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                                        investment_labels=investment_labels, log_level=log_level, utility_function_type=utility_function_type,
                                        c=c, s_curve_L=s_curve_L, s_curve_k=s_curve_k, s_curve_x0=s_curve_x0)
        optimizer.plot_synergy_heatmap()
    except Exception as e: logging.critical(f"Initialisierung fehlgeschlagen: {e}", exc_info=True); sys.exit(1)

    # --- Lade Analyseparameter ---
    analysis_params = config.get('analysis_parameters', {})
    sens_num_steps = analysis_params.get('sensitivity_num_steps', DEFAULT_SENSITIVITY_NUM_STEPS)
    sens_min_factor = analysis_params.get('sensitivity_budget_min_factor', DEFAULT_SENSITIVITY_BUDGET_MIN_FACTOR)
    sens_max_factor = analysis_params.get('sensitivity_budget_max_factor', DEFAULT_SENSITIVITY_BUDGET_MAX_FACTOR)
    robust_sims = analysis_params.get('robustness_num_simulations', DEFAULT_ROBUSTNESS_NUM_SIMULATIONS)
    robust_var_perc = analysis_params.get('robustness_variation_percentage', DEFAULT_ROBUSTNESS_VARIATION_PERCENTAGE)
    robust_workers_factor = analysis_params.get('robustness_num_workers_factor', DEFAULT_ROBUSTNESS_NUM_WORKERS_FACTOR)
    robust_num_workers = min(os.cpu_count() or 1, robust_workers_factor)
    param_sens_steps = analysis_params.get('parameter_sensitivity_num_steps', DEFAULT_PARAM_SENSITIVITY_NUM_STEPS)
    param_sens_min_factor = analysis_params.get('parameter_sensitivity_min_factor', DEFAULT_PARAM_SENSITIVITY_MIN_FACTOR)
    param_sens_max_factor = analysis_params.get('parameter_sensitivity_max_factor', DEFAULT_PARAM_SENSITIVITY_MAX_FACTOR)
    pareto_samples = analysis_params.get('pareto_num_samples', DEFAULT_PARETO_NUM_SAMPLES)
    de_maxiter = analysis_params.get('de_maxiter', 1500); de_popsize = analysis_params.get('de_popsize', 20)

    # --- Führe Optimierungen durch ---
    results_dict = {}
    optimizer_methods_to_run = ['SLSQP', 'trust-constr', 'DE', 'L-BFGS-B', 'Nelder-Mead']
    for method in optimizer_methods_to_run:
        print("\n" + "-" * 30 + f"\n--- Optimierung mit {method} ---")
        try:
             kwargs = {'maxiter': de_maxiter, 'popsize': de_popsize} if method=='DE' else {}
             result = optimizer.optimize(method=method, workers=robust_num_workers, **kwargs)
             if result and result.success:
                 print(f"Status: {result.message}"); print(f"Optimale Allokation: {np.round(result.x, 4)}"); print(f"Maximaler Nutzen: {-result.fun:.4f}")
                 results_dict[method] = {'utility': -result.fun, 'alloc': result.x}
             else: status = result.message if result else "Kein Ergebnis"; print(f"Optimierung fehlgeschlagen. Status: {status}")
        except Exception as e: logging.error(f"Fehler bei {method}-Optimierung: {e}", exc_info=True)

    # --- Vergleich der Ergebnisse ---
    print("\n" + "-" * 30 + "\n--- Vergleich der Optimierungsergebnisse ---")
    if results_dict:
        best_util = -np.inf; best_method = ""
        for method, result in results_dict.items(): print(f"Methode: {method:<15} | Nutzen: {result['utility']:.4f}");
        # Find best method robustly even if some utilities are NaN/inf
        valid_results = {m: r for m, r in results_dict.items() if np.isfinite(r['utility'])}
        if valid_results:
             best_method = max(valid_results, key=lambda k: valid_results[k]['utility'])
             print(f"\nBeste gefundene Lösung mit Methode: {best_method} (Nutzen: {valid_results[best_method]['utility']:.4f})")
        else: print("\nKeine gültigen Ergebnisse zum Finden der besten Methode.")
    else: print("Keine erfolgreichen Optimierungen zum Vergleichen.")

    # --- Sensitivitätsanalyse ---
    print("\n" + "-" * 30 + "\n--- Sensitivitätsanalyse (Budget) ---")
    try:
        B_min = optimizer.total_budget * sens_min_factor; B_max = optimizer.total_budget * sens_max_factor
        min_possible_budget = np.sum(optimizer.lower_bounds)
        if B_min < min_possible_budget: logging.warning(f"Min Budget ({B_min:.2f}) < Summe LB ({min_possible_budget:.2f}). Anpassen."); B_min = min_possible_budget
        if B_max < B_min + FLOAT_PRECISION: logging.warning("Max Budget < Min Budget. Überspringe Sensitivitätsanalyse.")
        else: B_values = np.linspace(B_min, B_max, sens_num_steps); B_sens, allocs_sens, utils_sens = optimizer.sensitivity_analysis(B_values, method='SLSQP'); optimizer.plot_sensitivity(B_sens, allocs_sens, utils_sens, method='SLSQP', interactive=args.interactive)
    except Exception as e: logging.error(f"Fehler bei Sensitivitätsanalyse: {e}", exc_info=True)
    finally: plt.close('all')

    # --- Top Synergien ---
    print("\n" + "-" * 30 + "\n--- Wichtigste Synergieeffekte ---")
    try:
        top_synergies = optimizer.identify_top_synergies_correct() # Verwendet Default N
        if top_synergies:
            for pair, value in top_synergies: print(f"- {optimizer.investment_labels[pair[0]]:<15} & {optimizer.investment_labels[pair[1]]:<15}: {value:.4f}")
        else: print("Keine Synergieeffekte gefunden/berechnet.")
    except Exception as e: logging.error(f"Fehler bei Top-Synergien: {e}", exc_info=True)

    # --- Robustheitsanalyse ---
    print("\n" + "-" * 30 + "\n--- Robustheitsanalyse ---")
    try:
        df_robust_stats, df_robust = optimizer.robustness_analysis(num_simulations=robust_sims, method='DE', variation_percentage=robust_var_perc, parallel=True, num_workers=robust_num_workers)
        print("\n(Deskriptive Statistik der robusten Allokationen)")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:.4f}'.format):
             print(df_robust_stats[['mean', 'std', '5%', '50%', '95%', 'cv']].fillna('-'))
        optimizer.plot_robustness_analysis(df_robust)
    except Exception as e: logging.error(f"Fehler bei Robustheitsanalyse: {e}", exc_info=True)
    finally: plt.close('all')

    # --- Pareto-Approximation ---
    print("\n" + "-" * 30 + "\n--- Pareto-Front Approximation ---")
    pareto_results_df = pd.DataFrame()
    try:
        # Definiere die Zielfunktionen hier wieder (oder global/in Klasse)
        def risk_func_main(x): return np.var(x)
        def sustainability_func_main(x): n = len(x); mean = np.mean(x) if n > 0 else 0; return -np.sum((x - mean)**2) if n > 0 else 0

        pareto_results_df = optimizer.find_pareto_approximation(
            risk_func=risk_func_main,
            sustainability_func=sustainability_func_main,
            num_weight_samples=pareto_samples,
            method='SLSQP' # SLSQP ist oft gut für die Einzelprobleme
        )

        if not pareto_results_df.empty:
             print(f"Gefundene nicht-dominierte Lösungen: {len(pareto_results_df)}")
             # Zeige einige Beispielpunkte (z.B. höchste Utility, niedrigstes Risiko)
             print("\nBeispiel-Lösungen (höchste Utility / niedrigstes Risiko):")
             if 'utility' in pareto_results_df.columns and not pareto_results_df['utility'].isnull().all():
                 print("- Max Utility:\n", pareto_results_df.loc[pareto_results_df['utility'].idxmax()])
             if 'risk' in pareto_results_df.columns and not pareto_results_df['risk'].isnull().all():
                 print("\n- Min Risk:\n", pareto_results_df.loc[pareto_results_df['risk'].idxmin()])
             # Plot
             optimizer.plot_pareto_approximation(pareto_results_df, interactive=args.interactive)
        else: print("Keine nicht-dominierten Lösungen gefunden.")

    except Exception as e: logging.error(f"Fehler bei Pareto-Approximation: {e}", exc_info=True)
    finally: plt.close('all') # Schließe Plots sicher

    # --- Parametersensitivitätsanalysen ---
    print("\n" + "-" * 30 + "\n--- Parametersensitivitätsanalysen ---")
    try:
        def get_param_range_main(current_value, min_factor, max_factor, num_steps, is_log_utility_roi):
            # Ensure current_value is a float for calculations
            current_value = float(current_value) if isinstance(current_value, (int, float, np.number)) else 0.0
            abs_min = MIN_INVESTMENT * 1.01 if is_log_utility_roi else 0.0
            p_min = max(abs_min, current_value * min_factor); p_max = current_value * max_factor
            # Ensure range > 0, handle cases where current_value is 0 or negative
            if p_max <= p_min + FLOAT_PRECISION:
                if current_value >= 0: p_max = p_min + FLOAT_PRECISION*10
                else: p_min = p_max - FLOAT_PRECISION*10 # Handle negative current_value case
            if p_min > p_max: p_min = p_max - FLOAT_PRECISION*10 # Ensure min <= max
            return np.linspace(p_min, p_max, num_steps)

        # ROI (wenn relevant)
        if optimizer.n > 0 and optimizer.utility_function_type != 's_curve':
             is_log = optimizer.utility_function_type == 'log'
             roi0_values = get_param_range_main(optimizer.roi_factors[0], param_sens_min_factor, param_sens_max_factor, param_sens_steps, is_log)
             optimizer.plot_parameter_sensitivity(roi0_values, 'roi_factors', parameter_index=0, method='SLSQP', interactive=args.interactive)
        # Synergie
        if optimizer.n > 1:
             current_syn01 = optimizer.synergy_matrix[0, 1]; syn01_min = max(0, current_syn01 * param_sens_min_factor)
             syn01_max = current_syn01 * param_sens_max_factor
             if syn01_max <= syn01_min + FLOAT_PRECISION: syn01_max = syn01_min + FLOAT_PRECISION*10;
             if current_syn01 < FLOAT_PRECISION and syn01_max < 0.1: syn01_max = max(0.1, syn01_max) # Test small positive range if current is 0
             synergy01_values = np.linspace(syn01_min, syn01_max, param_sens_steps)
             optimizer.plot_parameter_sensitivity(synergy01_values, 'synergy_matrix', parameter_index=(0, 1), method='SLSQP', interactive=args.interactive)
        # Parameter c (wenn relevant)
        if optimizer.n > 0 and optimizer.utility_function_type == 'quadratic' and optimizer.c is not None:
             c0_values = get_param_range_main(optimizer.c[0], param_sens_min_factor, param_sens_max_factor, param_sens_steps, False)
             optimizer.plot_parameter_sensitivity(c0_values, 'c', parameter_index=0, method='SLSQP', interactive=args.interactive)
        # TODO: Add plots for s_curve parameters if needed
    except Exception as e: logging.error(f"Fehler bei Parametersensitivitätsanalyse: {e}", exc_info=True)
    finally: plt.close('all')

    logging.info("-" * 30 + "\nAlle Analysen abgeschlossen.")

if __name__ == "__main__":
    main()