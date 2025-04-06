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
    print(f"Fehlende Pakete: {', '.join(missing_packages)}. Bitte installieren Sie sie mit pip install {' '.join(missing_packages)} oder pip install -r requirements.txt.")
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
            # Last resort: clip original and scale (might violate bounds slightly)
            x0 = np.clip(initial_allocation, lb, ub); x0 *= (total_budget / np.sum(x0)) if np.sum(x0)>FLOAT_PRECISION else 1; return np.clip(x0, lb, ub)
        elif remaining_budget > FLOAT_PRECISION:
            allocatable_range = ub - lb
            positive_range_indices = np.where(allocatable_range > FLOAT_PRECISION)[0]
            if len(positive_range_indices) > 0:
                allocatable_sum = np.sum(allocatable_range[positive_range_indices])
                if allocatable_sum > FLOAT_PRECISION:
                    proportions = allocatable_range[positive_range_indices] / allocatable_sum
                    x0[positive_range_indices] += remaining_budget * proportions
                else: # Falls Range 0, aber Indices existieren (lb=ub), verteile gleichmäßig auf die die erhöht werden *könnten*
                    increase_possible_indices = np.where(ub > lb + FLOAT_PRECISION)[0]
                    if len(increase_possible_indices) > 0 :
                        x0[increase_possible_indices] += remaining_budget / len(increase_possible_indices)
                    else: # If nothing can be increased, just distribute to all
                         x0 += remaining_budget / n

        x0 = np.clip(x0, lb, ub)
        current_sum = np.sum(x0)
        if not np.isclose(current_sum, total_budget, atol=tol * 100):
            logging.error(f"Anpassung der Startallokation fehlgeschlagen. Budgetsumme {current_sum} weicht stark von {total_budget} ab.")
            # Last resort again
            x0_final = np.clip(initial_allocation, lb, ub)
            current_sum_final = np.sum(x0_final)
            if current_sum_final > FLOAT_PRECISION and not np.isclose(current_sum_final, total_budget):
                x0_final *= (total_budget / current_sum_final)
            return np.clip(x0_final, lb, ub)

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
                # Vary x0 additively around the original value
                x0_var = np.abs(s_curve_x0 * variation_percentage) + FLOAT_PRECISION # Base variation scale
                s_curve_x0_sim = s_curve_x0 + np.random.uniform(-x0_var, x0_var, size=s_curve_x0.shape)
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

            # Validierung (Basis + Utility-spezifisch) - Check with potentially unadjusted initial alloc first
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds,
                            np.clip(initial_allocation, self.lower_bounds, self.upper_bounds), # Use clipped for initial check
                            self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0)

            # Adjust initial allocation *after* basic validation
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
            if 'c' in locals(): # Check if 'c' was passed as an argument (even if None)
                 if c is not None: # Wenn c nicht None ist (neuer Wert oder Array)
                     new_c = np.array(c)
                     if len(new_c) != self.n: raise ValueError("Dimension von c passt nicht.")
                     self.c = new_c; parameters_updated = True; utility_params_changed = True; logging.debug("Parameter c aktualisiert/gesetzt.")
                 elif self.c is not None: # Wenn c explizit None ist und vorher nicht None war
                     self.c = None; parameters_updated = True; utility_params_changed = True; logging.debug("Parameter c entfernt.")
            # Handling von S-Kurven Parametern (similar logic)
            if 's_curve_L' in locals():
                new_L = np.array(s_curve_L) if s_curve_L is not None else None
                if new_L is not None and len(new_L) != self.n: raise ValueError("Dimension von s_curve_L passt nicht.")
                # Check if changed before setting flags
                if not np.array_equal(self.s_curve_L, new_L, equal_nan=True):
                    self.s_curve_L = new_L; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_L aktualisiert.")
            if 's_curve_k' in locals():
                 new_k = np.array(s_curve_k) if s_curve_k is not None else None
                 if new_k is not None and len(new_k) != self.n: raise ValueError("Dimension von s_curve_k passt nicht.")
                 if not np.array_equal(self.s_curve_k, new_k, equal_nan=True):
                     self.s_curve_k = new_k; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_k aktualisiert.")
            if 's_curve_x0' in locals():
                 new_x0 = np.array(s_curve_x0) if s_curve_x0 is not None else None
                 if new_x0 is not None and len(new_x0) != self.n: raise ValueError("Dimension von s_curve_x0 passt nicht.")
                 if not np.array_equal(self.s_curve_x0, new_x0, equal_nan=True):
                    self.s_curve_x0 = new_x0; parameters_updated = True; utility_params_changed = True; logging.debug("s_curve_x0 aktualisiert.")

            # --- Initial Allocation anpassen ---
            current_initial_allocation = self.initial_allocation if initial_allocation is None else np.array(initial_allocation)
            if initial_allocation is not None:
                if len(current_initial_allocation) != self.n: raise ValueError("Länge der neuen initial_allocation passt nicht.")
                self.initial_allocation = adjust_initial_guess(current_initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget); parameters_updated = True; logging.debug("initial_allocation aktualisiert/angepasst.")
            elif parameters_updated: # Wenn andere Parameter geändert wurden, passe alte Allokation an
                logging.debug(f"Alte Startallokation vor Anpassung: {self.initial_allocation}")
                self.initial_allocation = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget); logging.debug(f"Bestehende initial_allocation angepasst: {self.initial_allocation}")

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
                # Check for valid inputs to log
                if np.any(x < MIN_INVESTMENT):
                     logging.debug(f"Warnung: x < MIN_INVESTMENT in log utility calculation for x={x}. Clipping.")
                return np.sum(self.roi_factors * np.log(np.maximum(x, MIN_INVESTMENT)))
            elif self.utility_function_type == 'quadratic':
                if self.c is None: raise ValueError("'c' fehlt für quadratic.")
                return np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
            elif self.utility_function_type == 's_curve':
                if self.s_curve_L is None or self.s_curve_k is None or self.s_curve_x0 is None: raise ValueError("S-Kurven Parameter fehlen.")
                # Clip exponent to avoid overflow/underflow
                # Ensure x>=0 before calculating exponent for stability
                exponent_term = np.clip(-self.s_curve_k * (np.maximum(x, 0) - self.s_curve_x0), -700, 700)
                return np.sum(self.s_curve_L / (1 + np.exp(exponent_term)))
            else: raise ValueError(f"Unbekannter Typ '{self.utility_function_type}'.")
        except FloatingPointError as fpe:
            logging.error(f"FloatingPointError in _calculate_utility (Typ: {self.utility_function_type}) für x={x}: {fpe}", exc_info=False)
            return -np.inf # Return negative infinity to signal error
        except Exception as e:
            logging.error(f"Fehler in _calculate_utility (Typ: {self.utility_function_type}) für x={x}: {e}", exc_info=False)
            return -np.inf # Return negative infinity to signal error to minimizer

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        """Zielfunktion (zu minimieren) mit Budget-Penalty."""
        utility = self._calculate_utility(x)
        if np.isinf(utility) and utility < 0: return np.inf # Error in utility -> return positive infinity for minimization
        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy
        budget_diff = np.sum(x) - self.total_budget
        penalty_budget = penalty_coeff * (budget_diff ** 2)
        result = -total_utility + penalty_budget
        # Return large finite number if calculation resulted in inf/nan
        if not np.isfinite(result):
            logging.debug(f"Non-finite result in objective_with_penalty: {-total_utility=}, {penalty_budget=}")
            return np.finfo(float).max / 10
        return result

    def objective_without_penalty(self, x):
        """Zielfunktion (zu minimieren) ohne explizite Penalties."""
        utility = self._calculate_utility(x)
        if np.isinf(utility) and utility < 0: return np.inf # Error in utility -> return positive infinity for minimization
        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy
        result = -total_utility
        # Return large finite number if calculation resulted in inf/nan
        if not np.isfinite(result):
            logging.debug(f"Non-finite result in objective_without_penalty: {-total_utility=}")
            return np.finfo(float).max / 10
        return result

    def identify_top_synergies_correct(self, top_n=DEFAULT_TOP_N_SYNERGIES):
        """Identifiziert die Top N Synergieeffekte."""
        if top_n <= 0 or self.n < 2: return []
        try:
            # Create a copy and fill diagonal to ignore self-synergy
            synergy_copy = self.synergy_matrix.copy(); np.fill_diagonal(synergy_copy, -np.inf)
            # Get upper triangle indices (k=1 excludes diagonal)
            triu_indices = np.triu_indices(self.n, k=1); synergy_values = synergy_copy[triu_indices]
            num_synergies = len(synergy_values); actual_top_n = min(top_n, num_synergies)
            if actual_top_n == 0: return []
            # Sort indices by synergy value (ascending)
            sorted_indices = np.argsort(synergy_values);
            # Get indices of the top N largest values
            top_indices = sorted_indices[-actual_top_n:]
            top_synergies = []
            # Iterate through the top indices (from largest to smallest)
            for flat_idx in reversed(top_indices):
                # Map flat index back to row/col
                row, col = triu_indices[0][flat_idx], triu_indices[1][flat_idx]
                value = synergy_copy[row, col] # Get value from the modified copy
                if np.isfinite(value): top_synergies.append(((row, col), value))
            return top_synergies
        except Exception as e:
            logging.error(f"Fehler bei Top-Synergien: {e}", exc_info=True); return []

    def optimize(self, method='SLSQP', max_retries=3, workers=None, penalty_coeff=1e8, **kwargs):
        """Führt die Optimierung mit der gewählten Methode durch."""
        # Bounds should ensure positivity for log utility
        scipy_bounds = Bounds(lb=np.maximum(self.lower_bounds, 0), ub=self.upper_bounds)
        # Tuple bounds needed for some methods (ensure lower <= upper after max(0, MIN_INVESTMENT))
        tuple_bounds = get_bounds(self.lower_bounds, self.upper_bounds)

        # Define constraints (use NonLinear for broader compatibility with minimize)
        constraints_for_minimize = NonlinearConstraint(fun=lambda x: np.sum(x), lb=self.total_budget, ub=self.total_budget)
        # DE prefers LinearConstraint if available
        constraints_for_de = LinearConstraint(np.ones((1, self.n)), lb=self.total_budget, ub=self.total_budget)

        # Check SciPy version for features
        scipy_version = version.parse(scipy.__version__)
        de_constraints_supported = scipy_version >= version.parse("1.7.0")
        de_workers_supported = scipy_version >= version.parse("1.4.0")
        actual_workers = workers if workers is not None and workers > 0 and de_workers_supported else 1
        de_updating = 'deferred' if actual_workers > 1 and de_workers_supported else 'immediate'

        # Determine objective function and constraints based on method
        use_penalty_objective = method in ['TNC', 'L-BFGS-B', 'Nelder-Mead']
        constraints_arg = None # Constraints object to pass to the optimizer
        if method == 'DE':
            if de_constraints_supported:
                 use_penalty_objective = False; constraints_arg = constraints_for_de; logging.debug("DE: Native Constraints.")
            else:
                 use_penalty_objective = True; logging.warning("DE: Fallback auf Penalty (alte SciPy < 1.7?).")
        elif method in ['SLSQP', 'trust-constr']: use_penalty_objective = False; constraints_arg = constraints_for_minimize
        elif method == 'BasinHopping':
            # BasinHopping uses its minimizer_kwargs for constraints/bounds
            use_penalty_objective = False

        objective_func = self.objective_with_penalty if use_penalty_objective else self.objective_without_penalty
        obj_args = (penalty_coeff,) if use_penalty_objective else ()

        # --- Define options, merge with kwargs ---
        def get_options(method_key, base_options, user_kwargs):
            opts = base_options.copy()
            # Only update if 'options' is provided for this specific method in user_kwargs
            if 'options' in user_kwargs and isinstance(user_kwargs['options'], dict):
                 opts.update(user_kwargs['options'])
            return opts

        slsqp_options = get_options('slsqp', {'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION}, kwargs)
        # DE takes kwargs directly, filter out 'options' and 'constraints' which are handled separately
        de_opts_base = {'strategy': 'best1bin', 'maxiter': 1000, 'popsize': 15, 'tol': 0.01,'mutation': (0.5, 1), 'recombination': 0.7,
                        'updating': de_updating, 'workers': actual_workers,
                         # Polish only if not using native constraints or if explicitly asked for
                        'polish': (not de_constraints_supported or kwargs.get('polish', False)) and de_constraints_supported is not None ,
                        'init': 'latinhypercube'}
        de_opts_base.update({k:v for k,v in kwargs.items() if k not in ['options', 'constraints', 'polish']}) # DE takes kwargs directly

        # BasinHopping options
        basinhopping_opts_base = {'niter': 100, 'stepsize': 0.5, 'T': 1.0, 'disp': False, 'niter_success': 10}
        # Define minimizer_kwargs specifically for BasinHopping's local step
        basinhopping_minimizer_kwargs = {
            'method': 'SLSQP', # Common choice for local minimization
            'bounds': tuple_bounds, # Use tuple bounds for SLSQP inside BH
            'constraints': constraints_for_minimize, # Use nonlinear constraints
            'options': {'maxiter': 200, 'ftol': FLOAT_PRECISION, 'disp': False},
            'args': () # Objective for the *local* minimizer (SLSQP) doesn't need penalty
        }
        # Allow overriding minimizer_kwargs if provided in main kwargs
        user_minimizer_kwargs = kwargs.get('minimizer_kwargs', {})
        if user_minimizer_kwargs and isinstance(user_minimizer_kwargs, dict):
            # Merge smartly: allow overriding method, bounds, constraints, options, args
            if 'method' in user_minimizer_kwargs: basinhopping_minimizer_kwargs['method'] = user_minimizer_kwargs['method']
            if 'bounds' in user_minimizer_kwargs: basinhopping_minimizer_kwargs['bounds'] = user_minimizer_kwargs['bounds']
            if 'constraints' in user_minimizer_kwargs: basinhopping_minimizer_kwargs['constraints'] = user_minimizer_kwargs['constraints']
            if 'options' in user_minimizer_kwargs and isinstance(user_minimizer_kwargs['options'], dict):
                basinhopping_minimizer_kwargs['options'].update(user_minimizer_kwargs['options'])
            if 'args' in user_minimizer_kwargs: basinhopping_minimizer_kwargs['args'] = user_minimizer_kwargs['args']

        basinhopping_opts_base['minimizer_kwargs'] = basinhopping_minimizer_kwargs
        # Update BH base options with other kwargs, excluding those handled
        basinhopping_opts_base.update({k:v for k,v in kwargs.items() if k not in ['options', 'minimizer_kwargs']})

        tnc_options = get_options('tnc', {'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION, 'gtol': FLOAT_PRECISION*10}, kwargs)
        lbfgsb_options = get_options('lbfgsb', {'disp': False, 'maxiter': 15000, 'ftol': 2.22e-09, 'gtol': 1e-5}, kwargs)
        trustconstr_options = get_options('trustconstr', {'disp': False, 'maxiter': 1000, 'gtol': FLOAT_PRECISION, 'xtol': FLOAT_PRECISION}, kwargs)
        neldermead_options = get_options('neldermead', {'disp': False, 'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4}, kwargs)


        # Lambda definitions using current state
        optimization_methods = {
            'SLSQP': lambda x0: minimize(objective_func, x0, method='SLSQP', bounds=tuple_bounds, constraints=constraints_arg, options=slsqp_options, args=obj_args),
            'DE': lambda x0: differential_evolution(objective_func, tuple_bounds, constraints=constraints_arg if de_constraints_supported else (), args=obj_args if use_penalty_objective else (), **de_opts_base),
            # BasinHopping calls the *non-penalty* objective directly, its internal minimizer handles constraints
            'BasinHopping': lambda x0: basinhopping(self.objective_without_penalty, x0, **basinhopping_opts_base),
            'TNC': lambda x0: minimize(objective_func, x0, method='TNC', bounds=tuple_bounds, options=tnc_options, args=obj_args),
            'L-BFGS-B': lambda x0: minimize(objective_func, x0, method='L-BFGS-B', bounds=tuple_bounds, options=lbfgsb_options, args=obj_args),
            'trust-constr': lambda x0: minimize(objective_func, x0, method='trust-constr', bounds=scipy_bounds, constraints=constraints_arg, options=trustconstr_options, args=obj_args),
            # Nelder-Mead does not formally support constraints, rely on penalty objective and bounds
            'Nelder-Mead': lambda x0: minimize(objective_func, x0, method='Nelder-Mead', bounds=tuple_bounds, options=neldermead_options, args=obj_args),
        }

        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods: raise ValueError(f"Methode {method} nicht unterstützt. Optionen: {supported_methods}")

        logging.info(f"Starte Optimierung mit Methode: {method}")
        if method == 'DE': logging.info(f"DE Details: Native Constraints = {de_constraints_supported}, Workers = {actual_workers}, Polish = {de_opts_base.get('polish', 'N/A')}")
        if method == 'BasinHopping': logging.info(f"BasinHopping Details: Minimizer={basinhopping_minimizer_kwargs.get('method', 'N/A')}")


        current_initial_allocation = self.initial_allocation.copy()
        for attempt in range(1, max_retries + 1):
            try:
                logging.debug(f"Optimierungsversuch {attempt}/{max_retries} mit Methode {method}...")
                current_method_func = optimization_methods[method]
                # Ensure the initial guess is valid for the objective function (e.g., > 0 for log)
                if self.utility_function_type == 'log':
                    current_initial_allocation = np.maximum(current_initial_allocation, MIN_INVESTMENT)
                    # Also re-adjust sum if needed after clipping
                    if not np.isclose(np.sum(current_initial_allocation), self.total_budget):
                         current_initial_allocation = adjust_initial_guess(current_initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)

                logging.debug(f"Versuch {attempt} Startpunkt: {np.round(current_initial_allocation, 5)}, Summe: {np.sum(current_initial_allocation):.6f}")

                opt_result = current_method_func(current_initial_allocation) # Pass current start point

                # --- Ergebnisextraktion ---
                if method == 'BasinHopping':
                    x_opt = opt_result.x
                    # BH fun is already the non-penalty value from the best local minimum
                    fun_opt_val = opt_result.fun
                    success = opt_result.lowest_optimization_result.success if hasattr(opt_result, 'lowest_optimization_result') and hasattr(opt_result.lowest_optimization_result, 'success') else False
                    # Attempt to get a meaningful message
                    message = opt_result.message
                    if isinstance(message, (list, tuple)): message = " ".join(map(str, message))
                    if not success and hasattr(opt_result, 'lowest_optimization_result') and hasattr(opt_result.lowest_optimization_result, 'message'):
                         message += f" (Local minimizer msg: {opt_result.lowest_optimization_result.message})"

                else: # Methods returning standard SciPy OptimizeResult
                    x_opt = opt_result.x
                    # Recalculate objective *without* penalty if penalty was used
                    fun_opt_val = self.objective_without_penalty(x_opt) if use_penalty_objective else opt_result.fun
                    success = opt_result.success
                    message = opt_result.message if hasattr(opt_result, 'message') else 'Optimierung abgeschlossen.'

                # --- Ergebnisbereinigung und Prüfung ---
                # Clip final result to bounds as some methods might slightly violate them
                x_opt = np.clip(x_opt, np.maximum(self.lower_bounds, 0), self.upper_bounds)
                # Final adjustment for budget constraint if slightly off and method doesn't enforce it strictly
                current_sum = np.sum(x_opt)
                if not np.isclose(current_sum, self.total_budget, atol=BUDGET_PRECISION * 10):
                     logging.debug(f"Ergebnis von {method} nicht exakt auf Budget ({current_sum:.6f}). Versuche finale Anpassung.")
                     x_opt = adjust_initial_guess(x_opt, self.lower_bounds, self.upper_bounds, self.total_budget, tol=BUDGET_PRECISION)


                # --- Constraint-Prüfung nach Bereinigung ---
                final_sum = np.sum(x_opt)
                budget_ok = np.isclose(final_sum, self.total_budget, atol=BUDGET_PRECISION)
                lower_bounds_ok = np.all(x_opt >= self.lower_bounds - BOUNDS_PRECISION)
                upper_bounds_ok = np.all(x_opt <= self.upper_bounds + BOUNDS_PRECISION)
                # Check for NaN/Inf in result
                valid_values = np.all(np.isfinite(x_opt))

                constraints_satisfied = budget_ok and lower_bounds_ok and upper_bounds_ok and valid_values

                logging.debug(f"Versuch {attempt} Ergebnis roh: Erfolg={success}, Budget OK={budget_ok} (Sum={final_sum:.6f}), Bounds OK={lower_bounds_ok and upper_bounds_ok}, Valide Werte={valid_values}, Nachricht='{message}'")

                if success and constraints_satisfied:
                    logging.info(f"Optimierung mit {method} erfolgreich in Versuch {attempt}.")
                    # Recalculate final 'fun' value based on cleaned x_opt
                    final_fun_val = self.objective_without_penalty(x_opt)
                    return OptimizationResult(x=x_opt, fun=final_fun_val, success=success, message=message, raw_result=opt_result) # fun is always without penalty here
                elif success and not constraints_satisfied:
                     logging.warning(f"Optimierung {method} nominell erfolgreich, aber Constraints verletzt nach Bereinigung (Budget Ok={budget_ok}, Bounds Ok={lower_bounds_ok and upper_bounds_ok}, Valid={valid_values}). Retry...")
                elif not success:
                     logging.warning(f"Optimierung {method} nicht erfolgreich (Nachricht: {message}). Retry...")
                elif not valid_values:
                     logging.warning(f"Optimierung {method} lieferte ungültige Werte (NaN/Inf). Retry...")


            except (ValueError, FloatingPointError, RuntimeError) as e:
                 logging.error(f"Optimierungsversuch {attempt} ({method}) fehlgeschlagen mit Fehler: {e}", exc_info=False)
            except Exception as e:
                 logging.error(f"Unerwarteter Fehler in Optimierungsversuch {attempt} ({method}): {e}", exc_info=True)

            # Prepare for next attempt if not the last one
            if attempt < max_retries:
                 # Perturb the *original* initial allocation for the next try
                 jitter = np.random.normal(0, self.total_budget * 0.02 * attempt, self.n) # Increase jitter slightly
                 perturbed_start = self.initial_allocation + jitter
                 current_initial_allocation = adjust_initial_guess(perturbed_start, self.lower_bounds, self.upper_bounds, self.total_budget)
                 logging.debug(f"Startpunkt leicht verändert für Versuch {attempt+1}.")

        logging.error(f"Optimierung mit Methode {method} nach {max_retries} Versuchen fehlgeschlagen.")
        return OptimizationResult(x=self.initial_allocation, fun=np.nan, success=False, message=f"Optimierung fehlgeschlagen nach {max_retries} Versuchen.") # Return a failed result object


    def sensitivity_analysis(self, B_values, method='SLSQP', tol=BUDGET_PRECISION, **kwargs):
        """Führt eine Sensitivitätsanalyse für verschiedene Budgets durch."""
        allocations, max_utilities = [], []
        # Store original state completely
        original_state = {
            'budget': self.total_budget, 'initial': self.initial_allocation.copy(),
            'roi': self.roi_factors.copy(), 'syn': self.synergy_matrix.copy(),
            'lb': self.lower_bounds.copy(), 'ub': self.upper_bounds.copy(),
            'utility_type': self.utility_function_type, 'c': self.c.copy() if self.c is not None else None,
            'L': self.s_curve_L.copy() if self.s_curve_L is not None else None,
            'k': self.s_curve_k.copy() if self.s_curve_k is not None else None,
            'x0': self.s_curve_x0.copy() if self.s_curve_x0 is not None else None
        }
        logging.info(f"Starte Sensitivitätsanalyse für Budgets [{min(B_values):.2f}-{max(B_values):.2f}] ({len(B_values)} Schritte, Methode: {method}).")
        last_successful_alloc = original_state['initial']

        for i, B_new in enumerate(B_values):
            logging.debug(f"Sensitivitätsanalyse Schritt {i+1}/{len(B_values)}, Budget = {B_new:.4f}")
            # Check if budget is feasible w.r.t bounds
            sum_lb = np.sum(self.lower_bounds)
            sum_ub = np.sum(self.upper_bounds)
            if B_new < sum_lb - BUDGET_PRECISION:
                logging.warning(f"Budget {B_new:.2f} < Summe Lower Bounds ({sum_lb:.2f}). Überspringe.")
                allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
                continue
            if B_new > sum_ub + BUDGET_PRECISION:
                 logging.warning(f"Budget {B_new:.2f} > Summe Upper Bounds ({sum_ub:.2f}). Überspringe.")
                 allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
                 continue

            # Use last successful allocation as a warm start, scaled to new budget
            current_sum = np.sum(last_successful_alloc);
            if current_sum > FLOAT_PRECISION:
                 new_x0_guess = last_successful_alloc * (B_new / current_sum)
            else: # Fallback if last allocation was zero
                 new_x0_guess = np.full(self.n, B_new / self.n)

            # Adjust guess to meet bounds and budget constraint of the *new* budget
            adjusted_x0 = adjust_initial_guess(new_x0_guess, self.lower_bounds, self.upper_bounds, B_new)
            logging.debug(f"Angepasster Startpunkt für B={B_new:.4f}: {np.round(adjusted_x0, 5)}")

            try:
                # Update only budget and initial allocation for this run
                # Use a temporary optimizer instance or carefully reset state
                temp_optimizer = InvestmentOptimizer(
                    roi_factors=original_state['roi'], synergy_matrix=original_state['syn'],
                    total_budget=B_new, # Set new budget
                    lower_bounds=original_state['lb'], upper_bounds=original_state['ub'],
                    initial_allocation=adjusted_x0, # Set adjusted initial guess
                    investment_labels=self.investment_labels,
                    log_level=logging.WARNING, # Reduce noise during analysis
                    utility_function_type=original_state['utility_type'], c=original_state['c'],
                    s_curve_L=original_state['L'], s_curve_k=original_state['k'], s_curve_x0=original_state['x0']
                )

                # Perform optimization with the temporary setup
                result = temp_optimizer.optimize(method=method, **kwargs)

                if result and result.success:
                    last_successful_alloc = result.x # Update for next warm start
                    allocations.append(result.x)
                    max_utilities.append(-result.fun) # Utility is -fun
                    logging.debug(f"Erfolg für B={B_new:.4f}. Nutzen: {-result.fun:.4f}")
                else:
                    logging.warning(f"Sensitivitätsanalyse für Budget {B_new:.2f} fehlgeschlagen. Nachricht: {result.message if result else 'N/A'}");
                    allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
                    # Optional: Reset last_successful_alloc to original if a step fails?
                    # last_successful_alloc = original_state['initial']

            except Exception as e:
                logging.error(f"Fehler bei Sensitivitätsanalyse Schritt B={B_new:.2f}: {e}", exc_info=True);
                allocations.append([np.nan]*self.n); max_utilities.append(np.nan)
                # last_successful_alloc = original_state['initial'] # Reset on error?

        # No need to reset self parameters as we used a temporary optimizer or were careful

        logging.info("Sensitivitätsanalyse abgeschlossen.")
        return B_values, allocations, max_utilities


    def robustness_analysis(self, num_simulations=DEFAULT_ROBUSTNESS_NUM_SIMULATIONS, method='DE', variation_percentage=DEFAULT_ROBUSTNESS_VARIATION_PERCENTAGE, parallel=True, num_workers=None):
        """ Führt eine Robustheitsanalyse durch. """
        if num_workers is None: cpu_count = os.cpu_count() or 1; num_workers = min(cpu_count, DEFAULT_ROBUSTNESS_NUM_WORKERS_FACTOR)
        actual_workers = num_workers if parallel else 1
        logging.info(f"Starte Robustheitsanalyse ({num_simulations} Sims, +/-{variation_percentage*100:.1f}%, Utility: {self.utility_function_type}, Method: {method}, Parallel: {parallel}, Workers: {actual_workers}).")

        # Pass all necessary parameters for the utility function calculation IN THE CORRECT ORDER for single_simulation
        args_template = (
             self.roi_factors, self.synergy_matrix, self.total_budget,
             self.lower_bounds, self.upper_bounds, self.initial_allocation, # Pass the adjusted initial alloc
             self.investment_labels, method, variation_percentage, InvestmentOptimizer,
             self.utility_function_type, self.c, self.s_curve_L, self.s_curve_k, self.s_curve_x0
        )

        results = []
        log_interval = max(1, num_simulations // 20); completed_count = 0
        executor_class = concurrent.futures.ProcessPoolExecutor if parallel and num_simulations > 1 else None # Use ProcessPool for CPU-bound tasks

        if executor_class:
            logging.info(f"Verwende ProcessPoolExecutor mit max. {actual_workers} Workern.")
            with executor_class(max_workers=actual_workers) as executor:
                # Map simulations to the executor
                futures = {executor.submit(single_simulation, i, *args_template): i for i in range(num_simulations)}
                for future in concurrent.futures.as_completed(futures):
                    sim_index = futures[future]
                    try:
                        result_alloc = future.result(); results.append(result_alloc);
                    except Exception as e:
                        logging.error(f"Fehler bei Ergebnisabruf für Robustheitssimulation {sim_index}: {e}", exc_info=False);
                        results.append([np.nan] * self.n) # Append NaNs on error

                    completed_count += 1
                    if completed_count % log_interval == 0:
                        logging.info(f"Robustheitsanalyse: {completed_count}/{num_simulations} abgeschlossen.")
        else: # Sequentiell
            logging.info("Führe Robustheitsanalyse sequentiell aus.")
            for sim in range(num_simulations):
                results.append(single_simulation(sim, *args_template)); completed_count += 1
                if completed_count % log_interval == 0:
                    logging.info(f"Robustheitsanalyse: {completed_count}/{num_simulations} abgeschlossen.")

        # --- Analyse der Ergebnisse ---
        df_results = pd.DataFrame(results, columns=self.investment_labels);
        num_failed = df_results.isna().any(axis=1).sum(); num_successful = len(df_results) - num_failed
        logging.info(f"Robustheitsanalyse abgeschlossen. {num_successful}/{num_simulations} Simulationen erfolgreich.")
        if num_failed > 0: logging.warning(f"{num_failed} Simulationen lieferten kein gültiges Ergebnis (NaN) und werden ignoriert.")

        df_clean = df_results.dropna()
        if not df_clean.empty:
            percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]; stats = df_clean.describe(percentiles=percentiles).transpose()
            mean_val = df_clean.mean()
            std_val = df_clean.std()
            # Calculate CV, handle division by zero or near-zero mean
            stats['cv'] = (std_val / mean_val.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
            stats['range'] = df_clean.max() - df_clean.min(); stats['iqr'] = stats['75%'] - stats['25%']
        else:
            logging.warning("Keine erfolgreichen Simulationen für Statistik.");
            stats = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'cv', 'range', 'iqr'], index=self.investment_labels)

        return stats, df_results # Return raw results as well


    def multi_criteria_optimization(self, alpha, beta, gamma, risk_func, sustainability_func, method='SLSQP'):
        """Führt eine multikriterielle Optimierung für EINE gegebene Gewichtung durch."""
        if not (callable(risk_func) and callable(sustainability_func)): raise TypeError("risk_func/sustainability_func müssen callable sein.")

        # Define the combined objective function to be *minimized*
        def multi_objective(x):
            # Calculate base utility (which is already negative in objective_without_penalty)
            neg_utility = self.objective_without_penalty(x) # This is -(utility + synergy)
            if np.isinf(neg_utility) or np.isnan(neg_utility):
                logging.debug(f"Ungültiger Nutzenwert ({neg_utility}) für x={x} in Multi-Obj.")
                return np.finfo(float).max / 10 # Return large value if base utility failed

            # Calculate risk (assuming higher is worse, so minimize it)
            try: risk = risk_func(x); risk = risk if np.isfinite(risk) else np.inf
            except Exception as e: logging.warning(f"Fehler risk_func: {e}"); risk = np.inf

            # Calculate sustainability (assuming higher is better, so minimize its negative)
            try: sustainability = sustainability_func(x); sustainability = sustainability if np.isfinite(sustainability) else -np.inf
            except Exception as e: logging.warning(f"Fehler sust_func: {e}"); sustainability = -np.inf

            # Combine: Minimize weighted sum of (-Utility), Risk, (-Sustainability)
            # Note: neg_utility is already -(Utility+Synergy)
            # So we minimize alpha*(-Utility) + beta*(Risk) + gamma*(-Sustainability)
            combined_value = alpha * neg_utility + beta * risk - gamma * sustainability

            if not np.isfinite(combined_value):
                 logging.debug(f"Ungültiger kombinierter Wert: U={neg_utility}, R={risk}, S={sustainability}, Gewichte={alpha,beta,gamma}")
                 return np.finfo(float).max / 10
            return combined_value

        # Bounds and constraints remain the same
        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        constraints = NonlinearConstraint(fun=lambda x: np.sum(x), lb=self.total_budget, ub=self.total_budget)
        logging.debug(f"Starte multikrit. Optimierung für Gewichte: U={alpha:.2f}, R={beta:.2f}, S={gamma:.2f} mit Methode {method}")

        # Use a robust optimization method like SLSQP or trust-constr
        try:
            # Initial guess adjustment
            initial_guess = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
            if self.utility_function_type == 'log': # Ensure positivity for log
                 initial_guess = np.maximum(initial_guess, MIN_INVESTMENT)
                 if not np.isclose(np.sum(initial_guess), self.total_budget):
                      initial_guess = adjust_initial_guess(initial_guess, self.lower_bounds, self.upper_bounds, self.total_budget)

            result = minimize(multi_objective, initial_guess, method=method, bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 1000, 'ftol': FLOAT_PRECISION})

            if result.success:
                final_x = np.clip(result.x, bounds[0][0], bounds[0][1]) # Clip to be sure
                final_x = adjust_initial_guess(final_x, self.lower_bounds, self.upper_bounds, self.total_budget) # Ensure budget

                # Recalculate objectives with the final allocation
                final_utility = -self.objective_without_penalty(final_x) # Get positive utility value
                final_risk = risk_func(final_x) if callable(risk_func) else np.nan
                final_sustainability = sustainability_func(final_x) if callable(sustainability_func) else np.nan
                logging.debug(f"Multikrit. Erfolg (U={alpha:.2f},R={beta:.2f},S={gamma:.2f}): U={final_utility:.4f}, R={final_risk:.4f}, S={final_sustainability:.4f}")
                return OptimizationResult(x=final_x, fun=result.fun, success=result.success, message=result.message, utility=final_utility, risk=final_risk, sustainability=final_sustainability)
            else:
                logging.debug(f"Multikrit. Optimierung fehlgeschlagen (U={alpha:.2f},R={beta:.2f},S={gamma:.2f}): {result.message}");
                return OptimizationResult(x=initial_guess, fun=np.nan, success=False, message=result.message, utility=np.nan, risk=np.nan, sustainability=np.nan) # Return failed result

        except Exception as e:
            logging.error(f"Fehler multikrit. Optimierung (U={alpha:.2f},R={beta:.2f},S={gamma:.2f}): {e}", exc_info=False);
            return OptimizationResult(x=self.initial_allocation, fun=np.nan, success=False, message=str(e), utility=np.nan, risk=np.nan, sustainability=np.nan) # Return failed result


    def generate_weights(self, num_objectives, num_samples):
        """Generiert Gewichtungsvektoren, die zu 1 summieren."""
        weights = []
        if num_objectives == 2:
            # Generate num_samples+1 points including 0 and 1
            for i in range(num_samples + 1):
                 w1 = i / num_samples; weights.append(np.array([w1, 1.0 - w1]))
        elif num_objectives == 3:
             # Use systematic approach (Simplex Lattice Design)
             # Determine steps needed for roughly num_samples
             # n_steps roughly sqrt(2 * num_samples) for triangular grid
             steps = int(round((2 * num_samples)**0.5)) # Adjust formula based on desired density
             steps = max(1, steps) # Need at least 1 step
             for i in range(steps + 1):
                 for j in range(steps - i + 1):
                     k = steps - i - j
                     weights.append(np.array([i, j, k]) / steps)
             # Ensure boundary weights are included explicitly
             for w_bound in [[1,0,0], [0,1,0], [0,0,1]]:
                 if not any(np.allclose(w, w_bound) for w in weights):
                     weights.append(np.array(w_bound))
             # Remove duplicates
             weights = np.unique(np.round(weights, decimals=5), axis=0)
             logging.info(f"Generiere {len(weights)} eindeutige Gewichtungsvektoren für 3 Ziele (target samples ~{num_samples}, steps={steps}).")
        else: # Fallback for other numbers of objectives
             weights = [np.full(num_objectives, 1.0 / num_objectives)] # Just one equal weight vector
        return [w for w in weights] # Return as list of arrays

    def is_dominated(self, solution_objectives, other_objectives_list):
        """
        Prüft, ob 'solution_objectives' von irgendeiner Lösung in 'other_objectives_list' dominiert wird.
        Assumes objectives are to be maximized.
        """
        sol = np.array(solution_objectives)
        for other in other_objectives_list:
            oth = np.array(other)
            # Check if 'other' is better or equal in all objectives AND strictly better in at least one
            if np.all(oth >= sol) and np.any(oth > sol):
                return True # sol is dominated by other
        return False # sol is not dominated by any in the list

    def filter_non_dominated(self, solutions_objectives):
        """
        Filtert eine Liste von Zielwert-Vektoren und gibt nur die nicht-dominierten zurück.
        Assumes objectives are formatted such that higher values are better for all.
        Input: List of tuples/lists, e.g., [(utility, -risk, sustainability), ...]
        Output: List of non-dominated objective vectors.
        """
        if not solutions_objectives: return []
        num_solutions = len(solutions_objectives)
        if num_solutions == 1: return solutions_objectives

        # Convert to numpy array for efficiency
        objectives_array = np.array(solutions_objectives)
        non_dominated_mask = np.ones(num_solutions, dtype=bool)

        for i in range(num_solutions):
            if not non_dominated_mask[i]: continue # Skip if already marked as dominated
            # Check against all others (that aren't already dominated)
            for j in range(i + 1, num_solutions):
                 if not non_dominated_mask[j]: continue
                 sol_i = objectives_array[i]
                 sol_j = objectives_array[j]

                 # Check if sol_j dominates sol_i
                 if np.all(sol_j >= sol_i) and np.any(sol_j > sol_i):
                     non_dominated_mask[i] = False
                     break # i is dominated, move to next i
                 # Check if sol_i dominates sol_j
                 elif np.all(sol_i >= sol_j) and np.any(sol_i > sol_j):
                     non_dominated_mask[j] = False # j is dominated

        # Extract non-dominated solutions
        non_dominated_set = [tuple(obj) for obj, mask in zip(solutions_objectives, non_dominated_mask) if mask]

        # Remove duplicates (using tuple representation for hashability)
        unique_solutions_set = set(non_dominated_set)
        unique_solutions = [list(sol) for sol in unique_solutions_set] # Convert back to list if needed

        logging.info(f"Filterung: {num_solutions} -> {sum(non_dominated_mask)} potenziell nicht dominiert -> {len(unique_solutions)} eindeutig nicht dominiert.")
        # Sort results for consistency (optional, e.g., by first objective)
        # unique_solutions.sort(key=lambda x: x[0], reverse=True)
        return unique_solutions


    def find_pareto_approximation(self, risk_func, sustainability_func, num_weight_samples=DEFAULT_PARETO_NUM_SAMPLES, method='SLSQP'):
        """Approximiert die Pareto-Front durch Iteration über verschiedene Gewichtungen."""
        logging.info(f"Starte Pareto-Front-Approximation ({num_weight_samples} Samples geplant, Methode: {method}).")
        weights_list = self.generate_weights(3, num_weight_samples) # 3 objectives: Utility, Risk, Sustainability
        results, successful_runs = [], 0

        # Store original state to restore later
        original_state = {
            'initial': self.initial_allocation.copy(),
             # Add other parameters if multi_criteria_optimization modifies them, though it shouldn't
        }

        for i, weights in enumerate(weights_list):
            alpha, beta, gamma = weights # alpha=Utility, beta=Risk, gamma=Sustainability
            logging.debug(f"Pareto Schritt {i+1}/{len(weights_list)} - Gewichte U:{alpha:.2f} R:{beta:.2f} S:{gamma:.2f}")

            # Reset initial allocation for each run? Or use previous result? Using original for now.
            self.initial_allocation = original_state['initial'].copy()

            result = self.multi_criteria_optimization(alpha, beta, gamma, risk_func, sustainability_func, method)

            if result and result.success:
                # Store relevant info: allocation and the THREE objective values
                results.append({
                    'allocation': result.x,
                    'utility': result.utility,
                    'risk': result.risk,
                    'sustainability': result.sustainability,
                    'alpha': alpha, 'beta': beta, 'gamma': gamma # Store weights for reference
                });
                successful_runs += 1
            else:
                 logging.debug(f"Pareto Schritt {i+1} fehlgeschlagen für Gewichte U:{alpha:.2f} R:{beta:.2f} S:{gamma:.2f}")

        logging.info(f"Pareto-Approximation: {successful_runs}/{len(weights_list)} Läufe erfolgreich.")
        if not results:
             logging.warning("Keine erfolgreichen Pareto-Läufe.");
             # Restore original state just in case
             self.initial_allocation = original_state['initial']
             return pd.DataFrame()

        # Prepare objective values for filtering: Maximize Utility, Minimize Risk, Maximize Sustainability
        # So we filter based on (utility, -risk, sustainability) where higher is better for all.
        objective_values = []
        valid_indices = [] # Keep track of original indices corresponding to objective_values
        for i, r in enumerate(results):
             # Ensure all objectives are finite before including
             if all(np.isfinite(v) for v in [r['utility'], r['risk'], r['sustainability']]):
                 objective_values.append((r['utility'], -r['risk'], r['sustainability']))
                 valid_indices.append(i)
             else:
                 logging.debug(f"Ignoriere Ergebnis {i} wegen ungültiger Zielwerte: U={r['utility']}, R={r['risk']}, S={r['sustainability']}")

        if not objective_values:
             logging.warning("Keine gültigen Zielwerte für Pareto-Filterung.");
             self.initial_allocation = original_state['initial'] # Restore
             return pd.DataFrame()

        # Filter the valid objectives to find the non-dominated set
        non_dominated_objectives = self.filter_non_dominated(objective_values)

        # --- Match non-dominated objectives back to original results ---
        # This is tricky due to potential floating point inaccuracies.
        # We find the original result that produced each non-dominated objective vector.
        final_results = []
        used_original_indices = set()
        tol = 1e-5 # Tolerance for matching objective vectors

        # Create a lookup from the original valid index to the objective vector tuple
        original_objectives_map = {valid_indices[i]: tuple(np.round(obj, 5)) for i, obj in enumerate(objective_values)}
        # Create a lookup for the non-dominated objectives (rounded tuples)
        non_dominated_tuples = {tuple(np.round(nd_obj, 5)) for nd_obj in non_dominated_objectives}


        for original_idx, obj_tuple in original_objectives_map.items():
             if obj_tuple in non_dominated_tuples:
                 if original_idx not in used_original_indices:
                     final_results.append(results[original_idx])
                     used_original_indices.add(original_idx)

        # Fallback matching (if rounding didn't work perfectly)
        if len(final_results) != len(non_dominated_objectives):
             logging.warning(f"Konnte nicht alle {len(non_dominated_objectives)} nicht-dominierten Punkte exakt zuordnen ({len(final_results)} gefunden). Versuche Näherungs-Matching.")
             # This part is more complex and potentially less reliable
             # For now, we proceed with the results found via tuple matching

        if not final_results:
            logging.warning("Keine nicht-dominierten Lösungen nach Filterung und Zuordnung.");
            self.initial_allocation = original_state['initial'] # Restore
            return pd.DataFrame()

        logging.info(f"Pareto-Approximation: {len(final_results)} nicht-dominierte Lösungen gefunden und zugeordnet.")
        df_results = pd.DataFrame(final_results).sort_values(by='utility', ascending=False).reset_index(drop=True)

        # Restore original state before returning
        self.initial_allocation = original_state['initial']
        return df_results


    # --- Plotting Methoden ---
    def plot_sensitivity(self, B_values, allocations, utilities, method='SLSQP', interactive=False):
        """Plottet die Ergebnisse der Sensitivitätsanalyse."""
        if not isinstance(B_values, (list, np.ndarray)) or len(B_values) == 0 or \
           not isinstance(allocations, (list, np.ndarray)) or len(allocations) == 0 or \
           not isinstance(utilities, (list, np.ndarray)) or len(utilities) == 0:
            logging.warning("Keine Daten für Sensitivitätsplot."); return

        # Ensure allocations is 2D if only one budget was run
        allocations = np.array(allocations)
        if allocations.ndim == 1: allocations = allocations.reshape(1, -1)

        df_alloc = pd.DataFrame(allocations, columns=self.investment_labels);
        df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})

        # Filter out rows where optimization failed (NaNs)
        valid_indices = df_util['Maximaler Nutzen'].notna() & (~df_alloc.isna().any(axis=1))
        if not valid_indices.any(): logging.warning("Keine gültigen Datenpunkte für Sensitivitätsplot."); return

        B_values_clean=df_util['Budget'][valid_indices].values
        allocations_clean=df_alloc[valid_indices].values;
        utilities_clean=df_util['Maximaler Nutzen'][valid_indices].values

        if allocations_clean.ndim == 1: allocations_clean = allocations_clean.reshape(1, -1) # Reshape again if only 1 valid point
        if len(B_values_clean) < 2: logging.warning("Weniger als 2 gültige Punkte für Sensitivitätsplot."); return

        logging.info("Erstelle Plot für Sensitivitätsanalyse...")
        fig=None
        try:
            if interactive and plotly_available:
                df_alloc_plt = pd.DataFrame(allocations_clean, columns=self.investment_labels); df_alloc_plt['Budget'] = B_values_clean
                df_util_plt = pd.DataFrame({'Budget': B_values_clean, 'Maximaler Nutzen': utilities_clean})
                # Plot allocations
                fig1 = px.line(df_alloc_plt, x='Budget', y=self.investment_labels, title=f'Allokationen vs. Budget ({method})', labels={'value': 'Betrag', 'variable': 'Bereich'}); fig1.show()
                # Plot utility
                fig2 = px.line(df_util_plt, x='Budget', y='Maximaler Nutzen', markers=True, title=f'Nutzen vs. Budget ({method})'); fig2.show()
            else:
                fig, ax1 = plt.subplots(figsize=(12, 7)); colors = plt.cm.viridis(np.linspace(0, 1, self.n))
                # Plot allocations on primary axis
                for i, label in enumerate(self.investment_labels): ax1.plot(B_values_clean, allocations_clean[:, i], label=label, color=colors[i], marker='o', linestyle='-', markersize=4, alpha=0.8)
                ax1.set_xlabel('Gesamtbudget'); ax1.set_ylabel('Investitionsbetrag', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.legend(loc='upper left', title='Bereiche'); ax1.grid(True, linestyle='--', alpha=0.6)
                # Plot utility on secondary axis
                ax2 = ax1.twinx(); ax2.plot(B_values_clean, utilities_clean, label='Maximaler Nutzen', color='tab:red', marker='x', linestyle='--', markersize=6); ax2.set_ylabel('Maximaler Nutzen', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red'); ax2.legend(loc='upper right')
                plt.title(f'Sensitivitätsanalyse: Allokation & Nutzen vs. Budget ({method})', pad=20); fig.tight_layout(); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Sensitivität: {e}", exc_info=True)
        finally:
            # Close the figure if it was created and not interactive Plotly
            if fig and not (interactive and plotly_available): plt.close(fig)

    def plot_robustness_analysis(self, df_results):
        """Plottet die Ergebnisse der Robustheitsanalyse."""
        if df_results is None or df_results.empty: logging.warning("Keine Daten für Robustheitsplot."); return
        df_clean = df_results.dropna(); num_successful_sims = len(df_clean)
        if num_successful_sims == 0: logging.warning("Keine gültigen Daten für Robustheitsplot."); return

        logging.info(f"Erstelle Plots für Robustheitsanalyse ({num_successful_sims} gültige Sims)...")
        try:
            # Boxplot
            df_melted = df_clean.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
            fig1, ax1 = plt.subplots(figsize=(max(8, self.n * 1.2), 6))
            sns.boxplot(x='Bereich', y='Investition', data=df_melted, palette='viridis', ax=ax1)
            ax1.set_xlabel('Investitionsbereich'); ax1.set_ylabel('Simulierter Investitionsbetrag'); ax1.set_title(f'Verteilung der Allokationen (Robustheit, n={num_successful_sims})')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right'); fig1.tight_layout(); plt.show();

            # Histograms / Density Plots
            num_cols = min(4, self.n)
            try:
                g = sns.FacetGrid(df_melted, col="Bereich", col_wrap=num_cols, sharex=False, sharey=False, height=3, aspect=1.2)
                g.map(sns.histplot, "Investition", kde=True, bins=15, color='skyblue', stat='density'); g.fig.suptitle(f'Histogramme der Allokationen (Robustheit, n={num_successful_sims})', y=1.03)
                g.set_titles("{col_name}"); plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show();
            except Exception as e_facet: logging.warning(f"Fehler beim Erstellen des FacetGrid/Histogramm-Plots: {e_facet}.")
            finally:
                 if 'g' in locals() and hasattr(g, 'fig'): plt.close(g.fig) # Ensure FacetGrid figure is closed


            # Pairplot (if not too many dimensions)
            if self.n <= 5:
                logging.debug("Erstelle Pairplot für Robustheit.")
                try:
                    g_pair = sns.pairplot(df_clean, kind='scatter', diag_kind='kde', plot_kws={'alpha':0.5, 's':20}, palette='viridis'); g_pair.fig.suptitle(f'Paarweise Beziehungen (Robustheit, n={num_successful_sims})', y=1.02)
                    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show();
                except Exception as e_pair: logging.warning(f"Fehler beim Erstellen des Pairplots: {e_pair}.")
                finally:
                    if 'g_pair' in locals() and hasattr(g_pair, 'fig'): plt.close(g_pair.fig) # Ensure Pairplot figure is closed
            elif self.n > 5: logging.info("Pairplot übersprungen (>5 Bereiche).")

        except Exception as e: logging.error(f"Fehler beim Plotten der Robustheit: {e}", exc_info=True)
        finally:
             # Close the individual figures if they exist
             if 'fig1' in locals() and fig1: plt.close(fig1)
             # plt.close('all') # Use specific closing if possible

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
        if parameter_values is None or len(parameter_values) == 0: logging.warning(f"Keine Werte für Parameter '{parameter_name}'. Plot übersprungen."); return

        # --- Store Original State ---
        # Store parameters that might be temporarily changed
        original_state = {'roi': self.roi_factors.copy(), 'syn': self.synergy_matrix.copy(),
                          'utility_type': self.utility_function_type,
                          'c': self.c.copy() if self.c is not None else None,
                          'L': self.s_curve_L.copy() if self.s_curve_L is not None else None,
                          'k': self.s_curve_k.copy() if self.s_curve_k is not None else None,
                          'x0': self.s_curve_x0.copy() if self.s_curve_x0 is not None else None,
                           # Also store initial allocation as it might be used as starting point
                          'initial': self.initial_allocation.copy()
                         }
        original_budget = self.total_budget # Budget usually stays constant here

        utilities, allocations = [], []; parameter_relevant = True; xlabel = f"Parameter: {parameter_name}" # Default xlabel

        # --- Parameter Specific Setup & Relevance Check ---
        param_label = "" # Specific label for the parameter being changed
        if parameter_name == 'roi_factors':
            if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 'roi_factors'."); return
            param_label = f'ROI Faktor für "{self.investment_labels[parameter_index]}"'
            xlabel = param_label
            if self.utility_function_type == 's_curve': logging.warning(f"ROI nicht relevant für S-Kurve. Überspringe '{param_label}' Sensitivität."); parameter_relevant = False
        elif parameter_name == 'synergy_matrix':
            if not (isinstance(parameter_index, tuple) and len(parameter_index)==2 and 0<=parameter_index[0]<self.n and 0<=parameter_index[1]<self.n and parameter_index[0]!=parameter_index[1]): logging.error("Ungültiger Index für 'synergy_matrix'. Muss Tupel (i, j) sein mit i!=j."); return
            i, j = parameter_index; i, j = min(i,j), max(i,j); parameter_index = (i,j); # Ensure consistent order
            param_label = f'Synergie "{self.investment_labels[i]}" & "{self.investment_labels[j]}"'
            xlabel = param_label
        elif parameter_name == 'c':
            if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 'c'."); return
            param_label = f'Parameter c für "{self.investment_labels[parameter_index]}"'
            xlabel = param_label
            if self.utility_function_type != 'quadratic': logging.warning(f"'c' nur relevant für 'quadratic'. Überspringe '{param_label}' Sensitivität."); parameter_relevant = False
        elif parameter_name == 's_curve_L':
            if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 's_curve_L'."); return
            param_label = f'S-Kurve L für "{self.investment_labels[parameter_index]}"' ; xlabel = param_label
            if self.utility_function_type != 's_curve': logging.warning(f"'s_curve_L' nur relevant für 's_curve'. Überspringe."); parameter_relevant = False
        elif parameter_name == 's_curve_k':
             if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 's_curve_k'."); return
             param_label = f'S-Kurve k für "{self.investment_labels[parameter_index]}"' ; xlabel = param_label
             if self.utility_function_type != 's_curve': logging.warning(f"'s_curve_k' nur relevant für 's_curve'. Überspringe."); parameter_relevant = False
        elif parameter_name == 's_curve_x0':
             if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n): logging.error("Ungültiger Index für 's_curve_x0'."); return
             param_label = f'S-Kurve x0 für "{self.investment_labels[parameter_index]}"' ; xlabel = param_label
             if self.utility_function_type != 's_curve': logging.warning(f"'s_curve_x0' nur relevant für 's_curve'. Überspringe."); parameter_relevant = False
        else: logging.error(f"Unbekannter Parametername für Sensitivitätsanalyse: {parameter_name}"); return

        if not parameter_relevant: return # Exit if parameter not relevant for current utility function

        logging.info(f"Starte Parametersensitivität für {param_label}...")

        # --- Iterate over parameter values ---
        last_successful_alloc = original_state['initial']
        for k, value in enumerate(parameter_values):
            logging.debug(f"Teste {param_label} = {value:.4f} (Schritt {k+1}/{len(parameter_values)})")
            # --- Temporarily modify the relevant parameter ---
            temp_state = {'roi': original_state['roi'].copy(), 'syn': original_state['syn'].copy(),
                          'c': original_state['c'].copy() if original_state['c'] is not None else None,
                          'L': original_state['L'].copy() if original_state['L'] is not None else None,
                          'k': original_state['k'].copy() if original_state['k'] is not None else None,
                          'x0': original_state['x0'].copy() if original_state['x0'] is not None else None
                         }
            param_valid = True
            try:
                if parameter_name == 'roi_factors':
                    if self.utility_function_type == 'log' and value <= MIN_INVESTMENT:
                        logging.warning(f"Wert {value:.4f} zu klein für log-Utility bei ROI. Überspringe."); param_valid = False
                    else: temp_state['roi'][parameter_index] = value
                elif parameter_name == 'synergy_matrix':
                    if value < 0: logging.warning(f"Negativer Wert {value:.4f} für Synergie. Überspringe."); param_valid = False
                    else: i, j = parameter_index; temp_state['syn'][i, j] = temp_state['syn'][j, i] = value
                elif parameter_name == 'c':
                    if value < 0: logging.warning(f"Negativer Wert {value:.4f} für 'c'. Überspringe."); param_valid = False
                    elif temp_state['c'] is None: logging.error("Cannot set 'c' value when base 'c' is None."); param_valid=False
                    else: temp_state['c'][parameter_index] = value
                elif parameter_name == 's_curve_L':
                     if value <= 0: logging.warning(f"Wert <= 0 ({value:.4f}) für 's_curve_L'. Überspringe."); param_valid = False
                     elif temp_state['L'] is None: logging.error("Cannot set 's_curve_L' when base is None."); param_valid=False
                     else: temp_state['L'][parameter_index] = value
                elif parameter_name == 's_curve_k':
                     if value <= 0: logging.warning(f"Wert <= 0 ({value:.4f}) für 's_curve_k'. Überspringe."); param_valid = False
                     elif temp_state['k'] is None: logging.error("Cannot set 's_curve_k' when base is None."); param_valid=False
                     else: temp_state['k'][parameter_index] = value
                elif parameter_name == 's_curve_x0':
                     # x0 can be anything theoretically, but let's assume non-negative makes sense
                     # if value < 0: logging.warning(f"Negativer Wert {value:.4f} für 's_curve_x0'. Überspringe."); param_valid = False
                     if temp_state['x0'] is None: logging.error("Cannot set 's_curve_x0' when base is None."); param_valid=False
                     else: temp_state['x0'][parameter_index] = value

                if not param_valid:
                     utilities.append(np.nan); allocations.append([np.nan]*self.n); continue

                # --- Run Optimization with temporary parameters ---
                # Use a temporary optimizer instance to avoid state issues
                temp_optimizer = InvestmentOptimizer(
                    roi_factors=temp_state['roi'], synergy_matrix=temp_state['syn'],
                    total_budget=original_budget, # Use original budget
                    lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds, # Use original bounds
                    # Use last successful allocation as warm start
                    initial_allocation=adjust_initial_guess(last_successful_alloc, self.lower_bounds, self.upper_bounds, original_budget),
                    investment_labels=self.investment_labels,
                    log_level=logging.WARNING, # Reduce noise
                    utility_function_type=original_state['utility_type'], # Keep original type
                    c=temp_state['c'], s_curve_L=temp_state['L'], s_curve_k=temp_state['k'], s_curve_x0=temp_state['x0']
                )
                result = temp_optimizer.optimize(method=method)

                if result and result.success:
                    utilities.append(-result.fun); allocations.append(result.x)
                    last_successful_alloc = result.x # Update warm start for next iteration
                else:
                    logging.warning(f"Optimierung fehlgeschlagen für {param_label} = {value:.4f}. Nachricht: {result.message if result else 'N/A'}");
                    utilities.append(np.nan); allocations.append([np.nan]*self.n)
                    # Reset warm start if failed?
                    # last_successful_alloc = original_state['initial']

            except Exception as e:
                logging.error(f"Fehler bei Parametersensitivität Schritt {param_label}={value:.4f}: {e}", exc_info=True);
                utilities.append(np.nan); allocations.append([np.nan]*self.n)
                # last_successful_alloc = original_state['initial']

        # --- Plotting Results ---
        df_param_sens = pd.DataFrame({'Parameterwert': parameter_values, 'Maximaler Nutzen': utilities});
        valid_indices = df_param_sens['Maximaler Nutzen'].notna()

        if not valid_indices.any(): logging.warning(f"Keine gültigen Ergebnisse für Parametersensitivität von {param_label}. Plot übersprungen."); return
        if valid_indices.sum() < 2: logging.warning(f"Weniger als 2 gültige Punkte für Parametersensitivität von {param_label}. Plot übersprungen."); return

        param_values_clean = df_param_sens['Parameterwert'][valid_indices].values;
        utilities_clean = df_param_sens['Maximaler Nutzen'][valid_indices].values

        logging.info(f"Erstelle Plot für Parametersensitivität von {param_label}...")
        fig = None
        try:
            if interactive and plotly_available:
                df_plot = pd.DataFrame({'Parameterwert': param_values_clean, 'Maximaler Nutzen': utilities_clean})
                fig_plotly = px.line(df_plot, x='Parameterwert', y='Maximaler Nutzen', markers=True, title=f'Sensitivität des Nutzens vs <br>{param_label}'); # Use param_label
                fig_plotly.update_layout(xaxis_title=xlabel); fig_plotly.show()
            else:
                fig, ax = plt.subplots(figsize=(10, 6));
                ax.plot(param_values_clean, utilities_clean, marker='o', linestyle='-');
                ax.set_xlabel(xlabel); ax.set_ylabel('Maximaler Nutzen');
                ax.set_title(f'Sensitivität des Nutzens vs\n{param_label}'); # Use param_label
                ax.grid(True, linestyle='--', alpha=0.6); fig.tight_layout(); plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Parametersensitivität für {param_label}: {e}", exc_info=True)
        finally:
            if fig and not (interactive and plotly_available): plt.close(fig)
            # No need to reset parameters as we used a temporary instance


    def plot_pareto_approximation(self, df_pareto, interactive=False):
        """ Plottet die approximierte Pareto-Front. """
        if df_pareto is None or df_pareto.empty: logging.warning("Keine Daten für Pareto-Plot."); return
        required_cols = ['utility', 'risk', 'sustainability']
        if not all(col in df_pareto.columns for col in required_cols): logging.error(f"Pareto DataFrame fehlen Spalten: {required_cols}. Plot nicht möglich."); return

        # Drop rows with NaN in essential columns for plotting
        df_plot = df_pareto[required_cols + ['allocation']].dropna(subset=required_cols).copy()
        if df_plot.empty: logging.warning("Keine gültigen Datenpunkte (ohne NaN) für Pareto-Plot."); return

        logging.info(f"Erstelle Plot für approximierte Pareto-Front ({len(df_plot)} Punkte)...")
        utility, risk, sustainability = df_plot['utility'].values, df_plot['risk'].values, df_plot['sustainability'].values
        fig = None # For matplotlib figure handling

        try:
            if interactive and plotly_available:
                color_col = 'sustainability'; hover_data_cols = ['utility', 'risk', 'sustainability', 'allocation']
                # Add weights to hover data if available
                if all(col in df_pareto for col in ['alpha', 'beta', 'gamma']):
                    df_plot['weights'] = df_pareto.loc[df_plot.index].apply(lambda r: f"U:{r.alpha:.2f}|R:{r.beta:.2f}|S:{r.gamma:.2f}", axis=1)
                    hover_data_cols.append('weights')
                    # Could potentially use weights for color, but sustainability might be more intuitive
                    # color_col = 'weights'

                fig_plotly = px.scatter_3d(df_plot, x='utility', y='risk', z='sustainability',
                                           color=color_col, # Color by sustainability
                                           hover_data=hover_data_cols,
                                           title='Approximierte Pareto-Front')
                fig_plotly.update_layout(scene = dict(
                                            xaxis_title='Utility (Max)',
                                            yaxis_title='Risk (Min)', # Lower risk is better
                                            zaxis_title='Sustainability (Max)'),
                                         margin=dict(l=0, r=0, b=0, t=40)) # Adjust margins
                fig_plotly.show()
            else: # Matplotlib 2D projections
                fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False); # Increased height slightly
                cmap = plt.cm.viridis

                # --- Utility vs Risk (Color by Sustainability) ---
                try: # Normalize Sustainability for color, handle single value case
                    sus_min, sus_max = sustainability.min(), sustainability.max()
                    norm_s = plt.Normalize(sus_min, sus_max) if sus_min != sus_max else plt.Normalize(vmin=sus_min-0.1, vmax=sus_max+0.1) # Handle single value
                except Exception: norm_s = None; logging.warning("Normalisierung für Sustainability-Farbe fehlgeschlagen.")
                c_values_s = sustainability if norm_s else 'blue' # Use blue if normalization fails
                scatter1 = axs[0].scatter(utility, risk, c=c_values_s, cmap=cmap, norm=norm_s, alpha=0.7, s=50) # Increased size
                axs[0].set_xlabel('Utility (Higher is better)'); axs[0].set_ylabel('Risk (Lower is better)'); axs[0].set_title('Utility vs. Risk'); axs[0].grid(True, linestyle='--', alpha=0.6)
                if norm_s: fig.colorbar(scatter1, ax=axs[0], label='Sustainability (Color)')

                # --- Utility vs Sustainability (Color by Risk) ---
                try: risk_min, risk_max = risk.min(), risk.max(); norm_r = plt.Normalize(risk_min, risk_max) if risk_min != risk_max else plt.Normalize(vmin=risk_min-0.1, vmax=risk_max+0.1)
                except Exception: norm_r = None; logging.warning("Normalisierung für Risk-Farbe fehlgeschlagen.")
                c_values_r = risk if norm_r else 'blue'
                scatter2 = axs[1].scatter(utility, sustainability, c=c_values_r, cmap=cmap, norm=norm_r, alpha=0.7, s=50)
                axs[1].set_xlabel('Utility (Higher is better)'); axs[1].set_ylabel('Sustainability (Higher is better)'); axs[1].set_title('Utility vs. Sustainability'); axs[1].grid(True, linestyle='--', alpha=0.6)
                if norm_r: fig.colorbar(scatter2, ax=axs[1], label='Risk (Color)')

                # --- Risk vs Sustainability (Color by Utility) ---
                try: util_min, util_max = utility.min(), utility.max(); norm_u = plt.Normalize(util_min, util_max) if util_min != util_max else plt.Normalize(vmin=util_min-0.1, vmax=util_max+0.1)
                except Exception: norm_u = None; logging.warning("Normalisierung für Utility-Farbe fehlgeschlagen.")
                c_values_u = utility if norm_u else 'blue'
                scatter3 = axs[2].scatter(risk, sustainability, c=c_values_u, cmap=cmap, norm=norm_u, alpha=0.7, s=50)
                axs[2].set_xlabel('Risk (Lower is better)'); axs[2].set_ylabel('Sustainability (Higher is better)'); axs[2].set_title('Risk vs. Sustainability'); axs[2].grid(True, linestyle='--', alpha=0.6)
                if norm_u: fig.colorbar(scatter3, ax=axs[2], label='Utility (Color)')

                plt.suptitle('Approximierte Pareto-Front (2D Projektionen)', fontsize=16);
                fig.tight_layout(rect=[0, 0.03, 1, 0.95]); # Adjust layout to prevent title overlap
                plt.show()
        except Exception as e: logging.error(f"Fehler beim Plotten der Pareto-Approximation: {e}", exc_info=True)
        finally:
            # Close the figure if it was created and not interactive Plotly
            if fig and not (interactive and plotly_available):
                 plt.close(fig)
            # elif not fig and not (interactive and plotly_available): plt.close('all') # Avoid closing all if only plotly was used


# --- Hilfsfunktionen und Main ---
def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description='Investment Optimizer')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Interaktive Plotly-Plots verwenden')
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging-Level')
    return parser.parse_args()

def optimize_for_startup(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation=None,
                         investment_labels=None, utility_function_type='log', c=None, s_curve_L=None, s_curve_k=None, s_curve_x0=None):
    """Vereinfachte Funktion zur schnellen Optimierung (z.B. für API)."""
    n = len(roi_factors);
    if initial_allocation is None: initial_allocation = np.full(n, total_budget / n) if n > 0 else np.array([])
    if investment_labels is None: investment_labels = [f'Bereich_{i}' for i in range(n)]
    try:
        # Use WARNING level to minimize output for API usage
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                                        investment_labels=investment_labels, log_level=logging.WARNING,
                                        utility_function_type=utility_function_type, c=c,
                                        s_curve_L=s_curve_L, s_curve_k=s_curve_k, s_curve_x0=s_curve_x0)
        # Use a reliable method like SLSQP or trust-constr for single shot
        result = optimizer.optimize(method='SLSQP')
        if result and result.success:
            alloc_dict = {f"{optimizer.investment_labels[i]}": round(val, 4) for i, val in enumerate(result.x)}
            return {'allocation': alloc_dict, 'max_utility': -result.fun, 'success': True, 'message': result.message}
        else:
            message = result.message if result else "Optimierung fehlgeschlagen";
            return {'allocation': None, 'max_utility': None, 'success': False, 'message': message}
    except Exception as e:
        logging.error(f"Fehler in optimize_for_startup: {e}", exc_info=False);
        return {'allocation': None, 'max_utility': None, 'success': False, 'message': str(e)}

def main():
    """Hauptfunktion zur Ausführung des Optimierers."""
    args = parse_arguments()
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    configure_logging(log_level)
    config = {}

    # Lade Config oder verwende Defaults
    try:
        if args.config:
            logging.info(f"Lade Konfiguration aus: {args.config}")
            with open(args.config, 'r') as f: config = json.load(f)
            # --- Parameter Extraktion aus Config ---
            required_keys = ['roi_factors', 'synergy_matrix', 'total_budget', 'lower_bounds', 'upper_bounds'] # initial_allocation optional
            if any(key not in config for key in required_keys):
                missing = [key for key in required_keys if key not in config]
                raise KeyError(f"Konfigurationsdatei fehlt erforderliche Schlüssel: {', '.join(missing)}")

            roi_factors_cfg = config.get('roi_factors')
            if roi_factors_cfg is None: raise ValueError("roi_factors fehlt in config.")
            n_cfg = len(roi_factors_cfg)
            investment_labels = config.get('investment_labels')
            if investment_labels and len(investment_labels) != n_cfg:
                 logging.warning("Länge von investment_labels passt nicht zu roi_factors. Ignoriere Labels.")
                 investment_labels = None
            if investment_labels is None: # Generate default labels if needed
                 investment_labels = [f'Bereich_{i}' for i in range(n_cfg)]

            roi_factors = np.array(roi_factors_cfg);
            synergy_matrix = np.array(config.get('synergy_matrix'))
            total_budget = config.get('total_budget');
            lower_bounds = np.array(config.get('lower_bounds'));
            upper_bounds = np.array(config.get('upper_bounds'))
            # Initial allocation is optional, create default if missing
            initial_allocation_cfg = config.get('initial_allocation')
            if initial_allocation_cfg is None:
                 logging.info("Keine 'initial_allocation' in Config gefunden. Erstelle gleichmäßige Verteilung.")
                 initial_allocation = np.full(n_cfg, total_budget / n_cfg) if n_cfg > 0 else np.array([])
                 # Need to adjust this default guess immediately
                 initial_allocation = adjust_initial_guess(initial_allocation, lower_bounds, upper_bounds, total_budget)
            else:
                 initial_allocation = np.array(initial_allocation_cfg) # Will be adjusted in Optimizer init

            utility_function_type = config.get('utility_function_type', DEFAULT_UTILITY_FUNCTION)
            # Safely get utility parameters
            c = np.array(config['c']) if config.get('c') is not None else None
            s_curve_L = np.array(config.get('s_curve_L')) if config.get('s_curve_L') is not None else None
            s_curve_k = np.array(config.get('s_curve_k')) if config.get('s_curve_k') is not None else None
            s_curve_x0 = np.array(config.get('s_curve_x0')) if config.get('s_curve_x0') is not None else None

        else: # Use Default values if no config file
            logging.info("Keine Konfigurationsdatei angegeben (--config), verwende Standard-Demowerte."); n_areas = 4
            investment_labels=['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']; roi_factors=np.array([1.5, 2.0, 2.5, 1.8])
            synergy_matrix=np.array([[0.,.1,.15,.05],[.1,0.,.2,.1],[.15,.2,0.,.25],[.05,.1,.25,0.]])
            total_budget=100.; lower_bounds=np.array([10.]*n_areas); upper_bounds=np.array([50.]*n_areas)
            # Default initial allocation (will be adjusted in Optimizer init)
            initial_allocation=np.array([total_budget/n_areas]*n_areas);
            utility_function_type=DEFAULT_UTILITY_FUNCTION
            c=None; s_curve_L, s_curve_k, s_curve_x0 = None, None, None
            config = {} # Empty config dict for analysis defaults

    except FileNotFoundError:
        logging.error(f"Konfigurationsdatei nicht gefunden: {args.config}"); sys.exit(1)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error(f"Fehler beim Laden/Verarbeiten der Konfiguration: {e}", exc_info=True); sys.exit(1)
    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim Setup: {e}", exc_info=True); sys.exit(1)


    # Initialisiere Optimizer
    try:
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
                                        investment_labels=investment_labels, log_level=log_level, utility_function_type=utility_function_type,
                                        c=c, s_curve_L=s_curve_L, s_curve_k=s_curve_k, s_curve_x0=s_curve_x0)
        # Plot initial synergy heatmap
        optimizer.plot_synergy_heatmap()
    except Exception as e:
        logging.critical(f"Initialisierung des Optimizers fehlgeschlagen: {e}", exc_info=True); sys.exit(1)

    # --- Lade Analyseparameter aus Config oder Defaults ---
    analysis_params = config.get('analysis_parameters', {})
    sens_num_steps = analysis_params.get('sensitivity_num_steps', DEFAULT_SENSITIVITY_NUM_STEPS)
    sens_min_factor = analysis_params.get('sensitivity_budget_min_factor', DEFAULT_SENSITIVITY_BUDGET_MIN_FACTOR)
    sens_max_factor = analysis_params.get('sensitivity_budget_max_factor', DEFAULT_SENSITIVITY_BUDGET_MAX_FACTOR)
    robust_sims = analysis_params.get('robustness_num_simulations', DEFAULT_ROBUSTNESS_NUM_SIMULATIONS)
    robust_var_perc = analysis_params.get('robustness_variation_percentage', DEFAULT_ROBUSTNESS_VARIATION_PERCENTAGE)
    robust_workers_factor = analysis_params.get('robustness_num_workers_factor', DEFAULT_ROBUSTNESS_NUM_WORKERS_FACTOR)
    # Determine num workers for parallel tasks (robustness, DE)
    cpu_count = os.cpu_count() or 1
    robust_num_workers = min(cpu_count, robust_workers_factor) if analysis_params.get('robustness_parallel', True) else 1

    param_sens_steps = analysis_params.get('parameter_sensitivity_num_steps', DEFAULT_PARAM_SENSITIVITY_NUM_STEPS)
    param_sens_min_factor = analysis_params.get('parameter_sensitivity_min_factor', DEFAULT_PARAM_SENSITIVITY_MIN_FACTOR)
    param_sens_max_factor = analysis_params.get('parameter_sensitivity_max_factor', DEFAULT_PARAM_SENSITIVITY_MAX_FACTOR)
    pareto_samples = analysis_params.get('pareto_num_samples', DEFAULT_PARETO_NUM_SAMPLES)
    # Allow overriding DE parameters via config
    de_options = analysis_params.get('de_options', {})
    de_maxiter = de_options.get('maxiter', 1500)
    de_popsize = de_options.get('popsize', 20)

    # --- Führe Optimierungen durch (mit verschiedenen Methoden) ---
    results_dict = {}
    # Define methods to run - consider adding/removing based on needs/performance
    optimizer_methods_to_run = ['SLSQP', 'trust-constr', 'DE', 'L-BFGS-B'] # Nelder-Mead often struggles with constraints
    if optimizer.n <= 10: # BasinHopping can be slow for high dimensions
        optimizer_methods_to_run.append('BasinHopping')

    for method in optimizer_methods_to_run:
        print("\n" + "=" * 30 + f"\n=== Optimierung mit {method} ===")
        try:
            # Pass specific kwargs for DE if method is DE
            kwargs = {'maxiter': de_maxiter, 'popsize': de_popsize, 'workers': robust_num_workers} if method=='DE' else {}
            # Pass worker info to BasinHopping's internal minimizer if applicable? No, BH itself isn't parallelized easily.
            if method == 'BasinHopping':
                # Maybe adjust niter based on problem size?
                kwargs['niter'] = analysis_params.get('basinhopping_niter', 100)

            # Run optimization
            result = optimizer.optimize(method=method, **kwargs)

            if result and result.success:
                print(f"Status: Erfolg") # Keep message short
                print(f"Optimale Allokation ({method}):")
                alloc_df = pd.DataFrame({'Bereich': optimizer.investment_labels, 'Betrag': result.x})
                print(alloc_df.to_string(index=False, float_format="%.4f"))
                print(f"Maximaler Nutzen: {-result.fun:.6f}")
                results_dict[method] = {'utility': -result.fun, 'alloc': result.x}
            else:
                status = result.message if result else "Kein Ergebnisobjekt";
                print(f"Optimierung mit {method} fehlgeschlagen. Status: {status}")
        except Exception as e:
            logging.error(f"Fehler bei {method}-Optimierungslauf: {e}", exc_info=True)

    # --- Vergleich der Ergebnisse ---
    print("\n" + "=" * 30 + "\n=== Vergleich der Optimierungsergebnisse ===")
    if results_dict:
        best_util = -np.inf; best_method = ""
        print(f"{'Methode':<15} | {'Nutzen':<15}")
        print("-" * 30)
        # Sort results by utility descending
        sorted_methods = sorted(results_dict, key=lambda k: results_dict[k]['utility'], reverse=True)
        for method in sorted_methods:
             result = results_dict[method]
             # Check for finite utility before printing/comparing
             if np.isfinite(result['utility']):
                 print(f"{method:<15} | {result['utility']:.6f}")
                 if result['utility'] > best_util:
                     best_util = result['utility']; best_method = method
             else:
                 print(f"{method:<15} | NaN / Inf")

        if best_method:
            print(f"\nBeste gefundene Lösung mit Methode: {best_method} (Nutzen: {best_util:.6f})")
            print("Allokation:")
            best_alloc_df = pd.DataFrame({'Bereich': optimizer.investment_labels, 'Betrag': results_dict[best_method]['alloc']})
            print(best_alloc_df.to_string(index=False, float_format="%.4f"))
        else:
            print("\nKeine gültigen numerischen Ergebnisse zum Finden der besten Methode.")
    else:
        print("Keine erfolgreichen Optimierungen zum Vergleichen.")

    # --- Sensitivitätsanalyse (Budget) ---
    print("\n" + "=" * 30 + "\n=== Sensitivitätsanalyse (Budget) ===")
    try:
        B_min = optimizer.total_budget * sens_min_factor; B_max = optimizer.total_budget * sens_max_factor
        min_possible_budget = np.sum(optimizer.lower_bounds)
        max_possible_budget = np.sum(optimizer.upper_bounds)

        # Adjust B_min and B_max to be within feasible bounds
        if B_min < min_possible_budget - BUDGET_PRECISION:
             logging.warning(f"Minimales Analyse-Budget ({B_min:.2f}) < Summe Mindestinvest. ({min_possible_budget:.2f}). Anpassen auf {min_possible_budget:.2f}.")
             B_min = min_possible_budget
        if B_max > max_possible_budget + BUDGET_PRECISION:
             logging.warning(f"Maximales Analyse-Budget ({B_max:.2f}) > Summe Höchstinvest. ({max_possible_budget:.2f}). Anpassen auf {max_possible_budget:.2f}.")
             B_max = max_possible_budget

        if B_max < B_min + FLOAT_PRECISION:
             logging.warning("Maximales Budget nicht größer als minimales Budget. Überspringe Budget-Sensitivitätsanalyse.")
        else:
             B_values = np.linspace(B_min, B_max, sens_num_steps);
             # Use a reliable method like SLSQP for the analysis steps
             B_sens, allocs_sens, utils_sens = optimizer.sensitivity_analysis(B_values, method='SLSQP');
             optimizer.plot_sensitivity(B_sens, allocs_sens, utils_sens, method='SLSQP', interactive=args.interactive)
    except Exception as e:
        logging.error(f"Fehler bei Budget-Sensitivitätsanalyse: {e}", exc_info=True)
    finally:
        plt.close('all') # Close any open matplotlib figures

    # --- Top Synergien ---
    print("\n" + "=" * 30 + "\n=== Wichtigste Synergieeffekte ===")
    try:
        top_n_synergies = analysis_params.get('top_n_synergies', DEFAULT_TOP_N_SYNERGIES)
        top_synergies = optimizer.identify_top_synergies_correct(top_n=top_n_synergies)
        if top_synergies:
            print(f"Top {len(top_synergies)} Synergiepaare:")
            for pair, value in top_synergies:
                 print(f"- {optimizer.investment_labels[pair[0]]:<15} & {optimizer.investment_labels[pair[1]]:<15}: {value:.4f}")
        else:
            print("Keine Synergieeffekte vorhanden oder gefunden.")
    except Exception as e:
        logging.error(f"Fehler bei Identifizierung der Top-Synergien: {e}", exc_info=True)

    # --- Robustheitsanalyse ---
    print("\n" + "=" * 30 + "\n=== Robustheitsanalyse ===")
    try:
        # Use DE for robustness as it explores the space well
        df_robust_stats, df_robust = optimizer.robustness_analysis(
            num_simulations=robust_sims, method='DE',
            variation_percentage=robust_var_perc,
            parallel=analysis_params.get('robustness_parallel', True), # Use config value or default True
            num_workers=robust_num_workers
        )
        print("\nDeskriptive Statistik der robusten Allokationen:")
        # Select and format columns for display
        stats_display_cols = ['mean', 'std', '5%', '50%', '95%', 'cv']
        # Ensure columns exist before selection
        stats_display_cols = [col for col in stats_display_cols if col in df_robust_stats.columns]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:.4f}'.format):
            print(df_robust_stats[stats_display_cols].fillna('-'))

        # Plot results
        optimizer.plot_robustness_analysis(df_robust) # Pass the raw results df for plotting
    except Exception as e:
        logging.error(f"Fehler bei Robustheitsanalyse: {e}", exc_info=True)
    finally:
        plt.close('all') # Close any open matplotlib figures

    # --- Pareto-Front Approximation ---
    print("\n" + "=" * 30 + "\n=== Pareto-Front Approximation ===")
    pareto_results_df = pd.DataFrame() # Initialize empty DataFrame
    # Define Risk and Sustainability functions (can be customized)
    # Example: Risk as variance, Sustainability as negative squared deviation from mean (encourages equal distribution)
    def risk_func_main(x): return np.var(x)
    def sustainability_func_main(x): n = len(x); mean = np.mean(x) if n > 0 else 0; return -np.sum((x - mean)**2) if n > 0 else 0

    try:
        # Check if Pareto analysis is enabled in config (optional)
        run_pareto = analysis_params.get('run_pareto_analysis', True)
        if run_pareto:
            pareto_method = analysis_params.get('pareto_method', 'SLSQP') # Method for individual multi-obj runs
            pareto_results_df = optimizer.find_pareto_approximation(
                risk_func=risk_func_main,
                sustainability_func=sustainability_func_main,
                num_weight_samples=pareto_samples,
                method=pareto_method
            )

            if not pareto_results_df.empty:
                print(f"Anzahl gefundener nicht-dominierter Lösungen: {len(pareto_results_df)}")
                # Show some example points (e.g., highest Utility, lowest Risk, highest Sustainability)
                print("\nBeispiel-Lösungen:")
                display_cols = ['utility', 'risk', 'sustainability'] # Columns to display along with allocation

                # Find indices using idxmax/idxmin safely
                idx_max_util = pareto_results_df['utility'].idxmax() if 'utility' in pareto_results_df.columns and pareto_results_df['utility'].notna().any() else None
                idx_min_risk = pareto_results_df['risk'].idxmin() if 'risk' in pareto_results_df.columns and pareto_results_df['risk'].notna().any() else None
                idx_max_sust = pareto_results_df['sustainability'].idxmax() if 'sustainability' in pareto_results_df.columns and pareto_results_df['sustainability'].notna().any() else None

                # Function to format output
                def print_solution(label, index, df):
                     if index is not None:
                         print(f"\n--- {label} ---")
                         solution = df.loc[index]
                         print(f"  Ziele: U={solution['utility']:.4f}, R={solution['risk']:.4f}, S={solution['sustainability']:.4f}")
                         alloc_df = pd.DataFrame({'Bereich': optimizer.investment_labels, 'Betrag': solution['allocation']})
                         print(alloc_df.to_string(index=False, float_format="%.4f"))
                     else:
                         print(f"\n--- {label}: Konnte nicht bestimmt werden (evtl. NaN-Werte). ---")

                print_solution("Max Utility", idx_max_util, pareto_results_df)
                print_solution("Min Risk", idx_min_risk, pareto_results_df)
                print_solution("Max Sustainability", idx_max_sust, pareto_results_df)

                # Plot Pareto front
                optimizer.plot_pareto_approximation(pareto_results_df, interactive=args.interactive)
            else:
                 print("Keine nicht-dominierten Lösungen gefunden oder Pareto-Analyse übersprungen.")
        else:
            print("Pareto-Analyse in Konfiguration deaktiviert.")

    except Exception as e:
        logging.error(f"Fehler bei Pareto-Approximation: {e}", exc_info=True)
    finally:
        plt.close('all') # Close any open matplotlib figures

    # --- Parametersensitivitätsanalysen ---
    print("\n" + "=" * 30 + "\n=== Parametersensitivitätsanalysen ===")
    try:
        run_param_sens = analysis_params.get('run_parameter_sensitivity', True)
        if not run_param_sens:
             print("Parametersensitivitätsanalysen in Konfiguration deaktiviert.")
        else:
            # Helper function to generate parameter range safely
            def get_param_range_main(current_value, min_factor, max_factor, num_steps, is_strictly_positive=False, name="Parameter"):
                current_value = float(current_value) if isinstance(current_value, (int, float, np.number)) else 0.0
                # Define absolute minimum value based on context
                abs_min = MIN_INVESTMENT * 1.01 if is_strictly_positive else -np.inf # Allow negative for things like x0

                p_min_raw = current_value * min_factor
                p_max_raw = current_value * max_factor

                # Ensure absolute minimum is respected
                p_min = max(abs_min, p_min_raw) if is_strictly_positive else p_min_raw
                p_max = p_max_raw # Max doesn't usually have an absolute limit unless specified

                 # Handle edge cases: zero or near-zero current value, ensure range > 0
                if abs(p_max - p_min) < FLOAT_PRECISION:
                     logging.debug(f"Parameter '{name}' hat sehr kleinen oder null Variationsbereich ({p_min:.2e} - {p_max:.2e}) bei Wert {current_value:.2e}. Erweitere künstlich.")
                     # If current value is near zero, create a small range around it or a default range
                     if abs(current_value) < FLOAT_PRECISION:
                          p_min = abs_min if is_strictly_positive else -0.1 # Example range
                          p_max = 0.1
                     else: # If current value is non-zero but factors lead to zero range
                          p_max = p_min + abs(p_min * 0.1) + FLOAT_PRECISION*10 # Add 10% or small value

                 # Ensure min <= max after adjustments
                if p_min > p_max: p_min = p_max - FLOAT_PRECISION*10

                logging.debug(f"Generiere Bereich für '{name}': [{p_min:.4f} - {p_max:.4f}] ({num_steps} Schritte)")
                return np.linspace(p_min, p_max, num_steps)

            param_sens_method = analysis_params.get('parameter_sensitivity_method', 'SLSQP')

            # ROI (wenn relevant)
            if optimizer.n > 0 and optimizer.utility_function_type != 's_curve':
                param_idx = 0 # Analyze first ROI factor
                is_log = optimizer.utility_function_type == 'log'
                current_val = optimizer.roi_factors[param_idx]
                param_values = get_param_range_main(current_val, param_sens_min_factor, param_sens_max_factor, param_sens_steps, is_strictly_positive=is_log, name=f"ROI[{param_idx}]")
                optimizer.plot_parameter_sensitivity(param_values, 'roi_factors', parameter_index=param_idx, method=param_sens_method, interactive=args.interactive)
                plt.close('all') # Close plot before next one

            # Synergie (wenn relevant)
            if optimizer.n > 1:
                 param_idx = (0, 1) # Analyze first synergy pair
                 current_val = optimizer.synergy_matrix[param_idx[0], param_idx[1]]
                 # Synergy must be non-negative
                 param_values = get_param_range_main(current_val, param_sens_min_factor, param_sens_max_factor, param_sens_steps, is_strictly_positive=True, name=f"Synergy{param_idx}")
                 optimizer.plot_parameter_sensitivity(param_values, 'synergy_matrix', parameter_index=param_idx, method=param_sens_method, interactive=args.interactive)
                 plt.close('all') # Close plot

            # Parameter c (wenn relevant)
            if optimizer.n > 0 and optimizer.utility_function_type == 'quadratic':
                 param_idx = 0 # Analyze first c parameter
                 if optimizer.c is not None:
                     current_val = optimizer.c[param_idx]
                     # c must be non-negative
                     param_values = get_param_range_main(current_val, param_sens_min_factor, param_sens_max_factor, param_sens_steps, is_strictly_positive=True, name=f"c[{param_idx}]")
                     optimizer.plot_parameter_sensitivity(param_values, 'c', parameter_index=param_idx, method=param_sens_method, interactive=args.interactive)
                     plt.close('all') # Close plot
                 else: logging.warning("Utility ist quadratic, aber 'c' Parameter ist None. Überspringe c-Sensitivität.")

            # S-Curve Parameter (wenn relevant)
            if optimizer.n > 0 and optimizer.utility_function_type == 's_curve':
                s_curve_params_to_analyze = {'s_curve_L': True, 's_curve_k': True, 's_curve_x0': False} # L, k must be > 0, x0 can be anything
                for param_name, is_pos in s_curve_params_to_analyze.items():
                     param_vector = getattr(optimizer, param_name)
                     if param_vector is not None:
                         param_idx = 0
                         current_val = param_vector[param_idx]
                         param_values = get_param_range_main(current_val, param_sens_min_factor, param_sens_max_factor, param_sens_steps, is_strictly_positive=is_pos, name=f"{param_name}[{param_idx}]")
                         optimizer.plot_parameter_sensitivity(param_values, param_name, parameter_index=param_idx, method=param_sens_method, interactive=args.interactive)
                         plt.close('all') # Close plot
                     else: logging.warning(f"Utility ist s_curve, aber '{param_name}' ist None. Überspringe {param_name}-Sensitivität.")

    except Exception as e:
        logging.error(f"Fehler bei Parametersensitivitätsanalyse: {e}", exc_info=True)
    finally:
        plt.close('all') # Ensure all plots are closed

    logging.info("=" * 30 + "\nAlle Analysen abgeschlossen.")

if __name__ == "__main__":
    main()