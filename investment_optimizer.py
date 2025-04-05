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
    """Konfiguriert das Logging-System."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s', # Zeitstempel hinzugefügt
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def is_symmetric(matrix, tol=1e-8):
    """Überprüft, ob eine Matrix symmetrisch ist."""
    return np.allclose(matrix, matrix.T, atol=tol)

def validate_inputs(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, c=None, epsilon_a=1e-6):
    """Validiert die Eingabeparameter für den Optimizer."""
    n = len(roi_factors)
    synergy_matrix = np.array(synergy_matrix)
    if synergy_matrix.shape != (n, n):
        raise ValueError(f"Synergie-Matrix muss quadratisch ({n}x{n}) sein und zur Länge der ROI-Faktoren ({n}) passen.")
    if not is_symmetric(synergy_matrix):
        raise ValueError("Synergie-Matrix muss symmetrisch sein.")
    if not (len(lower_bounds) == len(upper_bounds) == len(initial_allocation) == n):
        raise ValueError("lower_bounds, upper_bounds und initial_allocation müssen die gleiche Länge haben wie roi_factors.")
    if np.any(np.array(lower_bounds) > np.array(upper_bounds)):
        raise ValueError("Für jeden Bereich muss lower_bound <= upper_bound gelten.")
    # Erlaube Summe der Mindestinvestitionen gleich dem Budget
    if np.sum(lower_bounds) > total_budget + 1e-9: # Kleine Toleranz hinzugefügt
        raise ValueError(f"Die Summe der Mindestinvestitionen ({np.sum(lower_bounds)}) überschreitet das Gesamtbudget ({total_budget}).")
    # Erlaube Summe der Höchstinvestitionen gleich dem Budget
    if np.sum(upper_bounds) < total_budget - 1e-9: # Kleine Toleranz hinzugefügt
        raise ValueError(f"Die Summe der Höchstinvestitionen ({np.sum(upper_bounds)}) ist kleiner als das Gesamtbudget ({total_budget}).")
    if np.any(initial_allocation < np.array(lower_bounds) - 1e-9): # Toleranz
        raise ValueError("Initial allocation unterschreitet mindestens eine Mindestinvestition.")
    if np.any(initial_allocation > np.array(upper_bounds) + 1e-9): # Toleranz
        raise ValueError("Initial allocation überschreitet mindestens eine Höchstinvestition.")
    # Erlaube 0 als Mindestinvestition nicht direkt wegen log, aber MIN_INVESTMENT in get_bounds
    if np.any(np.array(lower_bounds) < 0):
        logging.warning("Einige Mindestinvestitionen sind < 0. Sie werden auf MIN_INVESTMENT ({MIN_INVESTMENT}) angehoben, wo nötig.")
        # Die eigentliche Anpassung erfolgt in get_bounds
    if np.any(np.array(roi_factors) <= epsilon_a) and c is None: # Nur relevant für Log-Zielfunktion
        raise ValueError(f"Alle ROI-Faktoren müssen größer als {epsilon_a} sein, wenn die logarithmische Zielfunktion verwendet wird.")
    if np.any(synergy_matrix < 0):
        raise ValueError("Alle Synergieeffekte müssen größer oder gleich Null sein.")
    if c is not None:
        c = np.array(c)
        if c.shape != (n,):
            raise ValueError("Optionaler Parameter 'c' muss die gleiche Form wie roi_factors haben.")
        if np.any(c < 0):
            raise ValueError("Alle Werte in 'c' müssen größer oder gleich Null sein.")

def get_bounds(lower_bounds, upper_bounds):
    """Erstellt die Bounds-Liste für scipy.optimize unter Berücksichtigung von MIN_INVESTMENT."""
    return [(max(lower_bounds[i], MIN_INVESTMENT), upper_bounds[i]) for i in range(len(lower_bounds))]

def compute_synergy(x, synergy_matrix):
    """Berechnet den Synergie-Term."""
    return 0.5 * np.dot(x, np.dot(synergy_matrix, x))

def adjust_initial_guess(initial_allocation, lower_bounds, upper_bounds, total_budget, tol=1e-6):
    """
    Passt die initiale Schätzung an, um Bounds und Budget-Summe (approximativ) zu erfüllen.
    Clippt zuerst und skaliert dann iterativ, um die Summe zu treffen, während Bounds respektiert werden.
    """
    x0 = np.array(initial_allocation, dtype=float)
    n = len(x0)
    lb = np.array(lower_bounds, dtype=float)
    ub = np.array(upper_bounds, dtype=float)

    # Schritt 1: Sicherstellen, dass die Anfangsschätzung innerhalb der Bounds liegt
    x0 = np.clip(x0, lb, ub)

    # Schritt 2: Iterativ anpassen, um die Budgetsumme zu erreichen
    current_sum = np.sum(x0)
    max_iter = 100
    iter_count = 0

    # Wenn Summe schon stimmt (innerhalb Toleranz), fertig
    if np.isclose(current_sum, total_budget, atol=tol):
        return x0

    # Skalierungslogik (vereinfacht und robuster)
    # Idee: Verteile die Differenz proportional, aber respektiere Bounds
    while not np.isclose(current_sum, total_budget, atol=tol) and iter_count < max_iter:
        diff = total_budget - current_sum
        # Identifiziere Indizes, die noch angepasst werden können
        can_increase = (x0 < ub)
        can_decrease = (x0 > lb)

        # Verteile die Differenz
        if diff > 0: # Budget erhöhen
            active_indices = np.where(can_increase)[0]
            if len(active_indices) == 0: break # Keine Erhöhung mehr möglich
            # Proportionale Verteilung basierend auf aktueller Allokation (oder gleichmäßig)
            weights = x0[active_indices] + 1e-9 # Kleine Konstante um 0 zu vermeiden
            adjustment = diff * weights / np.sum(weights)
            x0[active_indices] += adjustment
        else: # Budget verringern
            active_indices = np.where(can_decrease)[0]
            if len(active_indices) == 0: break # Keine Verringerung mehr möglich
            weights = x0[active_indices] + 1e-9
            adjustment = diff * weights / np.sum(weights) # diff ist negativ
            x0[active_indices] += adjustment

        # Stelle sicher, dass Bounds nach Anpassung eingehalten werden
        x0 = np.clip(x0, lb, ub)
        current_sum = np.sum(x0)
        iter_count += 1

    # Finale Prüfung
    if not np.isclose(current_sum, total_budget, atol=tol*10): # Größere Toleranz am Ende
        # Wenn immer noch nicht nahe genug, setze auf eine einfache gültige Verteilung
        logging.warning(f"Anpassung der Startallokation erreichte nicht exakt das Budget ({current_sum:.4f} vs {total_budget:.4f}). Verwende Fallback.")
        # Fallback: Gleichmäßige Verteilung oder Verteilung nahe der Bounds
        x0 = lb.copy()
        remaining_budget = total_budget - np.sum(x0)
        if remaining_budget < 0: # Sollte durch validate_inputs verhindert werden, aber sicher ist sicher
             logging.error("Summe der Lower Bounds überschreitet bereits das Budget in adjust_initial_guess Fallback.")
             x0 = np.clip(initial_allocation, lb, ub) # Letzter Versuch
             x0 = x0 * (total_budget / np.sum(x0))
             x0 = np.clip(x0, lb, ub)
             return x0

        allocatable_range = ub - lb
        positive_range_indices = np.where(allocatable_range > 1e-9)[0]

        if np.sum(allocatable_range) > 1e-9 and len(positive_range_indices) > 0:
             proportions = allocatable_range[positive_range_indices] / np.sum(allocatable_range[positive_range_indices])
             x0[positive_range_indices] += remaining_budget * proportions
        elif len(positive_range_indices) > 0 : # Wenn Range 0, aber Indizes existieren
             x0[positive_range_indices] += remaining_budget / len(positive_range_indices)

        x0 = np.clip(x0, lb, ub) # Finales Clipping
        current_sum = np.sum(x0)
        if not np.isclose(current_sum, total_budget, atol=tol*100):
             logging.error(f"Anpassung der Startallokation fehlgeschlagen. Budgetsumme {current_sum} weicht stark von {total_budget} ab.")
             # Keine gute Lösung mehr möglich, gebe ursprüngliche (geclippte) Version zurück
             return np.clip(initial_allocation, lb, ub)

    return x0


def validate_optional_param_c(c, reference_shape):
    """Validiert den optionalen Parameter c."""
    c = np.array(c)
    if c.shape != reference_shape:
        raise ValueError("Parameter 'c' muss die gleiche Form wie die ROI-Faktoren haben.")
    if np.any(c < 0):
        raise ValueError("Alle Werte in 'c' müssen größer oder gleich Null sein.")
    return c

class OptimizationResult:
    """Container für Optimierungsergebnisse."""
    def __init__(self, x, fun, success, message, **kwargs):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)

def single_simulation(sim_index, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels, method, variation_percentage, optimizer_class, c=None):
    """Führt eine einzelne Simulation für die Robustheitsanalyse durch."""
    try:
        # Parameter variieren
        roi_sim = roi_factors * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=len(roi_factors))
        if c is None: # Nur für Log-Zielfunktion relevant
             roi_sim = np.maximum(roi_sim, 1e-6) # Sicherstellen, dass ROI > 0 für log

        synergy_sim = synergy_matrix * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=synergy_matrix.shape)
        synergy_sim = np.maximum(synergy_sim, 0) # Sicherstellen, dass Synergien >= 0
        # Symmetrie wiederherstellen
        synergy_sim = (synergy_sim + synergy_sim.T) / 2
        np.fill_diagonal(synergy_sim, 0) # Diagonale auf 0 setzen

        # Optionalen Parameter c variieren, falls vorhanden
        c_sim = None
        if c is not None:
            c_sim = c * np.random.uniform(1 - variation_percentage, 1 + variation_percentage, size=c.shape)
            c_sim = np.maximum(c_sim, 0) # Sicherstellen, dass c >= 0

        # Neuen Optimizer mit variierten Parametern erstellen
        optimizer_sim = optimizer_class(
            roi_sim, synergy_sim, total_budget,
            lower_bounds, upper_bounds, initial_allocation, # Startpunkt bleibt gleich
            investment_labels=investment_labels, log_level=logging.CRITICAL, c=c_sim
        )
        result = optimizer_sim.optimize(method=method)
        if result and result.success:
            return result.x.tolist()
        else:
            # Logge fehlgeschlagene Simulation detaillierter
            # logging.warning(f"Simulation {sim_index} nicht erfolgreich. Methode: {method}, Nachricht: {result.message if result else 'Kein Ergebnis'}")
            return [np.nan] * len(roi_factors)
    except Exception as e:
        logging.error(f"Simulation {sim_index} fehlgeschlagen: {e}", exc_info=False) # exc_info=False für weniger Output
        return [np.nan] * len(roi_factors)


class InvestmentOptimizer:
    """
    Optimiert die Investitionsallokation unter Berücksichtigung von ROI-Faktoren (Effizienz),
    Synergieeffekten, Budgetrestriktionen sowie Ober- und Untergrenzen für Investitionen.

    Unterstützt zwei Zielfunktionsmodelle:
    1. Logarithmisch (Standard): Maximiert sum(roi * log(x)) + Synergie.
       Nimmt abnehmende Grenzerträge an. Erfordert roi_factors > 0.
    2. Quadratisch (optional, wenn 'c' bereitgestellt wird): Maximiert sum(roi * x - 0.5 * c * x^2) + Synergie.
       Modelliert Sättigungseffekte. Erfordert c >= 0.
    """
    def __init__(self, roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels=None, log_level=logging.INFO, c=None):
        """
        Initialisiert den InvestmentOptimizer.

        Args:
            roi_factors (list/np.array): Effizienzfaktoren für jeden Investitionsbereich.
            synergy_matrix (list/np.array): Symmetrische Matrix der Synergieeffekte zwischen Bereichen.
            total_budget (float): Das Gesamtbudget.
            lower_bounds (list/np.array): Mindestinvestitionen für jeden Bereich.
            upper_bounds (list/np.array): Höchstinvestitionen für jeden Bereich.
            initial_allocation (list/np.array): Startpunkt für die Optimierung.
            investment_labels (list, optional): Namen der Investitionsbereiche. Defaults to ['Bereich_0', ...].
            log_level (int, optional): Logging-Level. Defaults to logging.INFO.
            c (list/np.array, optional): Parameter für die quadratische Zielfunktion. Wenn None, wird die logarithmische verwendet. Defaults to None.
        """
        configure_logging(log_level)
        try:
            self.roi_factors = np.array(roi_factors)
            self.synergy_matrix = np.array(synergy_matrix)
            self.total_budget = total_budget
            self.lower_bounds = np.array(lower_bounds)
            self.upper_bounds = np.array(upper_bounds)
            self.n = len(roi_factors)
            self.investment_labels = investment_labels if investment_labels else [f'Bereich_{i}' for i in range(self.n)]
            if len(self.investment_labels) != self.n:
                 raise ValueError("Anzahl der investment_labels muss der Anzahl der roi_factors entsprechen.")

            self.c = np.array(c) if c is not None else None
            # Validierung vor Anpassung der Startallokation (ohne c-Prüfung hier)
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, initial_allocation)
            if self.c is not None: # Separate c-Validierung
                 self.c = validate_optional_param_c(self.c, self.roi_factors.shape)

            # Passe initial_allocation NACH der Validierung an
            self.initial_allocation = adjust_initial_guess(initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)

            # Finale Validierung mit angepasster Startallokation und c
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, self.c)

            logging.info("InvestmentOptimizer erfolgreich initialisiert.")
            logging.debug(f"Parameter: n={self.n}, Budget={self.total_budget}, log-Utility={self.c is None}")

        except Exception as e:
            logging.error(f"Fehler bei der Initialisierung des InvestmentOptimizers: {e}", exc_info=True)
            raise

    def update_parameters(self, roi_factors=None, synergy_matrix=None, total_budget=None, lower_bounds=None, upper_bounds=None, initial_allocation=None, c=None):
        """Aktualisiert die Parameter des Optimizers und validiert sie erneut."""
        try:
            parameters_updated = False
            if roi_factors is not None:
                self.roi_factors = np.array(roi_factors)
                parameters_updated = True
                logging.debug("roi_factors aktualisiert.")
            if synergy_matrix is not None:
                synergy_matrix = np.array(synergy_matrix)
                # Validierungen hier direkt durchführen
                if not is_symmetric(synergy_matrix):
                    raise ValueError("Synergie-Matrix muss symmetrisch sein.")
                if np.any(synergy_matrix < 0):
                    raise ValueError("Alle Synergieeffekte müssen größer oder gleich Null sein.")
                if synergy_matrix.shape != (self.n, self.n):
                     raise ValueError(f"Synergie-Matrix muss die Dimension {self.n}x{self.n} haben.")
                self.synergy_matrix = synergy_matrix
                parameters_updated = True
                logging.debug("synergy_matrix aktualisiert.")
            if total_budget is not None:
                self.total_budget = total_budget
                parameters_updated = True
                logging.debug("total_budget aktualisiert.")
            if lower_bounds is not None:
                self.lower_bounds = np.array(lower_bounds)
                parameters_updated = True
                logging.debug("lower_bounds aktualisiert.")
            if upper_bounds is not None:
                self.upper_bounds = np.array(upper_bounds)
                parameters_updated = True
                logging.debug("upper_bounds aktualisiert.")

            # Parameter c behandeln
            if c is not None:
                 self.c = np.array(c) # Wird unten validiert
                 logging.debug("Parameter c aktualisiert.")
            elif c is False: # Explizit Log-Utility erzwingen
                 self.c = None
                 logging.debug("Parameter c entfernt (logarithmische Zielfunktion erzwungen).")


            # Initial Allocation anpassen, wenn nötig
            if initial_allocation is not None:
                # Validierung der neuen initial_allocation (vor Anpassung)
                if len(initial_allocation) != self.n:
                     raise ValueError("Länge der neuen initial_allocation passt nicht.")
                # Passe die neue Allokation an die (möglicherweise auch neuen) Bounds und Budget an
                self.initial_allocation = adjust_initial_guess(initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
                logging.debug("initial_allocation aktualisiert und angepasst.")
            elif parameters_updated: # Wenn andere Parameter geändert wurden, passe die alte Allokation an
                self.initial_allocation = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
                logging.debug("Bestehende initial_allocation an neue Parameter angepasst.")

            # Finale Validierung aller (ggf. aktualisierten) Parameter
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, self.c)
            logging.info("Parameter erfolgreich aktualisiert und validiert.")

        except Exception as e:
            logging.error(f"Fehler beim Aktualisieren der Parameter: {e}", exc_info=True)
            raise

    def objective_with_penalty(self, x, penalty_coeff=1e8):
        """
        Zielfunktion (zu minimieren, daher negativ) mit Budget-Penalty.
        Verwendet logarithmische oder quadratische Nutzenfunktion basierend auf self.c.
        """
        # Sicherstellen, dass x positiv für log ist (sollte durch Bounds passieren)
        if self.c is None:
            x_clipped = np.maximum(x, MIN_INVESTMENT)
            utility = np.sum(self.roi_factors * np.log(x_clipped))
        else:
            utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)

        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy

        # Penalty für Budgetabweichung
        budget_diff = np.sum(x) - self.total_budget
        penalty_budget = penalty_coeff * (budget_diff ** 2)

        # Penalty für Bounds-Verletzung (optional, da Bounds oft vom Optimierer direkt behandelt werden)
        # penalty_bounds = penalty_coeff * (np.sum(np.maximum(0, self.lower_bounds - x)**2) + np.sum(np.maximum(0, x - self.upper_bounds)**2))

        return -total_utility + penalty_budget #+ penalty_bounds

    def objective_without_penalty(self, x):
        """
        Zielfunktion (zu minimieren, daher negativ) ohne explizite Penalties.
        Verwendet logarithmische oder quadratische Nutzenfunktion basierend auf self.c.
        Verlässt sich auf Constraints/Bounds im Optimierer.
        """
       # Sicherstellen, dass x positiv für log ist (sollte durch Bounds passieren)
        if self.c is None:
            x_clipped = np.maximum(x, MIN_INVESTMENT)
            utility = np.sum(self.roi_factors * np.log(x_clipped))
        else:
            utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)

        synergy = compute_synergy(x, self.synergy_matrix)
        total_utility = utility + synergy

        return -total_utility

    def identify_top_synergies_correct(self, top_n=6):
        """Identifiziert die Top N Synergieeffekte (Paare von Bereichen)."""
        if top_n <= 0 or self.n < 2:
             return []
        try:
            synergy_copy = self.synergy_matrix.copy()
            # Ignoriere Diagonalelemente explizit
            np.fill_diagonal(synergy_copy, -np.inf) # Setze auf -inf um sie auszuschließen

            # Finde Indizes der oberen Dreiecksmatrix (ohne Diagonale)
            triu_indices = np.triu_indices(self.n, k=1)
            synergy_values = synergy_copy[triu_indices]

            # Behandele den Fall, dass weniger Synergien als top_n vorhanden sind
            num_synergies = len(synergy_values)
            actual_top_n = min(top_n, num_synergies)
            if actual_top_n == 0:
                 return []

            # Finde die Indizes der größten Werte
            # np.argsort gibt Indizes sortiert von klein nach groß zurück
            sorted_indices = np.argsort(synergy_values)
            top_indices_in_flat_array = sorted_indices[-actual_top_n:] # N größten

            top_synergies = []
            for flat_idx in reversed(top_indices_in_flat_array): # Von größtem zum kleinsten
                original_row_idx = triu_indices[0][flat_idx]
                original_col_idx = triu_indices[1][flat_idx]
                synergy_value = synergy_copy[original_row_idx, original_col_idx]
                # Stelle sicher, dass wir keine -inf Werte von der Diagonale haben (sollte nicht passieren)
                if np.isfinite(synergy_value):
                     top_synergies.append(((original_row_idx, original_col_idx), synergy_value))

            # Bereits sortiert durch reversed(top_indices_in_flat_array)
            return top_synergies

        except Exception as e:
            logging.error(f"Fehler beim Identifizieren der Top-Synergien: {e}", exc_info=True)
            return []

    def optimize(self, method='SLSQP', max_retries=3, workers=None, **kwargs):
        """
        Führt die Optimierung mit der gewählten Methode durch.

        Args:
            method (str): Optimierungsmethode ('SLSQP', 'DE', 'BasinHopping', 'TNC').
            max_retries (int): Maximale Anzahl an Wiederholungsversuchen bei Fehlschlag.
            workers (int, optional): Anzahl paralleler Worker für DE (wenn > 1 und Scipy >= 1.4.0).
            **kwargs: Zusätzliche Optionen für die jeweilige scipy-Optimierungsfunktion.

        Returns:
            OptimizationResult: Objekt mit den Ergebnissen oder None bei Fehlschlag.
        """
        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}]

        scipy_version = version.parse(scipy.__version__)
        de_workers_supported = scipy_version >= version.parse("1.4.0")
        de_updating = 'deferred' if workers is not None and workers > 1 and de_workers_supported else 'immediate'
        actual_workers = workers if workers is not None and workers > 0 and de_workers_supported else 1

        # Wähle die Zielfunktion basierend auf der Methode
        # SLSQP und BasinHopping (mit SLSQP als local minimizer) können Constraints gut handhaben
        use_penalty_objective = method in ['DE', 'TNC'] # Diese können Bounds, aber keine Constraints direkt
        objective_func = self.objective_with_penalty if use_penalty_objective else self.objective_without_penalty
        obj_args = (1e8,) if use_penalty_objective else () # Penalty Koeffizient

        optimization_methods = {
            'SLSQP': lambda: minimize(
                objective_func,
                self.initial_allocation,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints if not use_penalty_objective else [], # Constraints nur wenn keine Penalty-Funktion
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}, # ftol erhöht
                args=obj_args
            ),
            'DE': lambda: differential_evolution(
                objective_func, # Immer Penalty-Funktion für DE
                bounds,
                strategy=kwargs.get('strategy', 'best1bin'),
                maxiter=kwargs.get('maxiter', 1000),
                popsize=kwargs.get('popsize', 15),
                tol=kwargs.get('tol', 0.01), # Standard tol ist oft ausreichend
                mutation=kwargs.get('mutation', (0.5, 1)),
                recombination=kwargs.get('recombination', 0.7),
                updating=de_updating,
                workers=actual_workers,
                polish=True, # Wichtig: Versucht, das Ergebnis mit L-BFGS-B zu verbessern
                init='latinhypercube',
                args=(1e8,) # Immer Penalty-Koeffizient für DE
            ),
            'BasinHopping': lambda: basinhopping(
                objective_func, # Zielfunktion ohne Penalty, da SLSQP die Constraints handhabt
                self.initial_allocation,
                niter=kwargs.get('niter', 100),
                stepsize=kwargs.get('stepsize', 0.5),
                minimizer_kwargs={
                    'method': 'SLSQP',
                    'bounds': bounds,
                    'constraints': constraints, # Constraints für den lokalen Minimierer
                    'options': {'maxiter': 200, 'ftol': 1e-9, 'disp': False}, # Weniger Iterationen pro Schritt
                    'args': () # Keine Penalty-Args für lokalen Minimierer
                 },
                T=kwargs.get('T', 1.0),
                disp=False,
                niter_success=kwargs.get('niter_success', 10) # Mehr Iterationen ohne Verbesserung, um zu stoppen
            ),
            'TNC': lambda: minimize(
                objective_func, # Penalty-Funktion, da TNC keine eq Constraints direkt kann
                self.initial_allocation,
                method='TNC',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-8}, # Toleranzen angepasst
                args=obj_args # Penalty-Args
            )
        }

        supported_methods = list(optimization_methods.keys())
        if method not in supported_methods:
            logging.error(f"Nicht unterstützte Optimierungsmethode: {method}. Unterstützte Methoden: {supported_methods}")
            raise ValueError(f"Nicht unterstützte Optimierungsmethode: {method}.")

        logging.info(f"Starte Optimierung mit Methode: {method}")
        if method == 'DE':
             logging.info(f"Differential Evolution verwendet {actual_workers} Worker.")

        for attempt in range(1, max_retries + 1):
            try:
                logging.debug(f"Optimierungsversuch {attempt}/{max_retries} mit Methode {method}...")
                opt_result = optimization_methods[method]()

                # Ergebnisse extrahieren (BasinHopping hat eine andere Struktur)
                if method == 'BasinHopping':
                    x_opt = opt_result.x
                    # fun_opt ist der Wert der Zielfunktion *mit* Penalties, falls verwendet
                    # Berechne den tatsächlichen Nutzen ohne Penalties neu
                    fun_opt_no_penalty = self.objective_without_penalty(x_opt)
                    success = opt_result.lowest_optimization_result.success
                    message = opt_result.message
                else:
                    x_opt = opt_result.x
                    # Berechne den tatsächlichen Nutzen ohne Penalties neu, falls Penalty verwendet wurde
                    fun_opt_no_penalty = self.objective_without_penalty(x_opt) if use_penalty_objective else opt_result.fun
                    success = opt_result.success
                    message = opt_result.message if hasattr(opt_result, 'message') else 'Optimierung abgeschlossen.'

                # Überprüfe Constraints explizit nach der Optimierung
                constraints_satisfied = (
                    np.isclose(np.sum(x_opt), self.total_budget, atol=1e-6) and
                    np.all(x_opt >= np.array(self.lower_bounds) - 1e-7) and # Toleranz erhöht
                    np.all(x_opt <= np.array(self.upper_bounds) + 1e-7)  # Toleranz erhöht
                )

                logging.debug(f"Versuch {attempt}: Erfolg={success}, Constraints erfüllt={constraints_satisfied}, Nachricht='{message}'")

                if success and constraints_satisfied:
                    logging.info(f"Optimierung mit {method} erfolgreich in Versuch {attempt}.")
                    # Gebe immer den Nutzen *ohne* Penalty zurück
                    return OptimizationResult(x=x_opt, fun=fun_opt_no_penalty, success=success, message=message)
                elif success and not constraints_satisfied:
                     logging.warning(f"Optimierung mit {method} erfolgreich, aber Constraints nicht erfüllt (Summe: {np.sum(x_opt):.4f}, Budget: {self.total_budget:.4f}). Retrying...")
                elif not success:
                     logging.warning(f"Optimierung mit {method} nicht erfolgreich (Nachricht: {message}). Retrying...")

            except Exception as e:
                logging.error(f"Optimierungsversuch {attempt} mit Methode {method} fehlgeschlagen: {e}", exc_info=False) # exc_info=False für weniger Output

            if attempt < max_retries:
                 # Optional: Kleinen Jitter zur Startallokation hinzufügen für nächsten Versuch
                 self.initial_allocation = adjust_initial_guess(
                      self.initial_allocation + np.random.normal(0, 0.01, self.n), # Kleiner Jitter
                      self.lower_bounds, self.upper_bounds, self.total_budget
                 )
                 logging.debug("Startpunkt leicht verändert für nächsten Versuch.")

        logging.error(f"Optimierung mit Methode {method} nach {max_retries} Versuchen fehlgeschlagen.")
        return None


    def sensitivity_analysis(self, B_values, method='SLSQP', tol=1e-6, **kwargs):
        """Führt eine Sensitivitätsanalyse für verschiedene Budgets durch."""
        allocations = []
        max_utilities = [] # Nutzen *ohne* Penalty
        original_budget = self.total_budget # Originalbudget speichern
        original_initial_allocation = self.initial_allocation.copy() # Original Startpunkt speichern

        logging.info(f"Starte Sensitivitätsanalyse für Budgets von {min(B_values)} bis {max(B_values)} mit Methode {method}.")

        for B_new in B_values:
            logging.debug(f"Sensitivitätsanalyse für Budget = {B_new:.2f}")
            try:
                # Versuche, eine sinnvolle neue Startallokation zu finden
                # Skaliere alte Lösung oder starte von Lower Bounds
                if len(allocations) > 0 and not np.isnan(allocations[-1]).any():
                     # Skaliere die letzte erfolgreiche Allokation
                     new_x0_guess = allocations[-1] * (B_new / np.sum(allocations[-1]))
                else:
                     # Skaliere den ursprünglichen Startpunkt
                     new_x0_guess = original_initial_allocation * (B_new / original_budget)

                # Passe die geratene Startallokation an die Bounds und das neue Budget an
                adjusted_x0 = adjust_initial_guess(new_x0_guess, self.lower_bounds, self.upper_bounds, B_new)

                # Erstelle temporären Optimizer oder update Parameter (Update ist effizienter)
                self.update_parameters(total_budget=B_new, initial_allocation=adjusted_x0)

                result = self.optimize(method=method, **kwargs)

                if result and result.success:
                    allocations.append(result.x)
                    # Stelle sicher, dass wir den Nutzen ohne Penalty speichern
                    max_utilities.append(-result.fun) # result.fun ist bereits der Nutzen ohne Penalty
                else:
                    logging.warning(f"Sensitivitätsanalyse für Budget {B_new:.2f} fehlgeschlagen oder Ergebnis ungültig.")
                    allocations.append([np.nan] * self.n)
                    max_utilities.append(np.nan)
            except Exception as e:
                logging.error(f"Schwerer Fehler in Sensitivitätsanalyse für Budget {B_new:.2f}: {e}", exc_info=True)
                allocations.append([np.nan] * self.n)
                max_utilities.append(np.nan)
            finally:
                # Parameter auf Originalwerte zurücksetzen für nächste Iteration oder Ende
                self.update_parameters(total_budget=original_budget, initial_allocation=original_initial_allocation)


        logging.info("Sensitivitätsanalyse abgeschlossen.")
        return B_values, allocations, max_utilities


    def robustness_analysis(self, num_simulations=100, method='DE', variation_percentage=0.1, parallel=True, num_workers=None):
        """
        Führt eine Robustheitsanalyse durch Variation der Eingabeparameter durch.

        Args:
            num_simulations (int): Anzahl der Simulationsläufe.
            method (str): Optimierungsmethode für jede Simulation.
            variation_percentage (float): Maximale prozentuale Abweichung der Parameter (z.B. 0.1 für +/- 10%).
            parallel (bool): Ob die Simulationen parallel ausgeführt werden sollen.
            num_workers (int, optional): Anzahl der Worker für die parallele Ausführung. Defaults to min(4, os.cpu_count()).

        Returns:
            tuple: (pd.DataFrame mit deskriptiven Statistiken, pd.DataFrame mit allen Simulationsergebnissen)
        """
        results = []
        if num_workers is None:
            num_workers = min(4, os.cpu_count() or 1) # Standardwert

        logging.info(f"Starte Robustheitsanalyse mit {num_simulations} Simulationen.")
        logging.info(f"Parameter Variation: +/- {variation_percentage*100:.1f}%, Methode: {method}, Parallel: {parallel}, Worker: {num_workers if parallel else 'N/A'}")

        # Wähle Ausführungsmethode
        if parallel and num_simulations > 1:
             # Stelle sicher, dass der Optimizer Thread-sicher ist (scheint der Fall zu sein)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        single_simulation, # Funktion für einen einzelnen Lauf
                        sim, self.roi_factors, self.synergy_matrix, self.total_budget,
                        self.lower_bounds, self.upper_bounds, self.initial_allocation,
                        self.investment_labels, method, variation_percentage, InvestmentOptimizer, self.c # 'c' übergeben
                    ) for sim in range(num_simulations)
                ]
                # Fortschrittsanzeige (optional, erfordert tqdm)
                # from tqdm import tqdm
                # for future in tqdm(concurrent.futures.as_completed(futures), total=num_simulations, desc="Robustness Sims"):
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                        if (i + 1) % max(1, num_simulations // 10) == 0: # Fortschritt ca. alle 10%
                             logging.debug(f"Robustheitsanalyse: {i+1}/{num_simulations} Simulationen abgeschlossen.")
                    except Exception as e:
                        # Fehler sollte schon in single_simulation geloggt werden, hier ggf. nochmal
                        logging.error(f"Fehler beim Abrufen des Ergebnisses einer Robustheitssimulation: {e}", exc_info=False)
                        results.append([np.nan] * self.n) # Füge NaN-Ergebnis hinzu
        else:
            # Sequentielle Ausführung
            for sim in range(num_simulations):
                 if (sim + 1) % max(1, num_simulations // 10) == 0: # Fortschritt ca. alle 10%
                     logging.debug(f"Robustheitsanalyse: {sim+1}/{num_simulations} Simulationen abgeschlossen.")
                 result = single_simulation(
                      sim, self.roi_factors, self.synergy_matrix, self.total_budget,
                      self.lower_bounds, self.upper_bounds, self.initial_allocation,
                      self.investment_labels, method, variation_percentage, InvestmentOptimizer, self.c # 'c' übergeben
                 )
                 results.append(result)

        # Ergebnisse in DataFrame umwandeln
        df_results = pd.DataFrame(results, columns=self.investment_labels)
        num_failed = df_results.isna().any(axis=1).sum()
        num_successful = len(df_results) - num_failed
        logging.info(f"Robustheitsanalyse abgeschlossen. {num_successful}/{num_simulations} Simulationen erfolgreich.")
        if num_failed > 0:
            logging.warning(f"{num_failed} Simulationen lieferten kein gültiges Ergebnis (NaN).")

        # Deskriptive Statistiken berechnen (nur für erfolgreiche Läufe)
        df_clean = df_results.dropna()
        if not df_clean.empty:
             additional_stats = df_clean.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
             # Zusätzliche Metriken
             additional_stats['cv'] = df_clean.std() / df_clean.mean() # Variationskoeffizient
             additional_stats['range'] = df_clean.max() - df_clean.min()
             additional_stats['iqr'] = additional_stats['75%'] - additional_stats['25%']
        else:
             logging.warning("Keine erfolgreichen Simulationen für statistische Auswertung vorhanden.")
             # Leeren DataFrame mit korrekten Spalten zurückgeben
             additional_stats = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'cv', 'range', 'iqr'], index=self.investment_labels)


        return additional_stats, df_results


    def multi_criteria_optimization(self, alpha, beta, gamma, risk_func, sustainability_func, method='SLSQP'):
        """
        Führt eine multikriterielle Optimierung durch.

        Maximiert: alpha * Nutzen - beta * Risiko - gamma * (-Nachhaltigkeit)
        (Minimiert: -alpha * Nutzen + beta * Risiko + gamma * (-Nachhaltigkeit))

        Args:
            alpha (float): Gewicht für den Nutzen (aus ROI + Synergie).
            beta (float): Gewicht für die Risikofunktion.
            gamma (float): Gewicht für die (negative) Nachhaltigkeitsfunktion.
            risk_func (callable): Funktion, die das Risiko aus der Allokation x berechnet.
            sustainability_func (callable): Funktion, die einen Nachhaltigkeitswert berechnet (höher ist besser).
            method (str): Optimierungsmethode (typischerweise SLSQP wegen Constraints).

        Returns:
            OptimizationResult: Objekt mit den Ergebnissen oder None bei Fehlschlag.
        """
        if not (callable(risk_func) and callable(sustainability_func)):
             raise TypeError("risk_func und sustainability_func müssen aufrufbare Funktionen sein.")

        # Definiere die kombinierte Zielfunktion (zu minimieren)
        def multi_objective(x):
            # Nutzen (positiv)
            synergy = compute_synergy(x, self.synergy_matrix)
            if self.c is not None:
                utility = np.sum(self.roi_factors * x - 0.5 * self.c * x**2)
            else:
                x_clipped = np.maximum(x, MIN_INVESTMENT)
                utility = np.sum(self.roi_factors * np.log(x_clipped))
            total_utility = utility + synergy

            # Risiko (soll minimiert werden, daher positiv in der Zielfunktion)
            risk = risk_func(x)
            # Nachhaltigkeit (soll maximiert werden, daher negativ in der Zielfunktion)
            sustainability = sustainability_func(x)

            # Kombinierte Funktion (minimieren)
            combined_value = - (alpha * total_utility) + (beta * risk) - (gamma * sustainability)
            return combined_value

        bounds = get_bounds(self.lower_bounds, self.upper_bounds)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}]

        logging.info(f"Starte multikriterielle Optimierung mit Gewichten: Nutzen={alpha}, Risiko={beta}, Nachhaltigkeit={gamma}")

        try:
            # Verwende einen geeigneten Optimierer (SLSQP ist gut für Constraints)
            result = minimize(
                multi_objective,
                self.initial_allocation,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                logging.info("Multikriterielle Optimierung erfolgreich.")
                # Berechne den reinen Nutzen (ohne Risiko/Nachhaltigkeit) für das Ergebnis
                final_utility = -self.objective_without_penalty(result.x)
                # Füge ggf. die einzelnen Komponenten zum Ergebnis hinzu
                final_risk = risk_func(result.x)
                final_sustainability = sustainability_func(result.x)
                return OptimizationResult(x=result.x,
                                          fun=result.fun, # Wert der kombinierten Zielfunktion
                                          success=result.success,
                                          message=result.message,
                                          utility=final_utility,
                                          risk=final_risk,
                                          sustainability=final_sustainability)
            else:
                logging.error(f"Multikriterielle Optimierung fehlgeschlagen: {result.message}")
                return None
        except Exception as e:
            logging.error(f"Fehler während der multikriteriellen Optimierung: {e}", exc_info=True)
            return None

    # --- Plotting Methoden ---

    def plot_sensitivity(self, B_values, allocations, utilities, method='SLSQP', interactive=False):
        """Plottet die Ergebnisse der Sensitivitätsanalyse."""
        if not B_values or not allocations or not utilities:
             logging.warning("Keine Daten zum Plotten der Sensitivitätsanalyse vorhanden.")
             return

        df_alloc = pd.DataFrame(allocations, columns=self.investment_labels)
        df_util = pd.DataFrame({'Budget': B_values, 'Maximaler Nutzen': utilities})

        # Behandle NaNs (z.B. durch Interpolation oder Auslassen)
        valid_indices = df_util['Maximaler Nutzen'].notna() & (~df_alloc.isna().any(axis=1))
        if not valid_indices.any():
             logging.warning("Keine gültigen Datenpunkte für den Sensitivitätsplot vorhanden.")
             return

        B_values_clean = df_util['Budget'][valid_indices].values
        allocations_clean = df_alloc[valid_indices].values
        utilities_clean = df_util['Maximaler Nutzen'][valid_indices].values

        if len(B_values_clean) < 2:
             logging.warning("Weniger als 2 gültige Datenpunkte für den Sensitivitätsplot - Plot wird übersprungen.")
             return


        logging.info("Erstelle Plot für Sensitivitätsanalyse...")
        try:
            if interactive and plotly_available:
                logging.debug("Verwende interaktiven Plotly-Plot für Sensitivität.")
                df_alloc_clean_plotly = pd.DataFrame(allocations_clean, columns=self.investment_labels)
                df_alloc_clean_plotly['Budget'] = B_values_clean
                df_util_clean_plotly = pd.DataFrame({'Budget': B_values_clean, 'Maximaler Nutzen': utilities_clean})

                fig1 = px.line(df_alloc_clean_plotly, x='Budget', y=self.investment_labels,
                               title=f'Optimale Investitionsallokationen vs. Budget ({method})',
                               labels={'value': 'Investitionsbetrag', 'variable': 'Bereich'})
                fig1.show()

                fig2 = px.line(df_util_clean_plotly, x='Budget', y='Maximaler Nutzen', markers=True,
                               title=f'Maximaler Nutzen vs. Budget ({method})')
                fig2.show()
            else:
                logging.debug("Verwende statischen Matplotlib-Plot für Sensitivität.")
                fig, ax1 = plt.subplots(figsize=(12, 7)) # Größe angepasst
                colors = plt.cm.viridis(np.linspace(0, 1, self.n)) # Geänderte Farbpalette

                # Allokationen plotten
                for i, label in enumerate(self.investment_labels):
                    ax1.plot(B_values_clean, allocations_clean[:, i], label=label, color=colors[i], marker='o', linestyle='-', markersize=4, alpha=0.8) # Marker und Stil
                ax1.set_xlabel('Gesamtbudget')
                ax1.set_ylabel('Investitionsbetrag', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.legend(loc='upper left', title='Investitionsbereiche')
                ax1.grid(True, linestyle='--', alpha=0.6)

                # Nutzen auf zweiter Y-Achse plotten
                ax2 = ax1.twinx()
                ax2.plot(B_values_clean, utilities_clean, label='Maximaler Nutzen', color='tab:red', marker='x', linestyle='--', markersize=6) # Anderer Marker/Stil
                ax2.set_ylabel('Maximaler Nutzen', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                ax2.legend(loc='upper right')

                plt.title(f'Sensitivitätsanalyse: Allokation & Nutzen vs. Budget ({method})', pad=20) # Titel mit Abstand
                fig.tight_layout() # Passt Layout an
                plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Sensitivitätsanalyse: {e}", exc_info=True)


    def plot_robustness_analysis(self, df_results):
        """Plottet die Ergebnisse der Robustheitsanalyse."""
        if df_results is None or df_results.empty:
             logging.warning("Keine Daten zum Plotten der Robustheitsanalyse vorhanden.")
             return

        num_total_sims = len(df_results)
        num_failed = df_results.isna().any(axis=1).sum()
        logging.info(f"Erstelle Plots für Robustheitsanalyse ({num_total_sims - num_failed} gültige Simulationen)...")

        df_clean = df_results.dropna()
        if df_clean.empty:
            logging.warning("Keine gültigen Daten (nach NaN-Entfernung) zum Plotten der Robustheit vorhanden.")
            return

        try:
            # 1. Boxplot aller Bereiche
            logging.debug("Erstelle Boxplot für Robustheit.")
            df_melted = df_clean.reset_index(drop=True).melt(var_name='Bereich', value_name='Investition')
            plt.figure(figsize=(max(8, self.n * 1.2), 6)) # Dynamische Breite
            sns.boxplot(x='Bereich', y='Investition', data=df_melted, palette='viridis')
            plt.xlabel('Investitionsbereich')
            plt.ylabel('Simulierter optimaler Investitionsbetrag')
            plt.title(f'Verteilung der Allokationen (Robustheitsanalyse, n={len(df_clean)})')
            plt.xticks(rotation=45, ha='right') # Bessere Lesbarkeit bei vielen Bereichen
            plt.tight_layout()
            plt.show()

            # 2. Histogramme/KDE pro Bereich
            logging.debug("Erstelle Histogramme für Robustheit.")
            num_cols = min(4, self.n) # Max 4 Spalten für FacetGrid
            g = sns.FacetGrid(df_melted, col="Bereich", col_wrap=num_cols, sharex=False, sharey=False, height=3, aspect=1.2)
            g.map(sns.histplot, "Investition", kde=True, bins=15, color='skyblue', stat='density') # Kleinere Bins, Dichte statt Anzahl
            g.fig.suptitle(f'Histogramme der Allokationen (Robustheitsanalyse, n={len(df_clean)})', y=1.03)
            g.set_titles("{col_name}") # Nur Bereichsnamen als Titel
            plt.tight_layout(rect=[0, 0, 1, 0.97]) # Platz für Haupttitel lassen
            plt.show()

            # 3. Pairplot (nur sinnvoll bei <= 5 Bereichen)
            if self.n <= 5:
                 logging.debug("Erstelle Pairplot für Robustheit.")
                 g_pair = sns.pairplot(df_clean, kind='scatter', diag_kind='kde', plot_kws={'alpha':0.5, 's':20}, palette='viridis') # Scatter mit KDE
                 g_pair.fig.suptitle(f'Paarweise Beziehungen der Allokationen (Robustheitsanalyse, n={len(df_clean)})', y=1.02)
                 plt.tight_layout(rect=[0, 0, 1, 0.97])
                 plt.show()
            elif self.n > 5:
                 logging.info("Pairplot übersprungen, da mehr als 5 Investitionsbereiche.")

        except Exception as e:
            logging.error(f"Fehler beim Plotten der Robustheitsanalyse: {e}", exc_info=True)


    def plot_synergy_heatmap(self):
        """Plottet eine Heatmap der Synergie-Matrix."""
        if self.synergy_matrix is None or self.n < 2:
             logging.warning("Keine Synergie-Matrix zum Plotten vorhanden oder nur ein Bereich.")
             return

        logging.info("Erstelle Heatmap der Synergieeffekte...")
        try:
            plt.figure(figsize=(max(6, self.n * 0.8), max(5, self.n * 0.7))) # Dynamische Größe
            sns.heatmap(self.synergy_matrix, annot=True, fmt=".2f", # Formatierung auf 2 Dezimalstellen
                        xticklabels=self.investment_labels, yticklabels=self.investment_labels,
                        cmap='viridis', linewidths=.5, linecolor='black', cbar_kws={'label': 'Synergiestärke'}) # Linien und Farblegende
            plt.title('Heatmap der Synergieeffekte zwischen Investitionsbereichen')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Synergie-Heatmap: {e}", exc_info=True)


    def plot_parameter_sensitivity(self, parameter_values, parameter_name, parameter_index=None, method='SLSQP', interactive=False):
        """
        Plottet die Sensitivität des maximalen Nutzens gegenüber Änderungen
        eines einzelnen ROI-Faktors oder Synergieeffekts.
        """
        if parameter_values is None or len(parameter_values) == 0:
            logging.warning(f"Keine Werte für Parameter '{parameter_name}' zum Testen angegeben.")
            return

        original_roi = self.roi_factors.copy()
        original_synergy = self.synergy_matrix.copy()
        original_initial = self.initial_allocation.copy() # Startpunkt für jede Iteration zurücksetzen

        utilities = []
        allocations = [] # Optional: Auch Allokationen speichern/plotten

        # Bestimme die Achsenbeschriftung
        if parameter_name == 'roi_factors':
            if not isinstance(parameter_index, int) or not (0 <= parameter_index < self.n):
                logging.error(f"Ungültiger parameter_index '{parameter_index}' für 'roi_factors'. Muss ein Int zwischen 0 und {self.n-1} sein.")
                return
            xlabel = f'ROI Faktor für "{self.investment_labels[parameter_index]}"'
            logging.info(f"Starte Parametersensitivität für {xlabel}...")
        elif parameter_name == 'synergy_matrix':
            if not (isinstance(parameter_index, tuple) and len(parameter_index) == 2 and
                    0 <= parameter_index[0] < self.n and 0 <= parameter_index[1] < self.n and
                    parameter_index[0] != parameter_index[1]):
                logging.error(f"Ungültiger parameter_index '{parameter_index}' für 'synergy_matrix'. Muss ein Tupel (i, j) mit i!=j sein.")
                return
            i, j = parameter_index
            xlabel = f'Synergieeffekt zwischen "{self.investment_labels[i]}" & "{self.investment_labels[j]}"'
            logging.info(f"Starte Parametersensitivität für {xlabel}...")
        else:
            logging.error(f"Unbekannter Parametername für Sensitivitätsanalyse: {parameter_name}")
            return

        for value in parameter_values:
            logging.debug(f"Teste {parameter_name}[{parameter_index}] = {value:.4f}")
            temp_roi = original_roi.copy()
            temp_synergy = original_synergy.copy()
            temp_c = self.c.copy() if self.c is not None else None # Kopiere c, falls vorhanden

            try:
                # Parameter temporär ändern
                if parameter_name == 'roi_factors':
                    if self.c is None and value <= MIN_INVESTMENT: # Nur für log-Zielfunktion kritisch
                         logging.warning(f"ROI-Faktor {value:.4f} ist zu klein für log-Zielfunktion, überspringe.")
                         utilities.append(np.nan)
                         allocations.append([np.nan] * self.n)
                         continue
                    temp_roi[parameter_index] = value
                elif parameter_name == 'synergy_matrix':
                     if value < 0:
                          logging.warning(f"Negativer Synergieeffekt {value:.4f} ist nicht erlaubt, überspringe.")
                          utilities.append(np.nan)
                          allocations.append([np.nan] * self.n)
                          continue
                     i, j = parameter_index
                     temp_synergy[i, j] = value
                     temp_synergy[j, i] = value # Symmetrie wahren

                # Optimizer mit temporären Werten und originalem Startpunkt neu initialisieren/updaten
                # Hier ist Update effizienter, da nur wenige Parameter geändert werden
                self.update_parameters(roi_factors=temp_roi, synergy_matrix=temp_synergy, initial_allocation=original_initial)
                # Stelle sicher, dass c nicht überschrieben wird, falls es nicht geändert werden soll
                self.c = temp_c

                # Optimieren
                result = self.optimize(method=method)

                if result and result.success:
                    utilities.append(-result.fun) # Nutzen ohne Penalty
                    allocations.append(result.x)
                else:
                    logging.warning(f"Optimierung fehlgeschlagen für {parameter_name}[{parameter_index}] = {value:.4f}")
                    utilities.append(np.nan)
                    allocations.append([np.nan] * self.n)

            except Exception as e:
                logging.error(f"Fehler bei Parametersensitivität für {parameter_name}[{parameter_index}] = {value:.4f}: {e}", exc_info=True)
                utilities.append(np.nan)
                allocations.append([np.nan] * self.n)
            finally:
                # Wichtig: Parameter auf Originalwerte zurücksetzen!
                 self.update_parameters(roi_factors=original_roi, synergy_matrix=original_synergy, initial_allocation=original_initial)
                 self.c = temp_c # Setze c zurück


        # Ergebnisse plotten
        df_param_sens = pd.DataFrame({'Parameterwert': parameter_values, 'Maximaler Nutzen': utilities})
        valid_indices = df_param_sens['Maximaler Nutzen'].notna()

        if not valid_indices.any():
            logging.warning(f"Keine gültigen Ergebnisse für Parametersensitivität von {xlabel} zum Plotten.")
            return
        if valid_indices.sum() < 2:
            logging.warning(f"Weniger als 2 gültige Punkte für Parametersensitivität von {xlabel} - Plot wird übersprungen.")
            return

        param_values_clean = df_param_sens['Parameterwert'][valid_indices].values
        utilities_clean = df_param_sens['Maximaler Nutzen'][valid_indices].values

        logging.info(f"Erstelle Plot für Parametersensitivität von {xlabel}...")
        try:
            if interactive and plotly_available:
                logging.debug(f"Verwende interaktiven Plotly-Plot für Parametersensitivität.")
                df_plot = pd.DataFrame({'Parameterwert': param_values_clean, 'Maximaler Nutzen': utilities_clean})
                fig = px.line(df_plot, x='Parameterwert', y='Maximaler Nutzen', markers=True,
                              title=f'Sensitivität des Nutzens gegenüber Änderungen von <br>{xlabel}')
                fig.update_layout(xaxis_title=xlabel)
                fig.show()
            else:
                logging.debug(f"Verwende statischen Matplotlib-Plot für Parametersensitivität.")
                plt.figure(figsize=(10, 6))
                plt.plot(param_values_clean, utilities_clean, marker='o', linestyle='-')
                plt.xlabel(xlabel)
                plt.ylabel('Maximaler Nutzen')
                plt.title(f'Sensitivität des Nutzens gegenüber Änderungen von\n{xlabel}') # Zeilenumbruch für lange Titel
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"Fehler beim Plotten der Parametersensitivität für {xlabel}: {e}", exc_info=True)


# --- Hilfsfunktionen und Main ---

def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description='Investment Optimizer für Startups und Ressourceneffizienz')
    parser.add_argument('--config', type=str, help='Pfad zur Konfigurationsdatei (JSON)')
    parser.add_argument('--interactive', action='store_true', help='Interaktive Plotly-Plots verwenden (falls verfügbar)')
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging-Level setzen')
    return parser.parse_args()

def optimize_for_startup(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation=None, c=None):
    """Vereinfachte Funktion zur schnellen Optimierung (z.B. für API)."""
    n = len(roi_factors)
    if initial_allocation is None:
        # Einfache gleichmäßige Startallokation
        initial_allocation = np.full(n, total_budget / n)

    try:
        optimizer = InvestmentOptimizer(
            roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation,
            log_level=logging.WARNING, # Weniger Output für API
            c=c
        )
        result = optimizer.optimize(method='SLSQP') # Standardmethode

        if result is not None and result.success:
            allocation_dict = {f"Bereich_{i}": round(val, 4) for i, val in enumerate(result.x)}
            return {'allocation': allocation_dict, 'max_utility': -result.fun, 'success': True}
        else:
            message = result.message if result else "Optimierung fehlgeschlagen"
            return {'allocation': None, 'max_utility': None, 'success': False, 'message': message}
    except Exception as e:
        logging.error(f"Fehler in optimize_for_startup: {e}", exc_info=False)
        return {'allocation': None, 'max_utility': None, 'success': False, 'message': str(e)}


def main():
    """Hauptfunktion zur Ausführung des Optimierers."""
    args = parse_arguments()
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    configure_logging(log_level)

    # --- Lade Konfiguration ---
    if args.config:
        logging.info(f"Lade Konfiguration aus: {args.config}")
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)

            # Verwende die neuen, deskriptiven Schlüssel
            required_keys = ['roi_factors', 'synergy_matrix', 'total_budget', 'lower_bounds', 'upper_bounds', 'initial_allocation']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                # Prüfe auf alte Schlüssel für bessere Fehlermeldung
                old_keys_map = {'a': 'roi_factors', 'b': 'synergy_matrix', 'B': 'total_budget', 'L': 'lower_bounds', 'U': 'upper_bounds', 'x0': 'initial_allocation'}
                found_old = [old for old, new in old_keys_map.items() if old in config and new in missing_keys]
                if found_old:
                     logging.error(f"Veraltete Schlüssel in config.json gefunden ({', '.join(found_old)}). Bitte benennen Sie sie um (z.B. 'a' -> 'roi_factors').")
                raise KeyError(f"Schlüssel {missing_keys} fehlen in der Konfigurationsdatei.")

            # Extrahiere Werte mit neuen Schlüsseln
            investment_labels = config.get('investment_labels', [f'Bereich_{i}' for i in range(len(config['roi_factors']))])
            roi_factors = np.array(config['roi_factors'])
            synergy_matrix = np.array(config['synergy_matrix'])
            total_budget = config['total_budget']
            lower_bounds = np.array(config['lower_bounds'])
            upper_bounds = np.array(config['upper_bounds'])
            initial_allocation = np.array(config['initial_allocation'])
            c = np.array(config['c']) if 'c' in config else None # Optionaler Parameter 'c'

        except FileNotFoundError:
            logging.error(f"Konfigurationsdatei nicht gefunden: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"Fehler beim Parsen der JSON-Konfigurationsdatei: {args.config}")
            sys.exit(1)
        except (KeyError, Exception) as e:
            logging.error(f"Fehler beim Verarbeiten der Konfiguration: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Standardwerte verwenden (mit deskriptiven Namen)
        logging.info("Keine Konfigurationsdatei angegeben, verwende Standardwerte.")
        investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        roi_factors = np.array([1.5, 2.0, 2.5, 1.8]) # Beispielwerte angepasst
        synergy_matrix = np.array([
            [0.0, 0.1, 0.15, 0.05],
            [0.1, 0.0, 0.2,  0.1 ],
            [0.15,0.2, 0.0,  0.25],
            [0.05,0.1, 0.25, 0.0 ]
        ])
        total_budget = 100.0 # Beispielbudget erhöht
        n_areas = len(investment_labels)
        lower_bounds = np.array([10.0] * n_areas) # Beispielgrenzen angepasst
        upper_bounds = np.array([50.0] * n_areas) # Beispielgrenzen angepasst
        # Einfache Startallokation
        initial_allocation = np.array([total_budget / n_areas] * n_areas)
        c = np.array([0.01, 0.015, 0.02, 0.01]) # Beispiel für quadratische Terme

    # --- Initialisiere Optimizer ---
    try:
        optimizer = InvestmentOptimizer(roi_factors, synergy_matrix, total_budget, lower_bounds, upper_bounds, initial_allocation, investment_labels=investment_labels, log_level=log_level, c=c)
        optimizer.plot_synergy_heatmap() # Zeige Heatmap direkt nach Initialisierung
    except Exception as e:
        # Fehler wurde bereits im Konstruktor geloggt
        logging.critical("Initialisierung des Optimizers fehlgeschlagen. Programm wird beendet.")
        sys.exit(1)


    # --- Führe Analysen durch (mit individueller Fehlerbehandlung) ---

    # 1. Optimierung mit SLSQP
    result_slsqp = None
    try:
        logging.info("-" * 30)
        result_slsqp = optimizer.optimize(method='SLSQP')
        print("\n--- Optimierung mit SLSQP ---")
        if result_slsqp is not None and result_slsqp.success:
            print(f"Status: {result_slsqp.message}")
            print(f"Optimale Allokation: {np.round(result_slsqp.x, 4)}")
            print(f"Maximaler Nutzen: {-result_slsqp.fun:.4f}") # fun ist bereits negativ maximierter Nutzen
        else:
            status = result_slsqp.message if result_slsqp else "Kein Ergebnis"
            print(f"Optimierung fehlgeschlagen oder kein Ergebnis verfügbar. Status: {status}")
    except Exception as e:
        logging.error(f"Fehler bei der SLSQP-Optimierung im Main-Loop: {e}", exc_info=True)

    # 2. Optimierung mit Differential Evolution
    result_de = None
    try:
        logging.info("-" * 30)
        # Nutze mehr Kerne wenn verfügbar, aber maximal 4 als Standard
        num_workers_de = min(os.cpu_count() or 1, 4)
        result_de = optimizer.optimize(method='DE', workers=num_workers_de, maxiter=1500, popsize=20) # Mehr Iterationen/Population für DE
        print("\n--- Optimierung mit Differential Evolution ---")
        if result_de is not None and result_de.success:
            print(f"Status: {result_de.message}")
            print(f"Optimale Allokation: {np.round(result_de.x, 4)}")
            print(f"Maximaler Nutzen: {-result_de.fun:.4f}")
        else:
            status = result_de.message if result_de else "Kein Ergebnis"
            print(f"Optimierung fehlgeschlagen oder kein Ergebnis verfügbar. Status: {status}")
    except Exception as e:
        logging.error(f"Fehler bei der DE-Optimierung im Main-Loop: {e}", exc_info=True)

    # 3. Sensitivitätsanalyse für verschiedene Budgets
    try:
        logging.info("-" * 30)
        # Definiere Budget-Range relativ zum aktuellen Budget
        B_values = np.linspace(optimizer.total_budget * 0.7, optimizer.total_budget * 1.5, 15) # 15 Schritte
        B_sens, allocations_sens, utilities_sens = optimizer.sensitivity_analysis(B_values, method='SLSQP') # SLSQP ist meist schneller hierfür
        optimizer.plot_sensitivity(B_sens, allocations_sens, utilities_sens, method='SLSQP', interactive=args.interactive)
    except Exception as e:
        logging.error(f"Fehler bei der Sensitivitätsanalyse oder deren Plot im Main-Loop: {e}", exc_info=True)

    # 4. Identifikation der Top-Synergien
    try:
        logging.info("-" * 30)
        print("\n--- Wichtigste Synergieeffekte ---")
        top_synergies = optimizer.identify_top_synergies_correct(top_n=min(6, optimizer.n * (optimizer.n - 1) // 2)) # Max 6 oder alle Paare
        if top_synergies:
            for pair, value in top_synergies:
                label1 = investment_labels[pair[0]]
                label2 = investment_labels[pair[1]]
                print(f"- {label1:<15} & {label2:<15}: {value:.4f}")
        else:
            print("Keine Synergieeffekte gefunden oder nur ein Bereich vorhanden.")
    except Exception as e:
        logging.error(f"Fehler bei der Identifikation der Top-Synergien im Main-Loop: {e}", exc_info=True)

    # 5. Robustheitsanalyse
    try:
        logging.info("-" * 30)
        # Nutze mehr Kerne wenn verfügbar, aber maximal 4 als Standard
        num_workers_robust = min(os.cpu_count() or 1, 4)
        df_robust_stats, df_robust = optimizer.robustness_analysis(
            num_simulations=200, # Mehr Simulationen für bessere Statistik
            method='DE', # DE ist oft robuster bei Variationen
            variation_percentage=0.15, # Etwas höhere Variation testen
            parallel=True,
            num_workers=num_workers_robust
            )
        print("\n--- Robustheitsanalyse (Deskriptive Statistik) ---")
        # Formatierte Ausgabe der Statistik
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:.4f}'.format):
             print(df_robust_stats[['mean', 'std', '5%', '50%', '95%', 'cv']]) # Ausgewählte Statistiken

        optimizer.plot_robustness_analysis(df_robust)
    except Exception as e:
        logging.error(f"Fehler bei der Robustheitsanalyse oder deren Plot im Main-Loop: {e}", exc_info=True)

    # 6. Multikriterielle Optimierung (Beispiel)
    result_mc = None
    try:
        logging.info("-" * 30)
        print("\n--- Multikriterielle Optimierung (Beispiel) ---")
        # Beispiel: Risiko als Varianz, Nachhaltigkeit als (negativer) Abstand zur Gleichverteilung
        def risk_func(x): return np.var(x)
        def sustainability_func(x):
             n = len(x)
             equal_share = np.sum(x) / n
             return -np.sum((x - equal_share)**2) # Minimierung der Abweichung -> Maximierung der Gleichverteilung

        # Beispielgewichte (sollten zusammen nicht unbedingt 1 ergeben)
        alpha, beta, gamma = 0.6, 0.2, 0.2
        result_mc = optimizer.multi_criteria_optimization(alpha=alpha, beta=beta, gamma=gamma, risk_func=risk_func, sustainability_func=sustainability_func, method='SLSQP')

        if result_mc is not None and result_mc.success:
            print(f"Status: {result_mc.message}")
            print(f"Gewichte: Nutzen={alpha}, Risiko={beta}, Nachhaltigkeit={gamma}")
            print(f"Optimale Allokation: {np.round(result_mc.x, 4)}")
            print(f"Kombinierte Zielfunktion (min): {result_mc.fun:.4f}")
            print(f"  -> Einzelkomponenten: Nutzen={result_mc.utility:.4f}, Risiko={result_mc.risk:.4f}, Nachhaltigkeit={result_mc.sustainability:.4f}")
        else:
            status = result_mc.message if result_mc else "Kein Ergebnis"
            print(f"Multikriterielle Optimierung fehlgeschlagen oder kein Ergebnis verfügbar. Status: {status}")
    except Exception as e:
        logging.error(f"Fehler bei der multikriteriellen Optimierung im Main-Loop: {e}", exc_info=True)

    # 7. Parametersensitivitätsanalysen
    try:
        logging.info("-" * 30)
        print("\n--- Parametersensitivitätsanalysen ---")
        # Sensitivität für ROI von Bereich 0
        if optimizer.n > 0:
            roi0_values = np.linspace(max(0.1 if optimizer.c is not None else MIN_INVESTMENT*1.1, optimizer.roi_factors[0]*0.5), optimizer.roi_factors[0]*1.5, 11)
            optimizer.plot_parameter_sensitivity(roi0_values, 'roi_factors', parameter_index=0, method='SLSQP', interactive=args.interactive)

        # Sensitivität für Synergie zwischen Bereich 0 und 1
        if optimizer.n > 1:
             synergy01_values = np.linspace(max(0, optimizer.synergy_matrix[0, 1]*0.5), optimizer.synergy_matrix[0, 1]*1.5 + 1e-9, 11) # +eps für oberen Wert
             optimizer.plot_parameter_sensitivity(synergy01_values, 'synergy_matrix', parameter_index=(0, 1), method='SLSQP', interactive=args.interactive)
    except Exception as e:
        logging.error(f"Fehler bei der Parametersensitivitätsanalyse im Main-Loop: {e}", exc_info=True)

    logging.info("-" * 30)
    logging.info("Alle Analysen abgeschlossen.")

if __name__ == "__main__":
    main()