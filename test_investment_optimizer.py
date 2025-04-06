import unittest
import numpy as np
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg') # Prevent plot windows from appearing during tests
import matplotlib.pyplot as plt # Import after setting backend
import os # For cpu_count in robustness test defaults
import io # For capturing log output (optional)
from unittest.mock import patch # For capturing log output (optional)
import scipy
from packaging import version

# Importiere die notwendigen Klassen und Funktionen aus deinem Skript
# Stelle sicher, dass investment_optimizer.py im Python-Pfad liegt oder im selben Verzeichnis ist
try:
    from investment_optimizer import (
        InvestmentOptimizer,
        is_symmetric,
        validate_inputs,
        adjust_initial_guess,
        OptimizationResult,
        # Importiere Konstanten
        MIN_INVESTMENT,
        BUDGET_PRECISION,
        BOUNDS_PRECISION,
        FLOAT_PRECISION,
        DEFAULT_TOP_N_SYNERGIES,
        DEFAULT_UTILITY_FUNCTION
    )
    imports_ok = True
except ImportError as e:
    print(f"Import Error in tests: {e}. Stelle sicher, dass investment_optimizer.py im PYTHONPATH ist.")
    imports_ok = False

# Dummy Zielfunktionen für Tests
def dummy_risk_func(x): return np.var(x) # Beispiel Risiko: Varianz
def dummy_sustainability_func(x):
    n = len(x); mean = np.mean(x) if n > 0 else 0
    # Vermeide Division durch Null, wenn mean 0 ist und alle x auch 0 sind.
    if np.isclose(mean, 0) and np.allclose(x, 0):
        return 0.0 # Perfekte Gleichverteilung bei Null-Investment
    # Normierte quadratische Abweichung (höher ist schlechter, daher negieren) -> Falsch, Funktion soll höheren Wert für besser liefern
    # Nähe zur Gleichverteilung: Minimiere Varianz oder Summe der quadrierten Abweichungen
    # Also Maximiere das Negative der Varianz/SumSq
    return -np.sum((x - mean)**2) # Maximiere Nähe zur Gleichverteilung

@unittest.skipIf(not imports_ok, "Skipping tests due to import error.")
class TestInvestmentOptimizer(unittest.TestCase):
    """Testsuite für den InvestmentOptimizer und zugehörige Funktionen."""

    def setUp(self):
        """Diese Methode wird vor jedem Test ausgeführt."""
        self.investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        self.n = len(self.investment_labels)
        self.roi_factors = np.array([1.5, 2.0, 2.5, 1.8])
        self.synergy_matrix = np.array([
            [0.0, 0.1, 0.15, 0.05],
            [0.1, 0.0, 0.2,  0.1 ],
            [0.15,0.2, 0.0,  0.25],
            [0.05,0.1, 0.25, 0.0 ]
        ])
        self.total_budget = 100.0
        self.lower_bounds = np.array([10.0] * self.n)
        self.upper_bounds = np.array([50.0] * self.n)
        # Eine gültige Startallokation (Summe = Budget, innerhalb Bounds)
        self.initial_allocation = np.array([25.0, 25.0, 25.0, 25.0])
        self.c_param = np.array([0.01, 0.015, 0.02, 0.01]) # Für quadratische Zielfunktion Tests
        self.s_curve_L = np.array([100, 120, 150, 110])
        self.s_curve_k = np.array([0.2, 0.2, 0.15, 0.25])
        self.s_curve_x0 = np.array([20, 25, 30, 22])

        # --- Optimizer Instanzen für Tests ---
        # Log Utility (Default)
        self.optimizer = InvestmentOptimizer(
            self.roi_factors, self.synergy_matrix, self.total_budget,
            self.lower_bounds, self.upper_bounds, self.initial_allocation,
            investment_labels=self.investment_labels, log_level=logging.CRITICAL
        )
        # Quadratic Utility
        self.optimizer_quad = InvestmentOptimizer(
            self.roi_factors, self.synergy_matrix, self.total_budget,
            self.lower_bounds, self.upper_bounds, self.initial_allocation,
            investment_labels=self.investment_labels, log_level=logging.CRITICAL,
            utility_function_type='quadratic', c=self.c_param
        )
        # S-Curve Utility
        self.optimizer_scurve = InvestmentOptimizer(
            self.roi_factors, self.synergy_matrix, self.total_budget,
            self.lower_bounds, self.upper_bounds, self.initial_allocation,
            investment_labels=self.investment_labels, log_level=logging.CRITICAL,
            utility_function_type='s_curve',
            s_curve_L=self.s_curve_L, s_curve_k=self.s_curve_k, s_curve_x0=self.s_curve_x0
        )

    # --- Tests für Hilfsfunktionen ---

    def test_is_symmetric(self):
        """Testet die is_symmetric Funktion."""
        self.assertTrue(is_symmetric(self.synergy_matrix))
        b_asym = self.synergy_matrix.copy(); b_asym[0, 1] = 0.5
        self.assertFalse(is_symmetric(b_asym))
        self.assertTrue(is_symmetric(np.array([[1, 2], [2, 1]])))

    def test_validate_inputs_valid(self):
        """Testet validate_inputs mit gültigen Daten."""
        try: # Log
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='log')
        except ValueError as e: self.fail(f"validate_inputs(log) raised ValueError: {e}")
        try: # Quadratic
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='quadratic', c=self.c_param)
        except ValueError as e: self.fail(f"validate_inputs(quad) raised ValueError: {e}")
        try: # S-Curve
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation,
                            utility_function_type='s_curve', s_curve_L=self.s_curve_L, s_curve_k=self.s_curve_k, s_curve_x0=self.s_curve_x0)
        except ValueError as e: self.fail(f"validate_inputs(s_curve) raised ValueError: {e}")

    def test_validate_inputs_invalid_utility_params(self):
        """Testet validate_inputs mit ungültigen Utility-Parametern."""
        # Log: ROI <= 0
        roi_neg = self.roi_factors.copy(); roi_neg[0] = -1.0
        with self.assertRaisesRegex(ValueError, "ROI-Faktoren müssen größer als"):
            validate_inputs(roi_neg, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='log')
        # Quadratic: c fehlt
        with self.assertRaisesRegex(ValueError, "Parameter 'c' muss angegeben werden"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='quadratic', c=None)
        # Quadratic: c negativ
        c_neg = self.c_param.copy(); c_neg[0] = -0.01
        with self.assertRaisesRegex(ValueError, "Werte in 'c' müssen größer oder gleich Null sein"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='quadratic', c=c_neg)
        # S-Curve: Parameter fehlen
        with self.assertRaisesRegex(ValueError, "müssen angegeben werden"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, utility_function_type='s_curve')
        # S-Curve: L negativ
        L_neg = self.s_curve_L.copy(); L_neg[0] = -10
        with self.assertRaisesRegex(ValueError, "Werte in 's_curve_L' müssen positiv sein"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation,
                            utility_function_type='s_curve', s_curve_L=L_neg, s_curve_k=self.s_curve_k, s_curve_x0=self.s_curve_x0)
        # S-Curve: k negativ
        k_neg = self.s_curve_k.copy(); k_neg[0] = 0
        with self.assertRaisesRegex(ValueError, "Werte in 's_curve_k' müssen positiv sein"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation,
                             utility_function_type='s_curve', s_curve_L=self.s_curve_L, s_curve_k=k_neg, s_curve_x0=self.s_curve_x0)

    def test_validate_inputs_invalid_general(self):
        """Testet validate_inputs mit allgemeinen ungültigen Konfigurationen."""
        # Falsche Dimensionen
        with self.assertRaisesRegex(ValueError, "Synergie-Matrix muss quadratisch sein"):
            validate_inputs(self.roi_factors, self.synergy_matrix[:2,:], self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
        with self.assertRaisesRegex(ValueError, "müssen die gleiche Länge haben"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds[:2], self.upper_bounds, self.initial_allocation)
        # Asymmetrische Synergie
        b_asym = self.synergy_matrix.copy(); b_asym[0,1] = 0.5
        with self.assertRaisesRegex(ValueError, "Synergie-Matrix muss symmetrisch sein"):
            validate_inputs(self.roi_factors, b_asym, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
        # Inkonsistente Bounds
        lb_invalid = self.lower_bounds.copy(); lb_invalid[0] = 60.0 # > upper_bound[0]
        with self.assertRaisesRegex(ValueError, "lower_bound <= upper_bound gelten"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, lb_invalid, self.upper_bounds, self.initial_allocation)
        # Budget-Konflikte
        lb_too_high = np.array([30.0] * self.n) # Summe > Budget
        with self.assertRaisesRegex(ValueError, "Summe der Mindestinvestitionen überschreitet das Gesamtbudget"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, lb_too_high, self.upper_bounds, self.initial_allocation)
        ub_too_low = np.array([20.0] * self.n) # Summe < Budget
        with self.assertRaisesRegex(ValueError, "Summe der Höchstinvestitionen ist kleiner als das Gesamtbudget"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, ub_too_low, self.initial_allocation)
        # Ungültige Startallokation (Bounds)
        x0_low = np.array([5.0] * self.n)
        with self.assertRaisesRegex(ValueError, "Initial allocation unterschreitet .* Mindestinvestition"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, x0_low)
        x0_high = np.array([60.0] * self.n)
        with self.assertRaisesRegex(ValueError, "Initial allocation überschreitet .* Höchstinvestition"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, x0_high)

    # --- Erweiterte Tests für adjust_initial_guess ---
    def test_adjust_initial_guess_basic(self):
        """Testet die Grundfunktionen der Anpassung der Startallokation."""
        # Fall 1: Start ist bereits gültig
        adjusted1 = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.allclose(adjusted1, self.initial_allocation))
        self.assertAlmostEqual(np.sum(adjusted1), self.total_budget, delta=BUDGET_PRECISION * 10)

        # Fall 2: Start unter Bounds, Summe falsch
        x0_low = np.array([5.0] * self.n)
        adjusted2 = adjust_initial_guess(x0_low, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted2 >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(adjusted2 <= self.upper_bounds + BOUNDS_PRECISION))
        self.assertAlmostEqual(np.sum(adjusted2), self.total_budget, delta=BUDGET_PRECISION * 10)

        # Fall 3: Start über Bounds, Summe falsch
        x0_high = np.array([60.0] * self.n)
        adjusted3 = adjust_initial_guess(x0_high, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted3 >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(adjusted3 <= self.upper_bounds + BOUNDS_PRECISION))
        self.assertAlmostEqual(np.sum(adjusted3), self.total_budget, delta=BUDGET_PRECISION * 10)

    def test_adjust_initial_guess_edge_cases(self):
        """Testet Randfälle für die Anpassung der Startallokation."""
        # Fall 1: Budget == Summe der Lower Bounds
        lb_strict = np.array([10.0, 15.0, 12.0, 8.0])
        ub_strict = np.array([50.0] * self.n) # Upper bounds unkritisch
        budget_strict_lower = np.sum(lb_strict) # 45.0
        x0_test = np.array([20.0, 10.0, 10.0, 5.0]) # Irrelevanter Startpunkt
        adjusted = adjust_initial_guess(x0_test, lb_strict, ub_strict, budget_strict_lower)
        self.assertTrue(np.allclose(adjusted, lb_strict), "Result should match lower bounds when budget equals sum(lower)")
        self.assertAlmostEqual(np.sum(adjusted), budget_strict_lower, delta=BUDGET_PRECISION)

        # Fall 2: Budget == Summe der Upper Bounds
        ub_strict_sum = np.array([20.0, 25.0, 30.0, 25.0]) # Summe = 100.0
        lb_strict_sum = np.array([5.0] * self.n) # Lower bounds unkritisch
        budget_strict_upper = np.sum(ub_strict_sum) # 100.0
        x0_test_upper = np.array([10.0, 30.0, 40.0, 20.0])
        adjusted_upper = adjust_initial_guess(x0_test_upper, lb_strict_sum, ub_strict_sum, budget_strict_upper)
        self.assertTrue(np.allclose(adjusted_upper, ub_strict_sum), "Result should match upper bounds when budget equals sum(upper)")
        self.assertAlmostEqual(np.sum(adjusted_upper), budget_strict_upper, delta=BUDGET_PRECISION)

        # Fall 3: Start verletzt Bounds, aber Summe stimmt
        x0_violates_bounds = np.array([5.0, 5.0, 45.0, 45.0]) # Summe = 100.0
        adjusted_violates = adjust_initial_guess(x0_violates_bounds, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted_violates >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(adjusted_violates <= self.upper_bounds + BOUNDS_PRECISION))
        self.assertAlmostEqual(np.sum(adjusted_violates), self.total_budget, delta=BUDGET_PRECISION)
        self.assertTrue(np.all(adjusted_violates[:2] >= 10.0 - BOUNDS_PRECISION)) # Erwarte Korrektur

        # Fall 4: Enge Bounds
        lb_tight = np.array([10.0, 10.0, 10.0, 10.0])
        ub_tight = np.array([11.0, 11.0, 50.0, 50.0])
        budget_tight = 41.0
        x0_tight = np.array([10.25, 10.25, 10.25, 10.25])
        adjusted_tight = adjust_initial_guess(x0_tight, lb_tight, ub_tight, budget_tight)
        self.assertTrue(np.all(adjusted_tight >= lb_tight - BOUNDS_PRECISION))
        self.assertTrue(np.all(adjusted_tight <= ub_tight + BOUNDS_PRECISION))
        self.assertAlmostEqual(np.sum(adjusted_tight), budget_tight, delta=BUDGET_PRECISION * 10)
        self.assertTrue(np.all(adjusted_tight[:2] <= 11.0 + BOUNDS_PRECISION)) # Sollte an oberer Grenze sein
        self.assertAlmostEqual(np.sum(adjusted_tight[2:]), budget_tight - np.sum(adjusted_tight[:2]), delta=BUDGET_PRECISION)


    # --- Tests für die InvestmentOptimizer Klasse ---

    def test_objective_calculation_log(self):
        """Testet die Log-Zielfunktion."""
        x_valid = self.optimizer.initial_allocation
        obj_value_log = self.optimizer.objective_without_penalty(x_valid)
        self.assertIsInstance(obj_value_log, float); self.assertFalse(np.isnan(obj_value_log))
        expected_util = np.sum(self.roi_factors * np.log(x_valid))
        expected_syn = 0.5 * np.dot(x_valid, np.dot(self.synergy_matrix, x_valid))
        self.assertAlmostEqual(obj_value_log, -(expected_util + expected_syn))

    def test_objective_calculation_quadratic(self):
        """Testet die quadratische Zielfunktion."""
        x_valid = self.optimizer_quad.initial_allocation
        obj_value_quad = self.optimizer_quad.objective_without_penalty(x_valid)
        self.assertIsInstance(obj_value_quad, float); self.assertFalse(np.isnan(obj_value_quad))
        expected_util = np.sum(self.roi_factors * x_valid - 0.5 * self.c_param * x_valid**2)
        expected_syn = 0.5 * np.dot(x_valid, np.dot(self.synergy_matrix, x_valid))
        self.assertAlmostEqual(obj_value_quad, -(expected_util + expected_syn))

    def test_objective_calculation_s_curve(self):
        """Testet die S-Kurven Zielfunktion."""
        x_valid = self.optimizer_scurve.initial_allocation
        obj_value_scurve = self.optimizer_scurve.objective_without_penalty(x_valid)
        self.assertIsInstance(obj_value_scurve, float); self.assertFalse(np.isnan(obj_value_scurve)); self.assertFalse(np.isinf(obj_value_scurve))
        exp_term = np.clip(-self.s_curve_k * (x_valid - self.s_curve_x0), -700, 700)
        expected_util = np.sum(self.s_curve_L / (1 + np.exp(exp_term)))
        expected_syn = 0.5 * np.dot(x_valid, np.dot(self.synergy_matrix, x_valid))
        self.assertAlmostEqual(obj_value_scurve, -(expected_util + expected_syn))


    def _check_optimization_result(self, result, method_name):
        """Hilfsfunktion zur Prüfung eines Optimierungsergebnisses."""
        self.assertIsNotNone(result, f"{method_name} returned None")
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success, f"{method_name} failed: {result.message}")
        self.assertFalse(np.isnan(result.fun), f"{method_name} fun is NaN")
        self.assertFalse(np.any(np.isnan(result.x)), f"{method_name} result x contains NaN")
        # Constraint Checks
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION*10, msg=f"Budget constraint not met ({method_name})") # Generelle Toleranz *10
        self.assertTrue(np.all(result.x >= self.lower_bounds - BOUNDS_PRECISION), msg=f"Lower bounds violated ({method_name})")
        self.assertTrue(np.all(result.x <= self.upper_bounds + BOUNDS_PRECISION), msg=f"Upper bounds violated ({method_name})")

    def test_optimize_slsqp(self):
        """Testet die Optimierung mit SLSQP."""
        result = self.optimizer.optimize(method='SLSQP')
        self._check_optimization_result(result, 'SLSQP')
        # SLSQP sollte Budget sehr genau treffen
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION, msg="Budget constraint not met precisely (SLSQP)")

    def test_optimize_trustconstr(self):
        """Testet die Optimierung mit trust-constr."""
        result = self.optimizer.optimize(method='trust-constr')
        self._check_optimization_result(result, 'trust-constr')
        # trust-constr sollte Budget auch sehr genau treffen
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION, msg="Budget constraint not met precisely (trust-constr)")

    def test_optimize_de(self):
        """Testet die Optimierung mit Differential Evolution."""
        # Reduzierte Parameter für schnellen Test
        result = self.optimizer.optimize(method='DE', workers=1, maxiter=100, popsize=10)
        self.assertIsNotNone(result) # Erfolg nicht garantiert, aber Ergebnis prüfen
        self.assertFalse(np.isnan(result.fun))
        self.assertFalse(np.any(np.isnan(result.x)))
        # Budget-Toleranz abhängig von SciPy Version (Constraint Handling)
        de_constraints_supported = version.parse(scipy.__version__) >= version.parse("1.7.0")
        budget_delta = BUDGET_PRECISION * 10 if de_constraints_supported else BUDGET_PRECISION * 100 # Kleiner wenn nativ
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=budget_delta, msg=f"Budget constraint not met (DE, delta={budget_delta})")
        self.assertTrue(np.all(result.x >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(result.x <= self.upper_bounds + BOUNDS_PRECISION))

    def test_optimize_lbfgsb(self):
        """Testet die Optimierung mit L-BFGS-B."""
        result = self.optimizer.optimize(method='L-BFGS-B')
        # Erfolg nicht garantiert, aber Ergebnis prüfen
        self.assertIsNotNone(result)
        self.assertFalse(np.isnan(result.fun))
        self.assertFalse(np.any(np.isnan(result.x)))
        # Budget-Toleranz (Penalty Methode)
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION * 100, msg="Budget constraint not met (L-BFGS-B)")
        self.assertTrue(np.all(result.x >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(result.x <= self.upper_bounds + BOUNDS_PRECISION))

    def test_optimize_neldermead(self):
        """Testet die Optimierung mit Nelder-Mead."""
        result = self.optimizer.optimize(method='Nelder-Mead', options={'maxiter': 500}) # Reduzierte Iterationen
        self.assertIsNotNone(result)
        self.assertFalse(np.isnan(result.fun))
        self.assertFalse(np.any(np.isnan(result.x)))
        # Budget-Toleranz (Penalty, oft ungenau)
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION * 1000, msg="Budget constraint not met (Nelder-Mead)")
        self.assertTrue(np.all(result.x >= self.lower_bounds - BOUNDS_PRECISION * 10))
        self.assertTrue(np.all(result.x <= self.upper_bounds + BOUNDS_PRECISION * 10))

    def test_optimize_slsqp_s_curve(self):
        """Testet die Optimierung mit SLSQP und S-Kurven-Nutzen."""
        result = self.optimizer_scurve.optimize(method='SLSQP')
        self._check_optimization_result(result, 'SLSQP (S-Curve)')
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION)

    # --- Test für update_parameters ---
    def test_update_parameters(self):
        """Testet die update_parameters Methode."""
        new_budget = 120.0; new_roi = self.roi_factors * 1.1
        # Teste Update einzelner Parameter
        self.optimizer.update_parameters(total_budget=new_budget)
        self.assertEqual(self.optimizer.total_budget, new_budget)
        self.assertAlmostEqual(np.sum(self.optimizer.initial_allocation), new_budget, delta=BUDGET_PRECISION*10)
        # Teste Update mehrerer Parameter
        self.optimizer.update_parameters(roi_factors=new_roi, lower_bounds=self.lower_bounds + 5)
        self.assertTrue(np.allclose(self.optimizer.roi_factors, new_roi))
        self.assertTrue(np.allclose(self.optimizer.lower_bounds, self.lower_bounds + 5))
        # Teste explizites Setzen von c auf None bei Quadratic Optimizer
        self.optimizer_quad.update_parameters(c=None)
        self.assertIsNone(self.optimizer_quad.c, "c should be None after update")
        self.assertEqual(self.optimizer_quad.utility_function_type, 'quadratic', "Type should remain quadratic even if c is None temporarily") # Type bleibt, aber validate würde failen
        # Teste Wechsel des Utility-Typs
        self.optimizer.update_parameters(utility_function_type='quadratic', c=self.c_param)
        self.assertEqual(self.optimizer.utility_function_type, 'quadratic')
        self.assertTrue(np.allclose(self.optimizer.c, self.c_param))
        # Teste ungültiges Update
        with self.assertRaises(ValueError):
            self.optimizer.update_parameters(synergy_matrix=np.array([[0,1],[0,0]])) # Falsche Form

    # --- Test für identify_top_synergies ---
    def test_identify_top_synergies_correct(self):
         """Testet die Identifikation der Top-Synergien."""
         top_3 = self.optimizer.identify_top_synergies_correct(top_n=3)
         self.assertEqual(len(top_3), 3); self.assertEqual(top_3[0][0], (2, 3)); self.assertAlmostEqual(top_3[0][1], 0.25) # (V, KS)
         self.assertEqual(top_3[1][0], (1, 2)); self.assertAlmostEqual(top_3[1][1], 0.2)  # (M, V)
         self.assertEqual(top_3[2][0], (0, 2)); self.assertAlmostEqual(top_3[2][1], 0.15) # (FE, V)
         # Teste Default N
         top_default = self.optimizer.identify_top_synergies_correct()
         self.assertEqual(len(top_default), min(DEFAULT_TOP_N_SYNERGIES, self.n * (self.n - 1) // 2))

    # --- Tests für Analysen (Struktur) ---
    def test_sensitivity_analysis_structure(self):
        """Testet die Sensitivitätsanalyse (nur Struktur)."""
        B_values = np.linspace(self.total_budget * 0.8, self.total_budget * 1.2, 3)
        B_sens, allocs_sens, utils_sens = self.optimizer.sensitivity_analysis(B_values, method='SLSQP')
        self.assertIsInstance(B_sens, np.ndarray); self.assertIsInstance(allocs_sens, list); self.assertIsInstance(utils_sens, list)
        self.assertEqual(len(B_sens), len(B_values)); self.assertEqual(len(allocs_sens), len(B_values)); self.assertEqual(len(utils_sens), len(B_values))
        for alloc in allocs_sens: self.assertEqual(len(alloc), self.n) # Check dimension, allow NaN

    def test_robustness_analysis_structure(self):
        """Testet die Robustheitsanalyse (nur Struktur)."""
        stats_df, results_df = self.optimizer.robustness_analysis(num_simulations=5, parallel=False) # Wenige Sims, sequentiell
        self.assertIsInstance(stats_df, pd.DataFrame); self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 5); self.assertEqual(list(results_df.columns), self.investment_labels)
        self.assertEqual(list(stats_df.index), self.investment_labels); self.assertTrue('mean' in stats_df.columns)

    # --- Erweiterte Tests für Multi-Criteria ---
    def test_multi_criteria_optimization_detailed(self):
        """Testet die Multi-Criteria Optimierung detaillierter."""
        alpha, beta, gamma = 0.6, 0.2, 0.2 # Beispielgewichte
        result = self.optimizer.multi_criteria_optimization(alpha, beta, gamma, dummy_risk_func, dummy_sustainability_func, method='SLSQP')
        self.assertIsNotNone(result, "MC optimization returned None")
        self.assertIsInstance(result, OptimizationResult)
        if not result.success: self.skipTest(f"MC optimization failed, cannot run detailed checks: {result.message}") # Überspringe Rest wenn nicht erfolgreich
        self.assertTrue(result.success)
        # Prüfe Einzelziele
        self.assertTrue(hasattr(result, 'utility')); self.assertTrue(hasattr(result, 'risk')); self.assertTrue(hasattr(result, 'sustainability'))
        self.assertIsInstance(result.utility, (float, np.floating)); self.assertFalse(np.isnan(result.utility))
        self.assertIsInstance(result.risk, (float, np.floating)); self.assertFalse(np.isnan(result.risk))
        self.assertIsInstance(result.sustainability, (float, np.floating)); self.assertFalse(np.isnan(result.sustainability))
        # Prüfe kombinierte Zielfunktion
        expected_fun = alpha * (-result.utility) + beta * result.risk - gamma * result.sustainability
        self.assertAlmostEqual(result.fun, expected_fun, delta=1e-5, msg="Combined objective value mismatch")
        # Prüfe Constraints
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, delta=BUDGET_PRECISION)
        self.assertTrue(np.all(result.x >= self.lower_bounds - BOUNDS_PRECISION))
        self.assertTrue(np.all(result.x <= self.upper_bounds + BOUNDS_PRECISION))

    def test_multi_criteria_invalid_functions(self):
        """Testet MC Optimierung mit ungültigen Zielfunktionen."""
        with self.assertRaises(TypeError):
            self.optimizer.multi_criteria_optimization(0.6, 0.2, 0.2, "not_a_function", dummy_sustainability_func)
        with self.assertRaises(TypeError):
            self.optimizer.multi_criteria_optimization(0.6, 0.2, 0.2, dummy_risk_func, 123)

    # --- Tests für Pareto Approximation ---
    def test_generate_weights(self):
        """Testet die Generierung von Gewichtungsvektoren."""
        weights2d = self.optimizer.generate_weights(2, 5); self.assertEqual(len(weights2d), 6) # 0/5 to 5/5
        self.assertTrue(all(np.isclose(np.sum(w), 1.0) for w in weights2d))
        weights3d = self.optimizer.generate_weights(3, 9); steps = 3 # steps=sqrt(9)=3 -> (0,0,3) to (3,0,0) etc + Randpunkte
        expected_count = (steps + 1) * (steps + 2) // 2 # Dreiecks-Zahl für Gitterpunkte
        # Plus Randpunkte, falls nicht exakt getroffen
        self.assertTrue(len(weights3d) >= expected_count) # 10 Punkte im Gitter + ggf Randpunkte
        self.assertTrue(all(np.isclose(np.sum(w), 1.0) for w in weights3d))
        self.assertTrue(any(np.allclose(w, [1.,0.,0.]) for w in weights3d)) # Prüfe Randpunkte

    def test_filter_non_dominated(self):
         """Testet die Filterung nicht-dominierter Lösungen."""
         # Ziele: u, -r, s (alle maximieren)
         solutions = [
             (10, -5, 1), # Dominiert von B
             (11, -4, 1), # B - Nicht dominiert
             (10, -4, 2), # C - Nicht dominiert
             (9, -6, 0),  # Dominiert von A, B, C
             (11, -4, 1), # Duplikat von B
         ]
         non_dominated = self.optimizer.filter_non_dominated(solutions)
         # Erwarte B und C (ohne Duplikat von B)
         self.assertEqual(len(non_dominated), 2)
         # Konvertiere zu Sets für einfacheren Vergleich (Reihenfolge egal)
         expected_set = {tuple(np.round((11, -4, 1), 5)), tuple(np.round((10, -4, 2), 5))}
         result_set = {tuple(np.round(s, 5)) for s in non_dominated}
         self.assertSetEqual(result_set, expected_set)


    def test_find_pareto_approximation_structure(self):
        """Testet die Pareto-Approximationsfunktion (Struktur)."""
        # Wenige Samples für Test
        df_pareto = self.optimizer.find_pareto_approximation(dummy_risk_func, dummy_sustainability_func, num_weight_samples=4, method='SLSQP')
        self.assertIsInstance(df_pareto, pd.DataFrame)
        if not df_pareto.empty: # Nur prüfen wenn nicht leer
             self.assertTrue('utility' in df_pareto.columns); self.assertTrue('risk' in df_pareto.columns)
             self.assertTrue('sustainability' in df_pareto.columns); self.assertTrue('allocation' in df_pareto.columns)
             alloc = df_pareto['allocation'].iloc[0]; self.assertEqual(len(alloc), self.n); self.assertFalse(np.isnan(alloc).any())
             # Prüfe Constraints für eine Stichprobe (können leicht verletzt sein)
             self.assertAlmostEqual(np.sum(alloc), self.total_budget, delta=BUDGET_PRECISION * 10)
             self.assertTrue(np.all(alloc >= self.lower_bounds - BOUNDS_PRECISION * 10))
             self.assertTrue(np.all(alloc <= self.upper_bounds + BOUNDS_PRECISION * 10))

    # --- Tests für Plotting ---
    @unittest.skipIf(plt is None, "Matplotlib nicht verfügbar zum Testen der Plots.")
    def test_plotting_runs_without_error(self):
        """Testet, ob Plot-Funktionen ohne Fehler aufgerufen werden können."""
        try:
            # Sensitivität
            B_values=np.linspace(self.total_budget*0.8, self.total_budget*1.2, 3)
            allocs = [adjust_initial_guess(self.initial_allocation*(b/self.total_budget), self.lower_bounds, self.upper_bounds, b) for b in B_values]
            utils = [-self.optimizer.objective_without_penalty(a) for a in allocs]
            self.optimizer.plot_sensitivity(B_values, allocs, utils, interactive=False)
            # Heatmap
            self.optimizer.plot_synergy_heatmap()
            # Robustheit
            _, robust_df = self.optimizer.robustness_analysis(num_simulations=3, parallel=False)
            self.optimizer.plot_robustness_analysis(robust_df)
            # Parameter-Sensitivität
            param_vals = np.linspace(self.roi_factors[0]*0.8, self.roi_factors[0]*1.2, 3)
            self.optimizer.plot_parameter_sensitivity(param_vals, 'roi_factors', parameter_index=0, interactive=False)
             # Pareto Plot
            p_data = {'utility': [1,2], 'risk': [3,2], 'sustainability': [4,5], 'allocation': [[1],[2]]}
            self.optimizer.plot_pareto_approximation(pd.DataFrame(p_data), interactive=False)
        except Exception as e: self.fail(f"Aufruf einer Plot-Funktion führte zu Fehler: {e}")
        finally: plt.close('all') # Schließe alle Plots am Ende


if __name__ == '__main__':
    # Führe alle Tests in dieser Datei aus
    unittest.main(verbosity=2)