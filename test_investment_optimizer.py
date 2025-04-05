import unittest
import numpy as np
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg') # Prevent plot windows from appearing during tests
import matplotlib.pyplot as plt # Import after setting backend

# Importiere die notwendigen Klassen und Funktionen aus deinem Skript
# Stelle sicher, dass investment_optimizer.py im Python-Pfad liegt oder im selben Verzeichnis ist
from investment_optimizer import (
    InvestmentOptimizer,
    is_symmetric,
    validate_inputs,
    adjust_initial_guess,
    OptimizationResult, # Importiere ggf. auch dieses, wenn benötigt
    MIN_INVESTMENT # Importiere ggf. Konstanten
)

class TestInvestmentOptimizer(unittest.TestCase):
    """Testsuite für den InvestmentOptimizer und zugehörige Funktionen."""

    def setUp(self):
        """Diese Methode wird vor jedem Test ausgeführt."""
        # Verwende deskriptive Parameternamen konsistent mit der Klasse
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

        # Erstelle eine Optimizer-Instanz für die Tests (mit Loglevel CRITICAL)
        self.optimizer = InvestmentOptimizer(
            self.roi_factors, self.synergy_matrix, self.total_budget,
            self.lower_bounds, self.upper_bounds, self.initial_allocation,
            investment_labels=self.investment_labels, log_level=logging.CRITICAL
        )
        # Zweite Instanz mit c-Parameter
        self.optimizer_quad = InvestmentOptimizer(
            self.roi_factors, self.synergy_matrix, self.total_budget,
            self.lower_bounds, self.upper_bounds, self.initial_allocation,
            investment_labels=self.investment_labels, log_level=logging.CRITICAL, c=self.c_param
        )

    # --- Tests für Hilfsfunktionen ---

    def test_is_symmetric(self):
        """Testet die is_symmetric Funktion."""
        self.assertTrue(is_symmetric(self.synergy_matrix))
        b_asym = self.synergy_matrix.copy()
        b_asym[0, 1] = 0.5 # Mache Matrix asymmetrisch
        self.assertFalse(is_symmetric(b_asym))
        self.assertTrue(is_symmetric(np.array([[1, 2], [2, 1]])))

    def test_validate_inputs_valid(self):
        """Testet validate_inputs mit gültigen Daten."""
        try:
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, c=self.c_param)
        except ValueError as e:
            self.fail(f"validate_inputs() raised ValueError unexpectedly: {e}")

    def test_validate_inputs_invalid(self):
        """Testet validate_inputs mit verschiedenen ungültigen Konfigurationen."""
        # Falsche Dimensionen
        with self.assertRaisesRegex(ValueError, "Synergie-Matrix muss quadratisch sein"):
            validate_inputs(self.roi_factors, self.synergy_matrix[:2,:], self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
        with self.assertRaisesRegex(ValueError, "müssen die gleiche Länge haben"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds[:2], self.upper_bounds, self.initial_allocation)

        # Asymmetrische Synergie
        b_asym = self.synergy_matrix.copy()
        b_asym[0,1] = 0.5
        with self.assertRaisesRegex(ValueError, "Synergie-Matrix muss symmetrisch sein"):
            validate_inputs(self.roi_factors, b_asym, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)

        # Inkonsistente Bounds
        lb_invalid = self.lower_bounds.copy()
        lb_invalid[0] = 60.0 # > upper_bound[0]
        with self.assertRaisesRegex(ValueError, "lower_bound <= upper_bound gelten"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, lb_invalid, self.upper_bounds, self.initial_allocation)

        # Budget-Konflikte
        lb_too_high = np.array([30.0] * self.n) # Summe > Budget
        with self.assertRaisesRegex(ValueError, "Summe der Mindestinvestitionen überschreitet das Gesamtbudget"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, lb_too_high, self.upper_bounds, self.initial_allocation)
        ub_too_low = np.array([20.0] * self.n) # Summe < Budget
        with self.assertRaisesRegex(ValueError, "Summe der Höchstinvestitionen ist kleiner als das Gesamtbudget"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, ub_too_low, self.initial_allocation)

        # Ungültige Startallokation
        x0_low = np.array([5.0] * self.n) # Unter Lower Bounds
        with self.assertRaisesRegex(ValueError, "Initial allocation unterschreitet mindestens eine Mindestinvestition"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, x0_low)
        x0_high = np.array([60.0] * self.n) # Über Upper Bounds
        with self.assertRaisesRegex(ValueError, "Initial allocation überschreitet mindestens eine Höchstinvestition"):
             validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, x0_high)

        # Negative Werte
        roi_neg = self.roi_factors.copy()
        roi_neg[0] = -1.0
        with self.assertRaisesRegex(ValueError, "ROI-Faktoren müssen größer als"): # Nur relevant für log
            validate_inputs(roi_neg, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
        syn_neg = self.synergy_matrix.copy()
        syn_neg[0,1] = syn_neg[1,0] = -0.1
        with self.assertRaisesRegex(ValueError, "Synergieeffekte müssen größer oder gleich Null sein"):
            validate_inputs(self.roi_factors, syn_neg, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation)
        c_neg = self.c_param.copy()
        c_neg[0] = -0.01
        with self.assertRaisesRegex(ValueError, "Werte in 'c' müssen größer oder gleich Null sein"):
            validate_inputs(self.roi_factors, self.synergy_matrix, self.total_budget, self.lower_bounds, self.upper_bounds, self.initial_allocation, c=c_neg)


    def test_adjust_initial_guess(self):
        """Testet die Anpassung der Startallokation."""
        # Fall 1: Start ist bereits gültig
        adjusted1 = adjust_initial_guess(self.initial_allocation, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.allclose(adjusted1, self.initial_allocation))
        self.assertAlmostEqual(np.sum(adjusted1), self.total_budget, places=5)

        # Fall 2: Start unter Bounds, Summe falsch
        x0_low = np.array([5.0] * self.n)
        adjusted2 = adjust_initial_guess(x0_low, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted2 >= self.lower_bounds - 1e-9))
        self.assertTrue(np.all(adjusted2 <= self.upper_bounds + 1e-9))
        self.assertAlmostEqual(np.sum(adjusted2), self.total_budget, places=5)

        # Fall 3: Start über Bounds, Summe falsch
        x0_high = np.array([60.0] * self.n)
        adjusted3 = adjust_initial_guess(x0_high, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted3 >= self.lower_bounds - 1e-9))
        self.assertTrue(np.all(adjusted3 <= self.upper_bounds + 1e-9))
        self.assertAlmostEqual(np.sum(adjusted3), self.total_budget, places=5)

        # Fall 4: Summe stimmt, aber nicht innerhalb Bounds
        x0_mix = np.array([5.0, 5.0, 45.0, 45.0]) # Summe = 100
        adjusted4 = adjust_initial_guess(x0_mix, self.lower_bounds, self.upper_bounds, self.total_budget)
        self.assertTrue(np.all(adjusted4 >= self.lower_bounds - 1e-9))
        self.assertTrue(np.all(adjusted4 <= self.upper_bounds + 1e-9))
        self.assertAlmostEqual(np.sum(adjusted4), self.total_budget, places=5)
        # Erwarte, dass die unteren Werte auf 10 angehoben wurden
        self.assertTrue(np.all(adjusted4[:2] >= 10.0 - 1e-9))


    # --- Tests für die InvestmentOptimizer Klasse ---

    def test_objective_calculation(self):
        """Testet die Berechnung der Zielfunktionen (log und quadratisch)."""
        # Teste Log-Zielfunktion
        # Verwende die angepasste Startallokation vom Optimizer
        x_valid = self.optimizer.initial_allocation
        obj_value_log = self.optimizer.objective_without_penalty(x_valid)
        self.assertIsInstance(obj_value_log, float)
        # Optional: Gegen einen bekannten Wert testen, falls berechenbar
        # expected_log_util = np.sum(self.roi_factors * np.log(x_valid))
        # expected_synergy = 0.5 * np.dot(x_valid, np.dot(self.synergy_matrix, x_valid))
        # self.assertAlmostEqual(obj_value_log, -(expected_log_util + expected_synergy))

        # Teste Quadratische Zielfunktion
        obj_value_quad = self.optimizer_quad.objective_without_penalty(x_valid)
        self.assertIsInstance(obj_value_quad, float)
        # Optional: Gegen einen bekannten Wert testen
        # expected_quad_util = np.sum(self.roi_factors * x_valid - 0.5 * self.c_param * x_valid**2)
        # self.assertAlmostEqual(obj_value_quad, -(expected_quad_util + expected_synergy))

        # Teste Penalty-Versionen (sollten > ohne Penalty sein, wenn Constraints verletzt)
        obj_value_log_penalty = self.optimizer.objective_with_penalty(x_valid)
        self.assertAlmostEqual(obj_value_log, obj_value_log_penalty, places=5) # Sollte gleich sein, da x_valid Budget erfüllt

        x_invalid_sum = x_valid * 1.1 # Verletze Budget-Constraint
        obj_value_log_penalty_invalid = self.optimizer.objective_with_penalty(x_invalid_sum)
        self.assertGreater(obj_value_log_penalty_invalid, self.optimizer.objective_without_penalty(x_invalid_sum))


    def test_optimize_slsqp(self):
        """Testet die Optimierung mit SLSQP."""
        result = self.optimizer.optimize(method='SLSQP')
        self.assertIsNotNone(result)
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
        self.assertFalse(np.isnan(result.fun))
        self.assertFalse(np.isnan(result.x).any())
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, places=4, msg="Budget constraint not met")
        self.assertTrue(np.all(result.x >= self.lower_bounds - 1e-7), msg="Lower bounds violated")
        self.assertTrue(np.all(result.x <= self.upper_bounds + 1e-7), msg="Upper bounds violated")


    def test_optimize_de(self):
        """Testet die Optimierung mit Differential Evolution."""
        # DE braucht ggf. mehr Iterationen, aber für Test reichen wenige
        result = self.optimizer.optimize(method='DE', workers=1, maxiter=100, popsize=10) # Reduzierte Parameter für schnellen Test
        self.assertIsNotNone(result)
        self.assertIsInstance(result, OptimizationResult)
        # DE liefert nicht immer success=True, aber Ergebnis sollte gültig sein
        # self.assertTrue(result.success) # Kann fehlschlagen, Toleranz ist wichtiger
        self.assertFalse(np.isnan(result.fun))
        self.assertFalse(np.isnan(result.x).any())
        # DE mit Penalty ist nicht immer exakt auf Budget, erlaube größere Toleranz
        self.assertAlmostEqual(np.sum(result.x), self.total_budget, places=2, msg="Budget constraint not met (DE)")
        self.assertTrue(np.all(result.x >= self.lower_bounds - 1e-7), msg="Lower bounds violated (DE)")
        self.assertTrue(np.all(result.x <= self.upper_bounds + 1e-7), msg="Upper bounds violated (DE)")

    def test_update_parameters(self):
        """Testet die update_parameters Methode."""
        new_budget = 120.0
        new_roi = self.roi_factors * 1.1
        # Teste Update einzelner Parameter
        self.optimizer.update_parameters(total_budget=new_budget)
        self.assertEqual(self.optimizer.total_budget, new_budget)
        # Teste, ob Startallokation angepasst wurde (Summe sollte jetzt ca. new_budget sein)
        self.assertAlmostEqual(np.sum(self.optimizer.initial_allocation), new_budget, places=4)

        # Teste Update mehrerer Parameter
        self.optimizer.update_parameters(roi_factors=new_roi, lower_bounds=self.lower_bounds + 5)
        self.assertTrue(np.allclose(self.optimizer.roi_factors, new_roi))
        self.assertTrue(np.allclose(self.optimizer.lower_bounds, self.lower_bounds + 5))
        # Budget sollte vom vorherigen Update noch gelten
        self.assertEqual(self.optimizer.total_budget, new_budget)

        # Teste ungültiges Update
        with self.assertRaises(ValueError):
            self.optimizer.update_parameters(synergy_matrix=np.array([[0,1],[0,0]])) # Falsche Form


    def test_identify_top_synergies_correct(self):
        """Testet die Identifikation der Top-Synergien."""
        top_3 = self.optimizer.identify_top_synergies_correct(top_n=3)
        self.assertEqual(len(top_3), 3)
        # Erwartete Reihenfolge basierend auf self.synergy_matrix: (2,3) > (1,3) > (1,2)
        self.assertEqual(top_3[0][0], (2, 3)) # Vertrieb & Kundenservice
        self.assertAlmostEqual(top_3[0][1], 0.25)
        self.assertEqual(top_3[1][0], (1, 3)) # Marketing & Kundenservice
        self.assertAlmostEqual(top_3[1][1], 0.1) # Korrektur: Dieser Wert ist 0.1 in der Matrix
        self.assertEqual(top_3[2][0], (1, 2)) # Marketing & Vertrieb
        self.assertAlmostEqual(top_3[2][1], 0.2) # Korrektur: Dieser Wert ist 0.2 in der Matrix

        # Sortierung prüfen: (2,3)=0.25 > (1,2)=0.2 > (0,2)=0.15 (Top 3 sollten diese sein)
        # Neu prüfen:
        # (2, 3) = 0.25
        # (1, 2) = 0.2
        # (0, 2) = 0.15
        # (0, 1) = 0.1
        # (1, 3) = 0.1
        # (0, 3) = 0.05
        self.assertEqual(top_3[0][0], (2, 3)) # Correct
        self.assertAlmostEqual(top_3[0][1], 0.25)
        self.assertEqual(top_3[1][0], (1, 2)) # Correct
        self.assertAlmostEqual(top_3[1][1], 0.2)
        self.assertEqual(top_3[2][0], (0, 2)) # Correct
        self.assertAlmostEqual(top_3[2][1], 0.15)

        # Teste mit n > Anzahl Synergien
        top_10 = self.optimizer.identify_top_synergies_correct(top_n=10)
        num_pairs = self.n * (self.n - 1) // 2
        self.assertEqual(len(top_10), min(10, num_pairs))

        # Teste mit n=0
        top_0 = self.optimizer.identify_top_synergies_correct(top_n=0)
        self.assertEqual(len(top_0), 0)


    def test_sensitivity_analysis(self):
        """Testet die Sensitivitätsanalyse (nur Struktur und Typen)."""
        B_values = np.linspace(self.total_budget * 0.8, self.total_budget * 1.2, 3) # Nur 3 Werte für Test
        B_sens, allocations_sens, utilities_sens = self.optimizer.sensitivity_analysis(B_values, method='SLSQP')

        self.assertIsInstance(B_sens, np.ndarray)
        self.assertIsInstance(allocations_sens, list)
        self.assertIsInstance(utilities_sens, list)
        self.assertEqual(len(B_sens), len(B_values))
        self.assertEqual(len(allocations_sens), len(B_values))
        self.assertEqual(len(utilities_sens), len(B_values))

        # Prüfe, ob Allokationen die richtige Dimension haben (oder NaN sind)
        for alloc in allocations_sens:
            if not np.isnan(alloc).all(): # Nur prüfen, wenn nicht komplett NaN
                 self.assertEqual(len(alloc), self.n)
                 self.assertFalse(np.isnan(alloc).any()) # Sollte keine einzelnen NaNs enthalten


    def test_robustness_analysis(self):
        """Testet die Robustheitsanalyse (nur Struktur und Typen)."""
        # Kleine Simulation für Test
        stats_df, results_df = self.optimizer.robustness_analysis(num_simulations=10, method='SLSQP', parallel=False) # Sequentiell für Test

        self.assertIsInstance(stats_df, pd.DataFrame)
        self.assertIsInstance(results_df, pd.DataFrame)

        self.assertEqual(len(results_df), 10) # Anzahl Zeilen = Anzahl Simulationen
        self.assertEqual(list(results_df.columns), self.investment_labels) # Spaltennamen

        # Prüfe Statistik-DataFrame
        self.assertEqual(list(stats_df.index), self.investment_labels) # Indexnamen
        self.assertTrue('mean' in stats_df.columns)
        self.assertTrue('std' in stats_df.columns)
        self.assertTrue('50%' in stats_df.columns) # Median


    @unittest.skipIf(plt is None, "Matplotlib nicht verfügbar zum Testen der Plots.")
    def test_plotting_runs_without_error(self):
        """Testet, ob Plot-Funktionen ohne Fehler aufgerufen werden können."""
        # Diese Tests prüfen nicht den Inhalt der Plots, nur die Ausführung.
        try:
            # Benötigt ggf. Beispieldaten
            B_values = np.array([90.0, 100.0, 110.0])
            # Erzeuge plausible Allokationen und Utilities
            allocs = [adjust_initial_guess(self.initial_allocation * (b / self.total_budget), self.lower_bounds, self.upper_bounds, b) for b in B_values]
            utils = [-self.optimizer.objective_without_penalty(a) for a in allocs]

            self.optimizer.plot_sensitivity(B_values, allocs, utils, interactive=False)
            plt.close() # Schließe Plot-Fenster

            self.optimizer.plot_synergy_heatmap()
            plt.close()

            # Robustheits-Plots benötigen Daten
            _ , robust_df = self.optimizer.robustness_analysis(num_simulations=5, parallel=False)
            self.optimizer.plot_robustness_analysis(robust_df)
            plt.close('all') # Schließe alle offenen Plots

            # Parameter-Sensitivität
            param_vals = np.linspace(self.roi_factors[0]*0.8, self.roi_factors[0]*1.2, 3)
            self.optimizer.plot_parameter_sensitivity(param_vals, 'roi_factors', parameter_index=0, interactive=False)
            plt.close()

        except Exception as e:
            self.fail(f"Aufruf einer Plot-Funktion führte zu einem Fehler: {e}")


if __name__ == '__main__':
    # Führe alle Tests in dieser Datei aus
    unittest.main(verbosity=2)