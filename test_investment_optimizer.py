import unittest
import numpy as np
from investment_optimizer import InvestmentOptimizer, is_symmetric, validate_inputs, adjust_initial_guess, objective

class TestInvestmentOptimizer(unittest.TestCase):
    def setUp(self):
        # Beispielparameter
        self.investment_labels = ['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']
        self.n = len(self.investment_labels)
        self.a = np.array([1, 2, 3, 4])
        self.b = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0]
        ])
        self.B = 10.0
        self.L = np.array([1] * self.n)
        self.U = np.array([5] * self.n)
        self.x0 = np.array([2, 2, 2, 4])
        self.optimizer = InvestmentOptimizer(self.a, self.b, self.B, self.L, self.U, self.x0, self.investment_labels, log_level=logging.CRITICAL)

    def test_is_symmetric(self):
        self.assertTrue(is_symmetric(self.b))
        b_asym = self.b.copy()
        b_asym[0,1] = 0.5
        self.assertFalse(is_symmetric(b_asym))

    def test_validate_inputs(self):
        # Test mit gültigen Eingaben
        try:
            validate_inputs(self.a, self.b, self.B, self.L, self.U, self.x0)
        except ValueError:
            self.fail("validate_inputs() raised ValueError unexpectedly!")

        # Test mit ungültiger Synergiematrix
        b_invalid = self.b.copy()
        b_invalid[0,1] = 0.5
        with self.assertRaises(ValueError):
            validate_inputs(self.a, b_invalid, self.B, self.L, self.U, self.x0)

    def test_optimize_slsqp(self):
        result = self.optimizer.optimize(method='SLSQP')
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertAlmostEqual(np.sum(result.x), self.B, places=4)

    def test_optimize_de(self):
        result = self.optimizer.optimize(method='DE', workers=1)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertAlmostEqual(np.sum(result.x), self.B, places=4)

    def test_objective(self):
        x = self.x0
        obj_value = objective(x, self.a, self.b)
        self.assertIsInstance(obj_value, float)

    def test_adjust_initial_guess(self):
        x0 = np.array([0, 0, 0, 0])  # Ungültig, da unter den Mindestinvestitionen
        adjusted_x0 = adjust_initial_guess(x0, self.L, self.U, self.B)
        self.assertTrue(np.all(adjusted_x0 >= self.L))
        self.assertLessEqual(np.sum(adjusted_x0), self.B)

if __name__ == '__main__':
    unittest.main()
