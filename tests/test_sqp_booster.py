import importlib
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

module = importlib.import_module('autode.operators.booster.sqp')


class SqpBoosterTests(unittest.TestCase):
    def run_booster(self, remaining=200, start=(4., 3., -2.)):
        optimum = np.array([.5, -1., 2.])
        trace = []
        def objective(x):
            return float(np.sum((x - optimum) ** 2))
        def evaluate(x):
            trace.append(np.array(x, copy=True))
            return objective(x)
        parent = SimpleNamespace(
            lb=lambda: [-5.] * 3, ub=lambda: [5.] * 3,
            max_evals=lambda: 1000, num_evals=lambda: 1000 - remaining,
            evaluate=evaluate,
        )
        individual = SimpleNamespace(pos=np.array(start))
        result = module.sqp(.2)(SimpleNamespace(parent=parent), individual)
        return result, np.asarray(trace), objective

    def test_uses_sqp_method_and_existing_tolerance(self):
        with patch.object(module, 'minimize', wraps=module.minimize) as minimize:
            self.run_booster()
        self.assertEqual(minimize.call_args.kwargs['method'], 'SLSQP')
        self.assertEqual(minimize.call_args.kwargs['tol'], 1e-8)

    def test_improves_solution_within_bounds(self):
        (position, fit), trace, objective = self.run_booster()
        self.assertLess(fit, 1e-8)
        self.assertAlmostEqual(objective(position), fit)
        self.assertTrue(np.all(trace >= -5))
        self.assertTrue(np.all(trace <= 5))

    def test_budget_includes_numerical_gradient_evaluations(self):
        for budget in (1, 3, 5):
            (position, fit), trace, objective = self.run_booster(budget)
            self.assertEqual(len(trace), budget)
            self.assertAlmostEqual(objective(position), fit)

    def test_zero_budget_performs_no_evaluations(self):
        (position, fit), trace, _ = self.run_booster(0)
        self.assertEqual(len(trace), 0)
        self.assertIsNone(position)
        self.assertEqual(fit, float('inf'))

    def test_reproducible_trace_without_random_draws(self):
        state = pickle.dumps(np.random.get_state())
        first, first_trace, _ = self.run_booster()
        second, second_trace, _ = self.run_booster()
        np.testing.assert_array_equal(first_trace, second_trace)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(pickle.dumps(np.random.get_state()), state)


if __name__ == '__main__':
    unittest.main()
