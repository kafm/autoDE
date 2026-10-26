import importlib
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from autode.core import BatchUpdateMethod, UpdateMethod

module = importlib.import_module('autode.operators.booster.sqp')


class SqpBoosterTests(unittest.TestCase):
    def run_booster(self, remaining=200, start=(4., 3., -2.)):
        optimum = np.array([.5, -1., 2.])
        trace = []
        def objective(x):
            return float(np.sum((x - optimum) ** 2))
        def evaluate_many(positions):
            positions = np.asarray(positions)
            self.assertEqual(positions.shape, (1, 3))
            trace.extend(positions.copy())
            return np.sum((positions - optimum) ** 2, axis=1)
        parent = SimpleNamespace(
            lb=lambda: [-5.] * 3, ub=lambda: [5.] * 3,
            max_evals=lambda: 1000, num_evals=lambda: 1000 - remaining,
        )
        individual = SimpleNamespace(pos=np.array(start))
        result = module.sqp(.2)(SimpleNamespace(parent=parent, evaluate_many=evaluate_many), individual)
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

    def test_booster_is_skipped_when_search_has_stopped(self):
        for method in (UpdateMethod, BatchUpdateMethod):
            with self.subTest(method=method.__name__):
                booster = Mock()
                population = Mock()
                population.get.return_value = []
                population.ranking.return_value = []
                ctx = SimpleNamespace(
                    population=population, archive=None,
                    random=np.random.default_rng(42),
                    parent=SimpleNamespace(hasNext=lambda: False),
                )
                method(mutation=Mock(), selection=Mock(),
                       convergence_booster=booster)._boost_if_applies(ctx)
                booster.applies.assert_not_called()
                booster.boost.assert_not_called()


if __name__ == '__main__':
    unittest.main()
