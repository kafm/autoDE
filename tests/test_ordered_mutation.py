import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from autode.core.population import DefaultPicker, Individual
from autode.operators.mutation import current_to_ord_best, current_to_ord_pbest


def individual(position, fit, index):
    return Individual(np.asarray(position, dtype=float), fit, index, index)


class OrderedMutationTests(unittest.TestCase):
    def test_equations_for_all_donor_orders_and_ties(self):
        rng = np.random.default_rng(42)
        for dimension, fits, scale in itertools.product(
            (1, 30, 100), ((1, 2, 3), (1, 1, 2), (1, 1, 1)), (0., .5, 1.)
        ):
            target = individual(rng.normal(size=dimension), 10, 0)
            donors = [individual(rng.normal(size=dimension), fit, i+1) for i, fit in enumerate(fits)]
            for order in itertools.permutations(donors):
                best, median, worst = sorted(order, key=lambda x: x.fit)
                expected = target.pos + (scale * (best.pos-target.pos) + scale * (median.pos-worst.pos))
                originals = [x.pos.copy() for x in (target, *donors)]
                for factory in (current_to_ord_best, current_to_ord_pbest):
                    picker = Mock()
                    picker.get_random_best.return_value = order[0]
                    picker.get_random.side_effect = (
                        [list(order[:2]), [order[2]]] if factory is current_to_ord_best
                        else [[order[1]], [order[2]]]
                    )
                    f = Mock(return_value=scale)
                    actual = factory(f)(SimpleNamespace(individual=target, picker=picker))
                    np.testing.assert_array_equal(actual, expected)
                    f.assert_called_once_with()
                    self.assertFalse(np.shares_memory(actual, target.pos))
                for obj, original in zip((target, *donors), originals):
                    np.testing.assert_array_equal(obj.pos, original)

    def test_pbest_worse_than_both_random_donors(self):
        target = individual([0], 100, 0)
        donors = [individual([10], 3, 1), individual([20], 1, 2), individual([30], 2, 3)]
        picker = Mock()
        picker.get_random_best.return_value = donors[0]
        picker.get_random.side_effect = [[donors[1]], [donors[2]]]
        p = Mock(return_value=.2)
        trial = current_to_ord_pbest(lambda: .5, p=p)(SimpleNamespace(individual=target, picker=picker))
        np.testing.assert_array_equal(trial, [20])
        p.assert_called_once_with()
        picker.get_random_best.assert_called_once_with(p=.2)
        calls = picker.get_random.call_args_list
        self.assertEqual([id(x) for x in calls[0].kwargs['ignore']], [id(target), id(donors[0])])
        self.assertEqual([id(x) for x in calls[1].kwargs['ignore']], [id(target), id(donors[0]), id(donors[1])])
        self.assertFalse(calls[1].kwargs['include_archive'])

    def test_archive_extension_is_explicit_and_keeps_sampling_sequence(self):
        target = individual([0], 100, 0)
        donors = [individual([10], 3, 1), individual([20], 1, 2), individual([30], 2, 3)]
        for factory in (current_to_ord_best, current_to_ord_pbest):
            picker = Mock()
            picker.get_random_best.return_value = donors[0]
            picker.get_random.side_effect = [donors[:2], [donors[2]]] if factory is current_to_ord_best else [[donors[1]], [donors[2]]]
            factory(lambda: .5, include_archive=True)(SimpleNamespace(individual=target, picker=picker))
            self.assertTrue(picker.get_random.call_args_list[-1].kwargs['include_archive'])
            self.assertFalse(picker.get_random.call_args_list[0].kwargs.get('include_archive', False))

    def test_real_picker_returns_finite_trials_without_modifying_population(self):
        rng = np.random.default_rng(51)
        population = [individual(rng.normal(size=30), i, i) for i in range(20)]
        picker = DefaultPicker(population, list(range(20)), rng=rng)
        for factory in (current_to_ord_best, current_to_ord_pbest):
            trial = factory(lambda: .5)(SimpleNamespace(individual=population[-1], picker=picker))
            self.assertEqual(trial.shape, (30,))
            self.assertTrue(np.isfinite(trial).all())

    def test_factories_reference_existing_ordered_mutations(self):
        import autode.operators.mutation as mutation
        root = Path(__file__).resolve().parents[1]
        for name in ('de.py', 'de_batch.py'):
            for node in ast.walk(ast.parse((root / name).read_text())):
                if isinstance(node, ast.Attribute) and node.attr.startswith('current_to_ord_'):
                    self.assertTrue(hasattr(mutation, node.attr), (name, node.attr))


if __name__ == '__main__':
    unittest.main()
