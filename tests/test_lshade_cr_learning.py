import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

from autode.core.params import ParamValueLearned
from autode.operators.params.lshade import LShade, _LShadeMemory


def sample(value, improvement=1., improved=True):
    return SimpleNamespace(
        value=value,
        individual=SimpleNamespace(improved=improved, improvement=improvement),
    )


class LShadeCrLearningTests(unittest.TestCase):
    def test_zero_successful_cr_enters_terminal_state_without_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            learned = LShade.normal_learning_strategy()(None, .5, [sample(0), sample(0, 9)])
        self.assertEqual(learned.value, 0)
        self.assertTrue(learned.think_fast)

    def test_positive_successes_use_improvement_weighted_lehmer_mean(self):
        learned = LShade.normal_learning_strategy()(None, .5, [sample(.2), sample(.8, 9), sample(1, improved=False)])
        self.assertAlmostEqual(learned.value, (.1 * .2**2 + .9 * .8**2) / (.1 * .2 + .9 * .8))
        self.assertFalse(learned.think_fast)

    def test_mixed_zero_and_positive_successes_do_not_lock(self):
        learned = LShade.normal_learning_strategy()(None, .5, [sample(0), sample(.8, 9)])
        self.assertAlmostEqual(learned.value, .8)
        self.assertFalse(learned.think_fast)

    def test_no_successes_preserve_previous_value(self):
        for samples in ([], [sample(.8, improved=False)]):
            learned = LShade.normal_learning_strategy()(None, .37, samples)
            self.assertEqual(learned.value, .37)
            self.assertFalse(learned.think_fast)

    def test_terminal_memory_persists_and_forces_zero_sampling(self):
        param = Mock()
        terminal = ParamValueLearned(value=0, think_fast=True)
        learned = _LShadeMemory(5)._get_param_learnings(param, terminal)
        self.assertIs(learned, terminal)
        param.get_learnings.assert_not_called()
        generator = LShade.normal()(0)
        for _ in range(10):
            self.assertEqual(generator.get(None, learned.value, learned.think_fast), 0)


if __name__ == '__main__':
    unittest.main()
