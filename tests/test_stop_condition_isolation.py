import unittest

import numpy as np

from autode.core import create_optimizer, PubSub, StopCondition, UpdateMethod
from autode.core.events import GENERATION_START_EVT
import autode.operators as ops


class StopConditionIsolationTests(unittest.TestCase):
    def optimizer(self, condition, popsize, budgets):
        events = PubSub()
        events.subscribe(GENERATION_START_EVT,
                         lambda ctx: budgets.append(ctx.parent.max_evals()))
        return create_optimizer(
            init_strategy=ops.population.init.random(popsize),
            update_strategy=[UpdateMethod(
                mutation=lambda ctx: ctx.individual.pos.copy(),
                crossover=ops.crossover.all(),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )],
            stop_condition=condition, events=events, seed=42,
        )

    def test_shared_generation_limit_infers_independent_budgets(self):
        condition = StopCondition(max_gen=2)
        budgets = []
        for size in (8, 12, 8):
            search = self.optimizer(condition, size, budgets)
            search(lambda x: float(np.sum(np.asarray(x)**2)), -5., 5., 3)
            self.assertIsNone(condition.max_evals())
            self.assertEqual(budgets[-1], 2 * size)

    def test_reused_optimizer_does_not_retain_inferred_budget(self):
        condition = StopCondition(max_gen=2)
        budgets = []
        search = self.optimizer(condition, 8, budgets).__self__
        search.search(lambda x: float(np.sum(np.asarray(x)**2)), -5., 5., 3)
        self.assertEqual(budgets[-1], 16)
        search._init_strategy = ops.population.init.random(12)
        search.search(lambda x: float(np.sum(np.asarray(x)**2)), -5., 5., 3)
        self.assertEqual(budgets[-1], 24)
        self.assertIsNone(search._stop_condition.max_evals())

    def test_explicit_budget_and_callback_are_preserved(self):
        calls = []
        condition = StopCondition(max_evals=40,
                                  callback=lambda ctx: calls.append(ctx.num_evals()) or True)
        search = self.optimizer(condition, 8, [])
        search(lambda x: float(np.sum(np.asarray(x)**2)), -5., 5., 3)
        self.assertTrue(calls)
        self.assertEqual(condition.max_evals(), 40)


if __name__ == "__main__":
    unittest.main()
