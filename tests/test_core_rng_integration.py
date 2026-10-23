import pickle
import unittest

import numpy as np

import autode.operators as ops
from autode.core import (
    create_optimizer, PubSub, StopCondition, UpdateMethod, BatchUpdateMethod,
    Params,
)
from autode.core.events import GENERATION_START_EVT, GENERATION_FINALIZED_EVT


class CoreRngIntegrationTests(unittest.TestCase):
    def run_search(self, seed, update_type, adaptive):
        events = PubSub()
        params = Params(events, memory=ops.params.lshade.memory(5))
        f = params.adaptative(ops.params.lshade.cauchy()) if adaptive else params.static(.5)
        cr = params.adaptative(ops.params.lshade.normal()) if adaptive else params.static(.7)
        archive = ops.archive.fixed(6)
        checked = {'generation': 0, 'individual': 0, 'finalized': 0}
        trace = []
        mutation = ops.mutation.current_to_pbest(f, p=.3, include_archive=True)
        def check_generation(ctx):
            self.assertIs(ctx.picker._rng, ctx.random)
            self.assertIs(archive._rng, ctx.random)
            checked['generation'] += 1
        def mutate(ctx):
            self.assertIs(ctx.picker._rng, ctx.random)
            checked['individual'] += 1
            return mutation(ctx)
        def check_finalized(ctx):
            # _boost_if_applies replaces this picker even without a booster.
            self.assertIs(ctx.picker._rng, ctx.random)
            checked['finalized'] += 1
        events.subscribe(GENERATION_START_EVT, check_generation)
        events.subscribe(GENERATION_FINALIZED_EVT, check_finalized)
        search = create_optimizer(
            init_strategy=ops.population.init.random_batch(12),
            update_strategy=[update_type(
                mutation=mutate, crossover=ops.crossover.binomial(cr),
                reparation=ops.reparation.clip(), selection=ops.selection.elitist(),
            )],
            stop_condition=StopCondition(max_evals=120), events=events,
            params=params, archive=archive, seed=seed,
        )
        def objective(x):
            array = np.asarray(x)
            trace.extend(array.reshape(-1, 3).copy())
            return np.sum(array**2, axis=-1)
        position, fitness = search(objective, -5., 5., 3)
        self.assertEqual(len(trace), 120)
        self.assertGreater(checked['generation'], 0)
        self.assertGreater(checked['individual'], 0)
        self.assertEqual(checked['generation'], checked['finalized'])
        self.assertAlmostEqual(float(position @ position), fitness)
        return position, fitness, np.asarray(trace)

    def test_seeded_scalar_and_batch_searches(self):
        for method in (UpdateMethod, BatchUpdateMethod):
            for adaptive in (False, True):
                with self.subTest(method=method.__name__, adaptive=adaptive):
                    first = self.run_search(42, method, adaptive)
                    second = self.run_search(42, method, adaptive)
                    np.testing.assert_array_equal(first[0], second[0])
                    self.assertEqual(first[1], second[1])
                    np.testing.assert_array_equal(first[2], second[2])
                    other = self.run_search(43, method, adaptive)
                    self.assertFalse(np.array_equal(first[2], other[2]))

    def test_search_leaves_global_numpy_rng_unchanged(self):
        state = pickle.dumps(np.random.get_state())
        self.run_search(42, BatchUpdateMethod, True)
        self.assertEqual(pickle.dumps(np.random.get_state()), state)


if __name__ == '__main__':
    unittest.main()
