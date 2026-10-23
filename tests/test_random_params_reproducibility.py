from types import SimpleNamespace
import unittest

import numpy as np

from autode.core.events import (
    PubSub, GENERATION_START_EVT, GENERATION_ENDED_EVT,
    UPDATED_START_EVT, UPDATED_END_EVT,
)
from autode.core.params import Params
from autode.operators.params.random import Random


class RandomParamsReproducibilityTests(unittest.TestCase):
    def replay(self, seed):
        events = PubSub()
        params = Params(events)
        uniform = params.adaptative(Random.uniform(-2, 3))
        boolean = params.adaptative(Random.boolean(low=.2, high=.8, weighted=True))
        allocation = params.adaptative(Random.range(3, events))
        rng = np.random.default_rng(seed)
        output = []
        for generation, size in enumerate((12, 9, 6, 3)):
            ctx = SimpleNamespace(random=rng, popsize=size)
            events.publish(GENERATION_START_EVT, ctx)
            categories = []
            for i in range(size):
                individual = SimpleNamespace(
                    id=i, fit=float(i+1), pos=np.array([float(i), float(generation)]),
                    improved=i % 2 == 0, improvement=float(i+1),
                )
                events.publish(UPDATED_START_EVT, individual)
                u, b, category = uniform(), boolean(), allocation()
                self.assertTrue(-2 <= u < 3)
                self.assertIsInstance(b, bool)
                self.assertIn(category, (0, 1, 2))
                output.extend((u, b, category))
                categories.append(category)
                events.publish(UPDATED_END_EVT, individual)
            self.assertEqual(set(categories), {0, 1, 2})
            events.publish(GENERATION_ENDED_EVT, ctx)
            self.assertTrue(.2 <= boolean.get_value() <= .8)
            probabilities = np.asarray(allocation.get_value())
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertAlmostEqual(probabilities.sum(), 1)
            output.extend(probabilities)
        return np.asarray(output)

    def test_seeded_replay_including_adaptive_updates(self):
        np.testing.assert_array_equal(self.replay(42), self.replay(42))

    def test_different_seeds_produce_different_sequences(self):
        self.assertFalse(np.array_equal(self.replay(42), self.replay(43)))

    def test_boolean_learning_keeps_weighted_update_formula(self):
        param = Random.boolean(low=.2, high=.8, weighted=True, learning_rate=.1)(0)
        samples = [
            SimpleNamespace(value=True, individual=SimpleNamespace(improved=True, improvement=3.)),
            SimpleNamespace(value=False, individual=SimpleNamespace(improved=True, improvement=1.)),
        ]
        ctx = SimpleNamespace(random=np.random.default_rng(42))
        self.assertAlmostEqual(param.get_learnings(ctx, .5, samples).value, .525)
        self.assertAlmostEqual(param.get_learnings(ctx, .5, []).value, .47)

    def test_range_without_learning_handles_small_populations(self):
        def replay(seed):
            events = PubSub()
            param = Random.range(3, events, learnable=False)(0)
            ctx = SimpleNamespace(random=np.random.default_rng(seed), popsize=2)
            results = []
            for _ in range(4):
                events.publish(GENERATION_START_EVT, ctx)
                results.extend(param.get(ctx, param.initial_value, False) for _ in range(2))
                self.assertEqual(param.get_learnings(ctx, param.initial_value, []).value, param.initial_value)
            return results
        self.assertEqual(replay(42), replay(42))
        self.assertTrue(all(value in (0, 1, 2) for value in replay(42)))

    def test_boolean_without_learning_uses_context_rng(self):
        param = Random.boolean(p=.7, learning_rate=0)(0)
        ctx = SimpleNamespace(random=np.random.default_rng(42))
        self.assertEqual(param.get_learnings(ctx, .7, []).value, .7)
        self.assertIsInstance(param.get(ctx, .7, False), bool)


if __name__ == '__main__':
    unittest.main()
