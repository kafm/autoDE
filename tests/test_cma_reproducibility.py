import importlib
import pickle
from types import SimpleNamespace
import unittest

import numpy as np

from autode.core.events import PubSub

cma = importlib.import_module('autode.operators.mutation.cma')


class CmaReproducibilityTests(unittest.TestCase):
    def optimizer(self, seed=42):
        return cma.Cma(np.zeros(3), sigma=.5, rng=np.random.default_rng(seed), population_size=6)

    def generation(self, optimizer, generation):
        positions = [optimizer.ask() for _ in range(6)]
        individuals = sorted(
            [SimpleNamespace(pos=p, fit=float(p @ p)) for p in positions],
            key=lambda i: i.fit,
        )
        ctx = SimpleNamespace(
            parent=SimpleNamespace(num_gen=lambda: generation),
            population=SimpleNamespace(get_ranked=lambda: individuals),
        )
        optimizer.tell(ctx)
        return np.array(positions)

    def test_same_seed_reproduces_sampling_and_adaptation(self):
        first, second = self.optimizer(), self.optimizer()
        for generation in range(4):
            np.testing.assert_array_equal(self.generation(first, generation), self.generation(second, generation))
            np.testing.assert_array_equal(first._C, second._C)
            self.assertEqual(first._sigma, second._sigma)

    def test_different_seeds_produce_different_proposals(self):
        self.assertFalse(np.array_equal(self.optimizer(42).ask(), self.optimizer(43).ask()))

    def test_pickle_continues_rng_and_adaptation(self):
        optimizer = self.optimizer()
        self.generation(optimizer, 0)
        restored = pickle.loads(pickle.dumps(optimizer))
        self.assertIsInstance(restored._rng, np.random.Generator)
        for generation in range(1, 4):
            np.testing.assert_array_equal(self.generation(optimizer, generation), self.generation(restored, generation))

    def test_pickle_preserves_shared_generator_identity(self):
        optimizer = self.optimizer()
        optimizer.ask()
        restored, rng = pickle.loads(pickle.dumps((optimizer, optimizer._rng)))
        self.assertIs(restored._rng, rng)
        np.testing.assert_array_equal(optimizer.ask(), restored.ask())
        np.testing.assert_array_equal(optimizer._rng.standard_normal(3), rng.standard_normal(3))

    def test_wrapper_initialization_and_resize_continue_shared_stream(self):
        wrappers = [cma.CmaWrapper(PubSub(), .5) for _ in range(2)]
        generators = [np.random.default_rng(42) for _ in range(2)]
        for wrapper, rng in zip(wrappers, generators):
            ctx = SimpleNamespace(popsize=6, parent=SimpleNamespace(ndims=lambda: 3), random=rng)
            wrapper._start(ctx)
            self.assertIs(wrapper._optimizer._rng, rng)
        np.testing.assert_array_equal(wrappers[0].ask(None), wrappers[1].ask(None))
        for wrapper, rng in zip(wrappers, generators):
            state = pickle.dumps(rng.bit_generator.state)
            wrapper._start(SimpleNamespace(popsize=4, random=rng))
            self.assertIs(wrapper._optimizer._rng, rng)
            self.assertEqual(pickle.dumps(rng.bit_generator.state), state)
        np.testing.assert_array_equal(wrappers[0].ask(None), wrappers[1].ask(None))


if __name__ == '__main__':
    unittest.main()
