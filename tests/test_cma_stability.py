import importlib
import math
from types import SimpleNamespace
import unittest

import numpy as np

cma = importlib.import_module('autode.operators.mutation.cma')


class CmaStabilityTests(unittest.TestCase):
    def update(self, optimizer, position):
        individuals = [SimpleNamespace(pos=np.array(position), fit=i) for i in range(4)]
        ctx = SimpleNamespace(
            parent=SimpleNamespace(num_gen=lambda: 0),
            population=SimpleNamespace(get_ranked=lambda: individuals),
        )
        with np.errstate(all='raise'):
            optimizer.tell(ctx)
        self.assertTrue(np.all(np.isfinite(optimizer._C)))

    def test_large_finite_path_saturates_without_overflow(self):
        optimizer = cma.Cma(np.zeros(2), sigma=0.5, population_size=4)
        self.update(optimizer, [1e6, 1e6])
        self.assertEqual(optimizer._sigma, cma._SIGMA_MAX)

    def test_small_sigma_is_floored(self):
        optimizer = cma.Cma(np.zeros(2), sigma=1e-20, population_size=4)
        self.update(optimizer, [0, 0])
        self.assertEqual(optimizer._sigma, cma._EPS)

    def test_normal_update_matches_multiplicative_formula(self):
        optimizer = cma.Cma(np.zeros(2), sigma=0.5, population_size=4)
        self.update(optimizer, [0.1, 0.2])
        expected = 0.5 * math.exp(
            optimizer._c_sigma / optimizer._d_sigma
            * (np.linalg.norm(optimizer._p_sigma) / optimizer._chi_n - 1)
        )
        self.assertAlmostEqual(optimizer._sigma, expected)

    def test_singular_covariance_update_is_finite(self):
        optimizer = cma.Cma(np.zeros(2), sigma=0.5, population_size=4, cov=np.zeros((2, 2)))
        self.update(optimizer, [0.1, 0.2])
        self.assertTrue(np.isfinite(optimizer._sigma))


if __name__ == '__main__':
    unittest.main()
