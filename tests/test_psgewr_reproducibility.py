import unittest
import copy
from unittest.mock import patch

import numpy as np
from joblib import Parallel, delayed

from autode.lpsge.auto_de import ProblemInstance, auto_de_evals, search_optimizer
from autode.lpsge.psgewr import EvalFn, PSGEWR
from autode.sge.parameters import params


def seeded_evaluation(individual, *, seed):
    return [float(np.random.default_rng(seed).random())]


class PSGEWRReproducibilityTests(unittest.TestCase):
    def search(self, seed=42):
        search = PSGEWR(EvalFn(seeded_evaluation, [0.0]), None, 4, nruns=3)
        search._evaluation_seed = seed
        return search

    def test_repetition_and_candidate_streams_are_distinct_and_replayable(self):
        search = self.search()
        first = search._evaluate("a", 0)
        second = search._evaluate("b", 3)
        self.assertEqual(first, self.search()._evaluate("a", 0))
        self.assertEqual(len(set(x[0] for x in first + second)), 6)
        self.assertNotEqual(first, self.search(43)._evaluate("a", 0))

    def test_parallel_workers_match_serial_execution(self):
        search = self.search()
        expected = [search._evaluate("a", offset) for offset in (0, 3)]
        actual = Parallel(n_jobs=2, backend="threading", pre_dispatch=2, batch_size=1)(
            delayed(search._evaluate)("a", offset) for offset in (0, 3)
        )
        self.assertEqual(expected, actual)

    def test_problem_seeds_are_derived_without_mutating_inputs(self):
        problems = [ProblemInstance(None, 2, -1, 1, 0, None, seed=s)
                    for s in (None, 7)]
        with patch("autode.lpsge.auto_de.auto_de_eval",
                   side_effect=lambda individual, problem: problem.seed):
            first = auto_de_evals("a", problems, seed=42)
            self.assertEqual(first, auto_de_evals("a", problems, seed=42))
            self.assertNotEqual(first, auto_de_evals("a", problems, seed=43))
            self.assertNotEqual(first[0], first[1])
            self.assertEqual(auto_de_evals("a", problems), [None, 7])
        self.assertEqual([p.seed for p in problems], [None, 7])

    def test_saved_64bit_seed_overrides_configuration_and_replays(self):
        original = copy.deepcopy(params)
        rng_state = np.random.get_state()
        def run(seed):
            search = PSGEWR(EvalFn(seeded_evaluation, [0.]), "parameters.yml", 4, seed=seed)
            with patch("autode.lpsge.psgewr.load_parameters",
                       side_effect=lambda **kw: params.update(SEED=99)), \
                 patch("autode.lpsge.psgewr.logger"), \
                 patch("autode.lpsge.psgewr.grammar"):
                search._setup()
            self.assertEqual(params["SEED"], seed)
            return np.random.random(5).tolist(), search._evaluate("a", 0)
        try:
            first = run(2**63 + 5)
            self.assertEqual(first, run(2**63 + 5))
            self.assertNotEqual(first, run(2**63 + 6))
        finally:
            params.clear()
            params.update(original)
            np.random.set_state(rng_state)

    def test_search_optimizer_forwards_training_seed(self):
        with patch("autode.lpsge.auto_de.PSGEWR") as search:
            search.return_value.run.return_value = {}
            search_optimizer([], None, 4, seed=2**63 + 5)
        self.assertEqual(search.call_args.kwargs["seed"], 2**63 + 5)


if __name__ == "__main__":
    unittest.main()
