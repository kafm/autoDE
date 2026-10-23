import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from autode.experiment.experiment import Experiment


class ExperimentSeedsTests(unittest.TestCase):
    def experiment(self, path, count=3, problems=("F1", "F3")):
        return Experiment(str(path), {}, dict.fromkeys(problems), max_experiments=count)

    def test_run_creates_schedule_even_with_no_remaining_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            path.write_text("no,algo,problem,solution,fit\n3,A,F1,[],0\n")
            experiment = self.experiment(path)
            experiment.run()
            seeds = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(set(seeds), {"F1", "F3"})
            self.assertTrue(all(len(values) == 3 for values in seeds.values()))
            self.assertEqual(len(set(seeds["F1"] + seeds["F3"])), 6)
            self.assertEqual(experiment._seeds, seeds)

    def test_extension_preserves_existing_seeds_and_other_problems(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            original = {"F1": [10], "F3": [20, 21, 22, 23], "old": [30]}
            path.with_suffix(".json").write_text(json.dumps(original))
            experiment = self.experiment(path, problems=("F1", "F3", "F4"))
            experiment._init_seeds()
            seeds = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(seeds["F1"][0], 10)
            self.assertEqual(len(seeds["F1"]), 3)
            self.assertEqual(seeds["F3"], original["F3"])
            self.assertEqual(seeds["old"], original["old"])
            self.assertEqual(len(seeds["F4"]), 3)
            saved = path.with_suffix(".json").read_bytes()
            with patch("autode.experiment.seeds.secrets.randbits", side_effect=AssertionError):
                self.experiment(path, count=2)._init_seeds()
            self.assertEqual(path.with_suffix(".json").read_bytes(), saved)

    def test_invalid_schedule_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            invalid = '{"F1": [null]}'
            path.with_suffix(".json").write_text(invalid)
            with self.assertRaises(ValueError):
                self.experiment(path)._init_seeds()
            self.assertEqual(path.with_suffix(".json").read_text(), invalid)

    def test_saved_seeds_reach_all_algorithms_and_csv_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            schedule = {"F1": [10, 11], "F3": [20, 21]}
            path.with_suffix(".json").write_text(json.dumps(schedule))
            calls = []
            def algorithm(problem, *, seed):
                calls.append((problem, seed))
                return [seed], float(seed)
            def run(count):
                Experiment(str(path), {"A": algorithm, "B": algorithm},
                           {"F1": "first", "F3": "second"},
                           max_experiments=count).run()
            run(1)
            run(2)
            self.assertEqual(calls, [("first", 10)] * 2 + [("second", 20)] * 2
                             + [("first", 11)] * 2 + [("second", 21)] * 2)
            with path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertEqual(int(row["seed"]), schedule[row["problem"]][int(row["no"]) - 1])

    def test_legacy_csv_preserves_rows_without_inventing_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            path.write_text("no,algo,problem,solution,fit\n1,A,F1,[],0\n")
            path.with_suffix(".json").write_text('{"F1": [10, 11]}')
            calls = []
            def algorithm(problem, *, seed):
                calls.append(seed)
                return [], 0.0
            Experiment(str(path), {"A": algorithm}, {"F1": None},
                       max_experiments=2).run()
            with path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(calls, [11])
            self.assertEqual(rows[0]["seed"], "")
            self.assertEqual(rows[1]["seed"], "11")

    def test_competitor_wrappers_replay_with_the_supplied_seed(self):
        from types import SimpleNamespace
        from autode.core import StopCondition
        from experiment_competitors import lshade_spacma, nlshade, ebde, eblshade
        for algorithm in (lshade_spacma, nlshade, ebde, eblshade):
            with self.subTest(algorithm=algorithm.__name__):
                def run(seed):
                    trace = []
                    def objective(x):
                        x = np.asarray(x)
                        trace.extend(x.reshape(-1, 3).copy())
                        return np.sum(x**2, axis=-1)
                    problem = SimpleNamespace(problem=objective, lb=-5., ub=5.)
                    result = algorithm(problem, ndims=3, popsize=12,
                                       stop_condition=StopCondition(max_evals=120), seed=seed)
                    return result, np.asarray(trace)
                first, trace = run(42)
                second, repeated = run(42)
                np.testing.assert_array_equal(first[0], second[0])
                self.assertEqual(first[1], second[1])
                np.testing.assert_array_equal(trace, repeated)
                self.assertFalse(np.array_equal(trace, run(43)[1]))


if __name__ == "__main__":
    unittest.main()
