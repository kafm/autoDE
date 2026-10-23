import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autode.lpsge.auto_de import ProblemInstance
from autode.lpsge.experiment_gen_algo import Experiment, ExperimentState


class PostTrainingExperimentTests(unittest.TestCase):
    def problems(self):
        return [ProblemInstance(None, dim, -5., 5., 0., None, seed=7)
                for dim in (3, 5)]

    def test_testing_uses_saved_seeds_and_resumes_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4, max_experiments=2)
            problems = self.problems()
            state = ExperimentState(traning_set="F1", problems=problems,
                                    test_algo="generated algorithm")
            calls = []
            def evaluate(phenotype, problem):
                calls.append((phenotype, problem.ndims, problem.seed))
                return [problem.seed], float(problem.seed)
            legacy = Path(f"{name}_test.csv")
            legacy.write_text("legacy results\n")
            with patch("autode.lpsge.experiment_gen_algo.run_auto_de", side_effect=evaluate), \
                 patch("autode.lpsge.experiment_gen_algo.search_optimizer", side_effect=AssertionError):
                experiment._test(state)
                experiment._test(state)
                self.assertEqual(len(calls), 4)
                experiment._max_experiments = 3
                experiment._test(state)
            self.assertEqual(len(calls), 6)
            self.assertEqual([p.seed for p in problems], [7, 7])
            self.assertEqual(legacy.read_text(), "legacy results\n")
            seeds = json.loads(Path(f"{name}_F1_test.json").read_text())
            self.assertEqual(set(seeds), {"F1_3D", "F1_5D"})
            with Path(f"{name}_F1_test.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            for row, (phenotype, dim, seed) in zip(rows, calls):
                self.assertEqual(phenotype, state.test_algo)
                self.assertEqual(row["problem"], f"F1_{dim}D")
                self.assertEqual(row["algo"], "AUTO")
                self.assertEqual(int(row["seed"]), seed)
                self.assertEqual(seed, seeds[row["problem"]][int(row["no"]) - 1])

    def test_training_sets_have_independent_progress_and_algorithms(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4, max_experiments=1)
            state = ExperimentState(problems=self.problems())
            with patch("autode.lpsge.experiment_gen_algo.run_auto_de", return_value=([], 0.)) as evaluate:
                for training_set in ("F1", "F3"):
                    state.traning_set = training_set
                    state.test_algo = training_set + " algorithm"
                    experiment._test(state)
                self.assertEqual([call.args[0] for call in evaluate.call_args_list],
                                 ["F1 algorithm"] * 2 + ["F3 algorithm"] * 2)
            for training_set in ("F1", "F3"):
                self.assertTrue(Path(f"{name}_{training_set}_test.csv").exists())
                self.assertTrue(Path(f"{name}_{training_set}_test.json").exists())

    def test_training_persists_seed_and_skips_completed_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, "parameters.yml", 4)
            state = ExperimentState(traning_set="F1", problems=self.problems())
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            with patch("autode.lpsge.experiment_gen_algo.search_optimizer", return_value=result) as search:
                experiment._train(state)
                experiment._train(state)
            search.assert_called_once_with(problems=state.problems,
                                           parameters_file="parameters.yml",
                                           popsize=4, nruns=5, elitesize=2,
                                           seed=json.loads(Path(f"{name}_train.json").read_text())["F1"][0])
            self.assertEqual(state.test_algo, "generated algorithm")
            with Path(f"{name}_train.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["phenotype"], "generated algorithm")
            self.assertEqual(int(rows[0]["seed"]), search.call_args.kwargs["seed"])

    def test_training_uses_each_sets_saved_seed_and_upgrades_legacy_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4)
            schedule = {"F1": [2**63 + 5], "F3": [2**63 + 6]}
            Path(f"{name}_train.json").write_text(json.dumps(schedule))
            Path(f"{name}_train.csv").write_text(
                "problem,genotype,phenotype,error,runtime(s),runtime(g),runtime(fes),pinit\n"
                "old,[],old algorithm,1,2,3,4,4\n")
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            state = ExperimentState(problems=self.problems())
            with patch("autode.lpsge.experiment_gen_algo.search_optimizer", return_value=result) as search:
                for name_of_set in ("F1", "F3"):
                    state.traning_set = name_of_set
                    experiment._train(state)
            self.assertEqual([c.kwargs["seed"] for c in search.call_args_list],
                             [schedule["F1"][0], schedule["F3"][0]])
            self.assertEqual(json.loads(Path(f"{name}_train.json").read_text()), schedule)
            with Path(f"{name}_train.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["seed"], "")
            self.assertEqual(rows[1]["seed"], str(schedule["F1"][0]))
            self.assertEqual(rows[2]["seed"], str(schedule["F3"][0]))


if __name__ == "__main__":
    unittest.main()
