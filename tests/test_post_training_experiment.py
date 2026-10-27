import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autode.lpsge.auto_de import ProblemInstance
from autode.lpsge.experiment_gen_algo import Experiment, ExperimentProblems, ExperimentState


class PostTrainingExperimentTests(unittest.TestCase):
    def problems(self):
        return [ProblemInstance(None, dim, -5., 5., 0., None, seed=7)
                for dim in (3, 5)]

    def test_testing_uses_saved_seeds_and_resumes_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4, max_experiments=2)
            problems = self.problems()
            state = ExperimentState(traning_set="F1", problems=ExperimentProblems(train=problems),
                                    test_algo="generated algorithm")
            calls = []
            def evaluate(phenotype, problem):
                calls.append((phenotype, problem.ndims, problem.seed))
                return [problem.seed], float(problem.seed)
            with patch("autode.lpsge.experiment_gen_algo.run_auto_de", side_effect=evaluate), \
                 patch("autode.lpsge.experiment_gen_algo.search_optimizer", side_effect=AssertionError):
                experiment._test(state)
                experiment._test(state)
                self.assertEqual(len(calls), 4)
                experiment._max_experiments = 3
                experiment._test(state)
            self.assertEqual(len(calls), 6)
            self.assertEqual([p.seed for p in problems], [7, 7])
            seeds = json.loads(Path(f"{name}_test.json").read_text())
            self.assertEqual(set(seeds), {"F1_3D", "F1_5D"})
            with Path(f"{name}_test.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            for row, (phenotype, dim, seed) in zip(rows, calls):
                self.assertEqual(phenotype, state.test_algo)
                self.assertEqual(row["problem"], f"F1_{dim}D")
                self.assertEqual(row["algo"], "AUTO_F1")
                self.assertEqual(int(row["seed"]), seed)
                self.assertEqual(seed, seeds[row["problem"]][int(row["no"]) - 1])

    def test_explicit_test_problems_are_kept_separate_from_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            train = self.problems()
            test = [ProblemInstance(None, 10, -5., 5., 0., None,
                                    name="held_out", seed=11)]
            experiment = Experiment(
                name, {"F1": ExperimentProblems(train=train, test=test)},
                "parameters.yml", 4, max_experiments=2)
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            with patch("autode.lpsge.experiment_gen_algo.search_optimizer", return_value=result) as search, \
                 patch("autode.lpsge.experiment_gen_algo.run_auto_de", return_value=([], 0.)) as evaluate:
                experiment.run()
                experiment.run()
            search.assert_called_once()
            self.assertIs(search.call_args.kwargs["problems"], train)
            self.assertEqual(evaluate.call_count, 2)
            seeds = json.loads(Path(f"{name}_test.json").read_text())
            self.assertEqual(set(seeds), {"held_out_10D"})
            for call, seed in zip(evaluate.call_args_list, seeds["held_out_10D"]):
                self.assertEqual(call.args[0], "generated algorithm")
                problem = call.args[1]
                self.assertEqual(problem.ndims, 10)
                self.assertEqual(problem.name, "held_out")
                self.assertEqual(problem.seed, seed)
                self.assertIsNot(problem, test[0])
            self.assertEqual([problem.seed for problem in train], [7, 7])
            self.assertEqual(test[0].seed, 11)
            with Path(f"{name}_test.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["problem"] for row in rows], ["held_out_10D"] * 2)

    def test_empty_test_set_does_not_fall_back_to_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4, max_experiments=1)
            state = ExperimentState(
                traning_set="F1",
                problems=ExperimentProblems(train=self.problems(), test=[]),
                test_algo="generated algorithm")
            with patch("autode.lpsge.experiment_gen_algo.run_auto_de") as evaluate:
                experiment._test(state)
            evaluate.assert_not_called()
            with Path(f"{name}_test.csv").open() as file:
                self.assertEqual(list(csv.DictReader(file)), [])

    def test_training_sets_have_independent_progress_and_algorithms(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4, max_experiments=1)
            state = ExperimentState(problems=ExperimentProblems(train=self.problems()))
            with patch("autode.lpsge.experiment_gen_algo.run_auto_de", return_value=([], 0.)) as evaluate:
                for training_set in ("F1", "F3"):
                    state.traning_set = training_set
                    state.test_algo = training_set + " algorithm"
                    experiment._test(state)
                self.assertEqual([call.args[0] for call in evaluate.call_args_list],
                                 ["F1 algorithm"] * 2 + ["F3 algorithm"] * 2)
                for training_set in ("F1", "F3"):
                    state.traning_set = training_set
                    state.test_algo = training_set + " algorithm"
                    experiment._test(state)
                self.assertEqual(evaluate.call_count, 4)
            with Path(f"{name}_test.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["algo"] for row in rows],
                             ["AUTO_F1"] * 2 + ["AUTO_F3"] * 2)
            seeds = json.loads(Path(f"{name}_test.json").read_text())
            self.assertEqual(set(seeds), {"F1_3D", "F1_5D", "F3_3D", "F3_5D"})
            self.assertEqual(len(list(Path(folder).glob("*_test.csv"))), 1)
            self.assertEqual(len(list(Path(folder).glob("*_test.json"))), 1)

    def test_resume_starts_at_first_incomplete_repetition(self):
        from autode.experiment.experiment import Experiment as EvaluationExperiment
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment_test.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["no", "algo", "problem", "solution", "fit", "seed"])
                for algo, problem, number in (
                    ("AUTO_F4", "F4_30D", 1), ("AUTO_F4", "F4_50D", 1),
                    ("AUTO_F19", "F19_30D", 1), ("AUTO_F19", "F19_50D", 1),
                    ("AUTO_F4", "F4_30D", 2),
                ):
                    writer.writerow([number, algo, problem, "[]", 0, number * 10])
            schedule = {key: [10, 20, 30] for key in
                        ("F4_30D", "F4_50D", "F19_30D", "F19_50D")}
            path.with_suffix(".json").write_text(json.dumps(schedule))
            evaluate = unittest.mock.Mock(return_value=([], 0.))
            experiment = EvaluationExperiment(
                str(path), {"AUTO_F4": evaluate},
                {"F4_30D": 30, "F4_50D": 50}, max_experiments=3,
                resume_by_pair=True)
            with patch("builtins.print") as output:
                experiment.run()
            self.assertNotIn(unittest.mock.call("Starting sample no 1"), output.call_args_list)
            self.assertIn(unittest.mock.call("Starting sample no 2"), output.call_args_list)
            self.assertEqual([(c.args[0], c.kwargs["seed"]) for c in evaluate.call_args_list],
                             [(50, 20), (30, 30), (50, 30)])
            experiment._init_csv()
            self.assertEqual(experiment._num_experiments, 3)
            experiment._algos = {"AUTO_F19": evaluate}
            experiment._problems = {"F19_30D": 30, "F19_50D": 50}
            experiment._init_csv()
            self.assertEqual(experiment._num_experiments, 1)

    def test_training_persists_seed_and_skips_completed_training(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, "parameters.yml", 4)
            state = ExperimentState(traning_set="F1", problems=ExperimentProblems(train=self.problems()))
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            with patch("autode.lpsge.experiment_gen_algo.search_optimizer", return_value=result) as search:
                experiment._train(state)
                experiment._train(state)
            search.assert_called_once_with(problems=state.problems.train,
                                           parameters_file="parameters.yml",
                                           popsize=4, nruns=5, elitesize=2,
                                           seed=json.loads(Path(f"{name}_train.json").read_text())["F1"][0])
            self.assertEqual(state.test_algo, "generated algorithm")
            with Path(f"{name}_train.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["phenotype"], "generated algorithm")
            self.assertEqual(int(rows[0]["seed"]), search.call_args.kwargs["seed"])

    def test_training_generates_one_master_seed_per_set_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            training_sets = {
                key: ExperimentProblems(train=self.problems())
                for key in ("F1", "F3")
            }
            experiment = Experiment(name, training_sets, None, 4,
                                    nruns=5, max_experiments=51)
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            with patch("autode.lpsge.experiment_gen_algo.search_optimizer", return_value=result) as search, \
                 patch.object(experiment, "_test"):
                experiment.run()
                seed_path = Path(f"{name}_train.json")
                saved = seed_path.read_bytes()
                schedule = json.loads(saved)
                self.assertEqual(set(schedule), set(training_sets))
                self.assertTrue(all(len(seeds) == 1 for seeds in schedule.values()))
                masters = [schedule[key][0] for key in training_sets]
                self.assertTrue(all(type(seed) is int and 0 <= seed < 2**32
                                    for seed in masters))
                self.assertEqual(len(set(masters)), len(training_sets))
                self.assertEqual([call.kwargs["seed"] for call in search.call_args_list],
                                 masters)
                with patch("autode.experiment.seeds.secrets.randbits",
                           side_effect=AssertionError("Existing seeds must be reused")):
                    experiment.run()
                self.assertEqual(seed_path.read_bytes(), saved)
                self.assertEqual(search.call_count, len(training_sets))

    def test_training_uses_each_sets_saved_seed_and_upgrades_legacy_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / "experiment")
            experiment = Experiment(name, {}, None, 4)
            schedule = {"F1": [2**32 - 2], "F3": [2**32 - 1]}
            Path(f"{name}_train.json").write_text(json.dumps(schedule))
            Path(f"{name}_train.csv").write_text(
                "problem,genotype,phenotype,error,runtime(s),runtime(g),runtime(fes),pinit\n"
                "old,[],old algorithm,1,2,3,4,4\n")
            result = dict(genotype=[], phenotype="generated algorithm", fitness=1.,
                          runtime=2., num_g=3, num_evals=4)
            state = ExperimentState(problems=ExperimentProblems(train=self.problems()))
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
