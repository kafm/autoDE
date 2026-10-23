import contextlib
import copy
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from joblib import Parallel

import autode.sge.grammar as grammar
from autode.sge.parameters import params
from autode.lpsge.psgewr import EvalFn, PSGEWR


def evaluate(individual, *, seed):
    return [float(np.random.default_rng(seed).uniform(1., 2.))]


class PSGEWRResetTests(unittest.TestCase):
    def test_full_runs_replay_and_restore_caller_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "grammar.pybnf").write_text("<start> ::= <choice>\n<choice> ::= a | b\n")
            configuration = {
                "GRAMMAR": str(path / "grammar.pybnf"), "GENERATIONS": 2,
                "MIN_TREE_DEPTH": 3, "MAX_TREE_DEPTH": 6, "TSIZE": 2,
                "ADAPTIVE_LF": True, "LEARNING_FACTOR": .01,
                "EXPERIMENT_NAME": str(path / "logs"),
            }
            config_path = path / "parameters.yml"
            config_path.write_text(yaml.safe_dump(configuration))
            search = PSGEWR(EvalFn(evaluate, [0.]), str(config_path), 4,
                            nruns=2, seed=42)
            original_parameters = copy.deepcopy(params)
            original_rng = pickle.dumps(np.random.get_state())
            def run(instance):
                # Compatibility shims allow exercising the actual evolution
                # loop on this environment's older joblib / newer NumPy.
                with patch("autode.lpsge.psgewr.logger"), \
                     patch("autode.lpsge.psgewr.Parallel", side_effect=lambda **kw:
                           Parallel(n_jobs=2, backend="threading", pre_dispatch=2, batch_size=1)), \
                     patch.object(np, "infty", np.inf, create=True), \
                     contextlib.redirect_stdout(io.StringIO()):
                    return instance.run()
            first = run(search)
            second = run(search)
            fresh = run(PSGEWR(EvalFn(evaluate, [0.]), str(config_path), 4,
                               nruns=2, seed=42))
            self.assertEqual(first, second)
            self.assertEqual(first, fresh)
            self.assertEqual(first["num_evals"], 20)
            self.assertEqual(params, original_parameters)
            self.assertEqual(pickle.dumps(np.random.get_state()), original_rng)

    def test_grammar_switch_clears_rules_and_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name, source in (("first", "<first> ::= a | b\n"),
                                 ("second", "<second> ::= c\n")):
                grammar_path = path / f"{name}.pybnf"
                grammar_path.write_text(source)
                config_path = path / f"{name}.yml"
                config_path.write_text(yaml.safe_dump({"GRAMMAR": str(grammar_path)}))
                search = PSGEWR(EvalFn(evaluate, [0.]), str(config_path), 4, seed=42)
                with patch("autode.lpsge.psgewr.logger"):
                    search._setup()
                self.assertEqual(set(grammar.get_dict()), {f"<{name}>"})
                self.assertEqual(list(grammar.ordered_non_terminals), [f"<{name}>"])
                grammar.count_number_of_options_in_production()
            self.assertEqual(grammar.count_number_of_options_in_production(), {"<second>": 1})

    def test_failure_restores_parameters_rng_and_resets_next_run(self):
        original_parameters = copy.deepcopy(params)
        original_rng = pickle.dumps(np.random.get_state())
        search = PSGEWR(EvalFn(evaluate, [0.]), None, 4, seed=42)
        def fail():
            params["LEARNING_FACTOR"] = .99
            np.random.random(10)
            search._numevals = 123
            raise ValueError("failed")
        with patch.object(search, "_run", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "failed"):
                search.run()
        self.assertEqual(params, original_parameters)
        self.assertEqual(pickle.dumps(np.random.get_state()), original_rng)
        with patch("autode.lpsge.psgewr.logger"), patch("autode.lpsge.psgewr.grammar"):
            search._setup()
        self.assertEqual(search._numevals, 0)
        self.assertEqual(params["LEARNING_FACTOR"], search._base_parameters["LEARNING_FACTOR"])

    def setUp(self):
        self.parameters = copy.deepcopy(params)
        self.random_state = np.random.get_state()
        self.grammar_state = copy.deepcopy(grammar._inst.__dict__)

    def tearDown(self):
        params.clear()
        params.update(self.parameters)
        np.random.set_state(self.random_state)
        grammar._inst.__dict__.clear()
        grammar._inst.__dict__.update(self.grammar_state)
        grammar.ordered_non_terminals = grammar._inst.ordered_non_terminals


if __name__ == "__main__":
    unittest.main()
