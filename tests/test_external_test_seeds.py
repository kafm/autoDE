import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autode.lpsge.auto_de import ProblemInstance
from autode.lpsge.experiment_gen_algo import Experiment, ExperimentProblems, ExperimentState


class ExternalTestSeedsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.name = str(self.root / 'experiment')
        self.train = [ProblemInstance(None, 3, -5., 5., 0., None, name='train')]
        self.test = [ProblemInstance(None, dim, -5., 5., 0., None, name='F1', seed=7)
                     for dim in (30, 50)]

    def source(self, name, seeds):
        path = self.root / name
        path.write_text(json.dumps(seeds))
        return str(path)

    def experiment(self, source, count=2):
        return Experiment(self.name, {}, None, 4, max_experiments=count,
                          test_seed_file=source)

    def state(self, name='training'):
        return ExperimentState(traning_set=name, test_algo=name,
                               problems=ExperimentProblems(self.train, self.test))

    def test_dimension_sources_are_shared_across_algorithms_and_resume(self):
        sources = {30: self.source('30d.json', {'F1': [10, 11, 12]}),
                   50: self.source('50d.json', {'F1': [20, 21, 22]})}
        original = {path: Path(path).read_bytes() for path in sources.values()}
        experiment = self.experiment(sources)
        with patch('autode.lpsge.experiment_gen_algo.run_auto_de', return_value=([], 0.)) as evaluate:
            for name in ('first', 'second'):
                experiment._test(self.state(name))
                experiment._test(self.state(name))
            self.assertEqual([(c.args[1].ndims, c.args[1].seed) for c in evaluate.call_args_list],
                             [(30, 10), (50, 20), (30, 11), (50, 21)] * 2)
            experiment._max_experiments = 3
            experiment._test(self.state('first'))
            self.assertEqual([c.args[1].seed for c in evaluate.call_args_list[-2:]], [12, 22])
        for path, data in original.items():
            self.assertEqual(Path(path).read_bytes(), data)
        self.assertEqual([p.seed for p in self.test], [7, 7])
        saved = json.loads(Path(f'{self.name}_first_test.json').read_text())
        self.assertEqual(saved, {'F1_30D': [10, 11, 12], 'F1_50D': [20, 21, 22]})

    def test_single_file_uses_full_problem_keys(self):
        source = self.source('seeds.json', {'F1_30D': [10, 11], 'F1_50D': [20, 21]})
        with patch('autode.lpsge.experiment_gen_algo.run_auto_de', return_value=([], 0.)) as evaluate:
            self.experiment(source)._test(self.state())
        self.assertEqual([c.args[1].seed for c in evaluate.call_args_list], [10, 20, 11, 21])

    def test_invalid_or_incomplete_sources_fail_before_evaluation(self):
        for seeds in ({'F1_30D': [1, 2]},
                      {'F1_30D': [1], 'F1_50D': [2, 3]},
                      {'F1_30D': [True, 2], 'F1_50D': [2, 3]},
                      {'F1_30D': [-1, 2], 'F1_50D': [2, 3]}):
            with self.subTest(seeds=seeds):
                source = self.source('invalid.json', seeds)
                original = Path(source).read_bytes()
                with patch('autode.lpsge.experiment_gen_algo.run_auto_de') as evaluate:
                    with self.assertRaises(ValueError):
                        self.experiment(source)._test(self.state())
                    evaluate.assert_not_called()
                self.assertEqual(Path(source).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'dimension 50'):
            self.experiment({30: self.source('30d.json', {'F1': [1, 2]})})._test(self.state())
        self.assertFalse(Path(f'{self.name}_training_test.csv').exists())

    def test_changed_source_is_rejected_without_overwriting_results(self):
        source = self.source('seeds.json', {'F1_30D': [10, 11], 'F1_50D': [20, 21]})
        experiment = self.experiment(source)
        with patch('autode.lpsge.experiment_gen_algo.run_auto_de', return_value=([], 0.)):
            experiment._test(self.state())
        result = Path(f'{self.name}_training_test.csv')
        schedule = result.with_suffix('.json')
        original = (result.read_bytes(), schedule.read_bytes())
        Path(source).write_text(json.dumps({'F1_30D': [99, 11], 'F1_50D': [20, 21]}))
        with patch('autode.lpsge.experiment_gen_algo.run_auto_de') as evaluate:
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                experiment._test(self.state())
            evaluate.assert_not_called()
        self.assertEqual((result.read_bytes(), schedule.read_bytes()), original)

    def test_existing_csv_seeds_must_match_even_without_saved_schedule(self):
        source = self.source('seeds.json', {'F1_30D': [10, 11], 'F1_50D': [20, 21]})
        result = Path(f'{self.name}_training_test.csv')
        with result.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['no', 'algo', 'problem', 'solution', 'fit', 'seed'])
            writer.writerow([1, 'AUTO', 'F1_30D', '[]', 0, 999])
        with self.assertRaisesRegex(ValueError, 'Existing results conflict'):
            self.experiment(source)._test(self.state())
        self.assertFalse(result.with_suffix('.json').exists())


if __name__ == '__main__':
    unittest.main()
