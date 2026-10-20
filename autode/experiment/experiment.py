from typing import Callable, Tuple, Dict, List, Any
import os, csv, time
from .cec2017_problems import Benchmark


class Experiment:
    def __init__(
        self,
        filename: str,
        algos: Dict[str, Callable[[Benchmark], Tuple[List[float], float]]],
        problems: Dict[str, Benchmark],
        max_experiments: int = 51,
        save_point: int = 1,
    ):
        self._filename = filename
        self._algos = algos
        self._problems = problems
        self._max_experiments = max_experiments
        self._save_point = save_point
        self._headers = ["no", "algo", "problem", "solution", "fit"]
        self._num_experiments = 0

    def run(self):
        self._init_csv()
        i = self._num_experiments
        gstart = time.time()
        print(f"Starting experiment")
        while i < self._max_experiments:
            i += 1
            results = []
            istart = time.time()
            print(f"Starting sample no {i}")
            for pname, pfn in self._problems.items():
                for aname, afn in self._algos.items():
                    print(f"Running {aname} for problem {pname}")
                    astart = time.time()
                    solution, fit = afn(pfn)
                    results.append([i, aname, pname, solution, fit])
                    print(f"Algo {aname} ended after {(time.time() - astart)/60} minutes")
            self._save_results(results)
            iend = time.time()
            print(f"Ended sample no {i} after {(iend - istart)/60} minutes")
        gend = time.time()
        print(f"Ended experiment no {i} after {(gend - gstart)/60} minutes")
    def _save_results(self, results: List[List[Any]]):
        with open(self._filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(results)

    def _init_csv(self):
        if not os.path.exists(self._filename):
            with open(self._filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self._headers)
            self._num_experiments = 0
        else:
            self._restore_state_from_csv()

    def _restore_state_from_csv(self):
        with open(self._filename, mode="r") as file:
            reader = csv.reader(file)
            rows = list(reader)
            self._num_experiments = 0 if len(rows) < 2 else int(rows[-1][0])
