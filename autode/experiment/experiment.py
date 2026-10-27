from typing import Callable, Tuple, Dict, List, Any
import os, csv, time
import tempfile
from .cec2017_problems import Benchmark
from .seeds import load_seed_schedule, save_external_seed_schedule


class Experiment:
    def __init__(
        self,
        filename: str,
        algos: Dict[str, Callable[..., Tuple[List[float], float]]],
        problems: Dict[str, Benchmark],
        max_experiments: int = 51,
        save_point: int = 1,
        seed_schedule: Dict[str, List[int]] = None,
        resume_by_pair: bool = False,
    ):
        self._filename = filename
        self._algos = algos
        self._problems = problems
        self._max_experiments = max_experiments
        self._save_point = save_point
        self._headers = ["no", "algo", "problem", "solution", "fit", "seed"]
        self._num_experiments = 0
        self._seeds: Dict[str, List[int]] = {}
        self._seed_schedule = seed_schedule
        self._resume_by_pair = resume_by_pair
        self._completed = set()

    def run(self):
        self._init_seeds()
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
                seed = self._seeds[pname][i - 1]
                for aname, afn in self._algos.items():
                    if self._resume_by_pair and (i, aname, pname) in self._completed:
                        continue
                    print(f"Running {aname} for problem {pname}")
                    astart = time.time()
                    solution, fit = afn(pfn, seed=seed)
                    results.append([i, aname, pname, solution, fit, seed])
                    print(f"Algo {aname} ended after {(time.time() - astart)/60} minutes")
            self._save_results(results)
            iend = time.time()
            print(f"Ended sample no {i} after {(iend - istart)/60} minutes")
        gend = time.time()
        print(f"Ended experiment no {i} after {(gend - gstart)/60} minutes")

    def _init_seeds(self):
        if self._seed_schedule is not None:
            self._seeds = save_external_seed_schedule(
                self._filename, self._seed_schedule, self._problems, self._max_experiments
            )
            return
        self._seeds = load_seed_schedule(
            self._filename, self._problems, self._max_experiments
        )

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
        if rows and rows[0] == self._headers[:-1]:
            # Historical runs did not receive these seeds. Leave their seed
            # cells empty rather than attributing the new schedule to them.
            rows = [self._headers] + [row + [""] for row in rows[1:]]
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", newline="", dir=os.path.dirname(os.path.abspath(self._filename)),
                    delete=False,
                ) as file:
                    temporary_path = file.name
                    csv.writer(file).writerows(rows)
                os.replace(temporary_path, self._filename)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        elif rows and rows[0] != self._headers:
            raise ValueError(f"Unexpected CSV header in {self._filename}")
        if self._resume_by_pair:
            self._completed = {(int(row[0]), row[1], row[2]) for row in rows[1:]}
            self._num_experiments = 0
            pairs = [(aname, pname) for aname in self._algos for pname in self._problems]
            if pairs:
                while self._num_experiments < self._max_experiments and all(
                    (self._num_experiments + 1, aname, pname) in self._completed
                    for aname, pname in pairs
                ):
                    self._num_experiments += 1
        else:
            self._num_experiments = 0 if len(rows) < 2 else int(rows[-1][0])
