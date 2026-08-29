from typing import Callable, Optional, Dict, List, Any
from dataclasses import dataclass
import os, csv, time
import pandas as pd
from pandas.errors import EmptyDataError
from lpsge.auto_de import ProblemInstance, search_optimizer, run_auto_de
from core.stop_condition import StopCondition

@dataclass
class ExperimentRunResult:
    data: List[Dict[str, Any]]
    done: bool = False

class ExperimentRunnable:
    def __init__(
        self,
        filename: str,
        experiment_callback: Callable[[], ExperimentRunResult], 
        name: Optional[str] = None
    ): 
        self._filename = filename
        self._name = name if name is not None else filename
        self._experiment_callback = experiment_callback
        self._headers = None
        self._mode = "a"

    def _df(self):
       #try:
        if not self._is_file_empty():
            return pd.read_csv(self._filename)
        #except EmptyDataError:
        #    pass
       

    def count_by(self, callback: Callable[[Optional[pd.DataFrame]], int]):
        df = self._df()
        if df is None:
            return 0
        return callback(df)
    
    def get_row(self, selector: Callable[[pd.DataFrame], pd.Series]) -> Optional[pd.Series]:
        try:
            df = self._df()
            if df is not None:
                return selector(df)
        except IndexError:
            pass
        return None
    
    def find_any(self, callback: Callable[[Optional[pd.DataFrame]], Any]):
        df = self._df()
        if df is None:
            return 0
        return callback(df)
       

    def _do_run(self, num_exp = 1):
        #istart = time.time()
        res:ExperimentRunResult = self._experiment_callback()
        if res.data is not None:
            #print(f"Starting sample no {num_exp} for experiment {self._name}")
            self._save_results(res.data)
            #iend = time.time()
            #print(f"Ended sample no {num_exp} after {(iend - istart)/60} minutes")
        if not res.done:
            self._do_run(num_exp=num_exp+1)
        

    def run(self):
        self._init_csv()
        gstart = time.time()
        print(f"Starting experiment {self._name}")
        self._do_run()
        gend = time.time()
        print(f"Ended experiment after {(gend - gstart)/60} minutes")
    
    def _save_results(self, data: List[Dict[str, Any]]):
        if len(data) == 0:
            return
        with open(self._filename, mode=self._mode, newline="") as file:
            writer = csv.writer(file)
            if self._headers is None:
                self._headers = list(data[0].keys()) 
                if self._mode == "w":
                    writer.writerow(self._headers)
                    self._mode = "a"
            writer.writerows([list(item.values()) for item in data]) 

    def _init_csv(self):
        if self._is_file_empty():
            self._mode = "w" 
        else:
            self._restore_save_point()

    def _is_file_empty(self):
        return not os.path.exists(self._filename) or os.path.getsize(self._filename) == 0

    def _restore_save_point(self):
        rows = None
        with open(self._filename, mode="r") as file:
            reader = csv.reader(file)
            rows = list(reader)
        if rows is None or len(rows) < 2:
                return 
        self._headers = rows[0]

@dataclass
class ExperimentState:
    run_cnt: int = 0
    traning_set: Optional[str] = None
    problems: Optional[List[ProblemInstance]] = None 
    test_algo: Optional[str] = None 
    train_experiment: Optional[ExperimentRunnable] = None
    test_experiment: Optional[ExperimentRunnable] = None

class Experiment:
    def __init__(
        self,
        name: str,
        problems: Dict[str, List[ProblemInstance]],
        parameters_file: str, 
        popsize: int, 
        max_experiments: int = 51,
        pmin: int = 2,
        elitesize: int = 2,
        rmethod:str ="kruns", 
        nruns: int = 5
    ):
        self._name = name
        self._max_experiments = max_experiments
        self._problems = problems
        self._parameters_file = parameters_file
        self._popsize = popsize
        self._pmin = pmin
        self._elitesize = elitesize
        self._nruns = nruns
        self._rmethod = rmethod

    def run(self):
        state = ExperimentState()
        for name, problems in self._problems.items():
            state.traning_set = name
            state.problems = problems
            self._train(state)
            self._test(state)

        
    def _train(self, state:ExperimentState):
        def experiment_callback()->ExperimentRunResult:
            res = search_optimizer(
                problems=state.problems,
                parameters_file=self._parameters_file, 
                popsize=self._popsize,
                nruns=self._nruns,
                elitesize=self._elitesize
            )
            return ExperimentRunResult(data=[{
                "problem": state.traning_set,
                "genotype": res["genotype"],
                "phenotype": res["phenotype"],
                "error": res["fitness"],
                "runtime(s)": res["runtime"],
                "runtime(g)": res["num_g"],
                "runtime(fes)": res["num_evals"],
                "pinit": self._popsize,
            }], done=True)
        if  state.train_experiment is None:
            state.train_experiment = ExperimentRunnable(
                filename=f"{self._name}_train.csv",
                experiment_callback=experiment_callback)
        algo_cnt = state.train_experiment.count_by(lambda df: df["problem"].eq(state.traning_set).sum())
        if algo_cnt == 0:
            print(f"[{self._name}] Traning for {state.traning_set}")
            state.train_experiment.run()
        algo_cnt = state.train_experiment.count_by(lambda df: df["problem"].eq(state.traning_set).sum())
        algo = state.train_experiment.get_row(lambda df: df.loc[df["problem"] == state.traning_set, "phenotype"].iloc[0])
        if algo is None:
            raise ValueError(f"At this point algo cannot be None {algo_cnt}")
        state.test_algo = algo

    def _test(self, state:ExperimentState):
        def experiment_callback()->ExperimentRunResult:
            if state.run_cnt/len(state.problems) >= self._max_experiments:
                print(f"Reached max experiments threshold runs of {self._max_experiments} for {state.traning_set} with num runs = {state.run_cnt/len(state.problems)}")
                return ExperimentRunResult(data=None, done=True)
            data = []
            for problem in state.problems:
                problem_name = state.traning_set if problem.name is None else problem.name
                solution, fitness = run_auto_de(state.test_algo, problem)
                data.append({
                "algo": "AUTO",
                "problem_set": state.traning_set, 
                "problem": problem_name,
                "ndims": problem.ndims,
                "solution": solution,
                "fit": fitness
                })
                state.run_cnt += 1
            return ExperimentRunResult(data=data, done=False)

        if state.test_experiment is None:
            state.test_experiment = ExperimentRunnable(
                filename=f"{self._name}_test.csv",
                experiment_callback=experiment_callback
            )
        state.run_cnt = state.test_experiment.count_by(lambda df:  df["problem_set"].eq(state.traning_set).sum())
        print(f"[{self._name}] Testing for {state.traning_set}")
        state.test_experiment.run()

