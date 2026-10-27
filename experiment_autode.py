from typing import List, Tuple

import os, sys
from autode.lpsge import Experiment, ExperimentProblems, ProblemInstance
from autode.core.stop_condition import StopCondition
from cec2017.functions import *

CEC_PROBLEMS = {
    "F1": (f1, 100.0),
    "F3": (f3, 300.0),
    "F4": (f4, 400.0),
    "F5": (f5, 500.0),
    "F6": (f6, 600.0),
    "F7": (f7, 700.0),
    "F8": (f8, 800.0),
    "F9": (f9, 900.0),
    "F10": (f10, 1000.0),
    "F11": (f11, 1100.0),
    "F12": (f12, 1200.0),
    "F13": (f13, 1300.0),
    "F14": (f14, 1400.0),
    "F15": (f15, 1500.0),
    "F16": (f16, 1600.0),
    "F17": (f17, 1700.0),
    "F18": (f18, 1800.0),
    "F19": (f19, 1900.0),
    "F20": (f20, 2000.0),
    "F21": (f21, 2100.0),
    "F22": (f22, 2200.0),
    "F23": (f23, 2300.0),
    "F24": (f24, 2400.0),
    "F25": (f25, 2500.0),
    "F26": (f26, 2600.0),
    "F27": (f27, 2700.0),
    "F28": (f28, 2800.0),
    "F29": (f29, 2900.0),
    "F30": (f30, 3000.0),
}

def cec_instance(problem: Tuple[callable, float], ndims: int, name: str)->ProblemInstance:
    fn,bias = problem
    return ProblemInstance(
        problem=fn,
        ndims=ndims, 
        name=name,
        stop_condition=StopCondition(
            max_evals=ndims*10000, 
            callback=lambda ctx: (ctx.best()[1] - bias) <= 0
        ),
        lb=-100, 
        ub=100,
        global_optimum=bias
    )

def get_problems(fnames: str, train_ndims: List[int], test_ndims: List[int]):
    fns =  CEC_PROBLEMS.keys() if not fnames else fnames.split(",")
    problems = {}
    for fn in fns:
        fname = fn.strip()
        if fname not in CEC_PROBLEMS:
            raise Exception(f"ERROR: Benchmark function {fname} does not exist")
        problem = CEC_PROBLEMS[fname]
        problems[fname] = ExperimentProblems(
            train=[cec_instance(problem, dim, fname) for dim in train_ndims],
            test=[cec_instance(problem, dim, fname) for dim in test_ndims],
        )
    return problems

__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
)

train_ndims = [10, 30]
test_ndims = [30, 50]
test_seed_files = {
    dim: os.path.join(__location__, f"experiment_competitors_all_{dim}d.json")
    for dim in test_ndims
}
parameters_file = os.path.join(__location__, "autode", "standard.yml")
nruns = 5
popsize = 20
elitesize = 2
pmin = 4


if __name__ == "__main__":
    argsize = len(sys.argv)
    if argsize < 2:
        print("ERROR: Benchmark functions not provided")
        sys.exit(1)
    print(sys.argv)
    fnames = sys.argv[1]

    print(f"Running experiment autode psgewr on {fnames}")
    fnames_part = fnames.replace(",", "_")

    Experiment(
        name=f"experiment_autode_{fnames_part}", 
        problems=get_problems(fnames if fnames != "all" else None, train_ndims, test_ndims),
        parameters_file=parameters_file, 
        popsize=popsize, 
        max_experiments=51,
        pmin = pmin,
        elitesize = elitesize,
        rmethod ="kruns", 
        nruns = nruns,
        test_seed_file=test_seed_files,
    ).run()
    sys.exit(0)