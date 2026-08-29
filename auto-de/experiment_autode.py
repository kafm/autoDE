import operators as ops
import core 
from typing import Callable, List
from opfunu.cec_based.cec2014 import *
from opfunu.cec_based.cec2017 import *
import os, sys
from lpsge import Experiment,ProblemInstance
from core.stop_condition import StopCondition
from opfunu.cec_based.cec2014 import *
from opfunu.cec_based.cec2017 import *

CEC_PROBLEMS = {
    "F1": F12017,
    "F2": F22017,
    "F3": F32017,
    "F4": F42017,
    "F5": F52017,
    "F6": F62017,
    "F7": F72017,
    "F8": F82017,
    "F9": F92017,
    "F10": F102017,
    "F11": F112017,
    "F12": F122017,
    "F13": F132017,
    "F14": F142017,
    "F15": F152017,
    "F16": F162017,
    "F17": F172017,
    "F18": F182017,
    "F19": F192017,
    "F20": F202017,
    "F21": F212017,
    "F22": F222017,
    "F23": F232017,
    "F24": F242017,
    "F25": F252017,
    "F26": F272017,
    "F27": F272017,
    "F28": F282017,
    "F29": F292017
}

def cec_instance(problem: Callable[[int], CecBenchmark], ndims: int)->ProblemInstance:
    cec = problem(ndims)
    return ProblemInstance(
        problem=cec.evaluate,
        ndims=ndims, 
        stop_condition=core.StopCondition(
            max_evals=ndims*10000, 
            callback=lambda ctx: (ctx.best()[1] - cec.f_bias) <= 0
        ),
        lb=-100, 
        ub=100,
        global_optimum=cec.f_bias
    )

def get_problems(fnames: str, ndims: List[int]):
    fns =  CEC_PROBLEMS.keys() if not fnames else fnames.split(",")
    problems = {}
    for fn in fns:
        fname = fn.strip()
        if fname not in CEC_PROBLEMS:
            raise Exception(f"ERROR: Benchmark function {fname} does not exist")
        problem = CEC_PROBLEMS[fname]
        problems[fname] = [cec_instance(problem, dim) for dim in ndims]
    return problems

__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
)

ndims = [30, 50]
parameters_file = os.path.join(__location__,"standard.yml")
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
        problems=get_problems(fnames, ndims),
        parameters_file=parameters_file, 
        popsize=popsize, 
        max_experiments=1,
        pmin = pmin,
        elitesize = elitesize,
        rmethod ="kruns", 
        nruns = nruns,
    ).run()
    sys.exit(0)



