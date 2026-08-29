import sys
from functools import partial
from experiment import cec2017, Experiment, Benchmark
import operators as ops
from core import (
    create_optimizer,
    UpdateMethod,
    Params,
    ensemble,
    PubSub
)
import operators as ops
from experiment_autode import CEC_PROBLEMS, cec_instance

def lshade_spacma(problem: Benchmark, ndims=None, stop_condition=None, popsize=None, f=.5, cr=.5, h=5, p=.11, r_arc=1.4, p_ensemble=.5, learning_rate=.8):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    fit_fn = problem.problem
    lb=problem.lb
    ub=problem.ub
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.either(
                    ops.mutation.current_to_pbest(
                        f=params.adaptative(ops.params.lshadespa.cauchy(initial_value=f)),
                        p=p, 
                        include_archive=True
                    ),
                    ops.mutation.cma(events=events), 
                    params.adaptative(ops.params.random.boolean(low=.2,high=.8, p=p_ensemble, weighted=True, learning_rate=learning_rate))
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.lshade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

def nlshade(problem:Benchmark, ndims=None, popsize=None, stop_condition=None, f=.5, cr=.5, r_arc=2.6):
    events = PubSub()
    params = Params(events=events, memory=ops.params.nshade.memory())
    fit_fn = problem.problem
    lb=problem.lb
    ub=problem.ub
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.nshade.cauchy(initial_value=f)),
                    p=params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                    include_archive=True
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.nshade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

def ebde(problem:Benchmark, ndims=None, popsize=None, stop_condition=None, h=None, f=.5, cr=.5, p=.11): #, include_archive=False
    events = PubSub()
    fit_fn = problem.problem
    lb=problem.lb
    ub=problem.ub
    params = Params(events=events, memory=ops.params.shade.memory(popsize if h is None else h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random(popsize),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_ord_pbest(
                    params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                    #p=p,
                    p=params.adaptative(ops.params.random.uniform(low=p, high=0.2)),
                    include_archive=False
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        #archive=ops.archive.fixed(round(popsize)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)  


def get_stop_condition(ndims):
    return ops.stop_condition.max_evaluations(max_evals=ndims*10000)

def get_problems(fnames: str, ndims: int):
    fns =  CEC_PROBLEMS.keys() if not fnames else fnames.split(",")
    problems = {}
    for fn in fns:
        fname = fn.strip()
        if fname not in CEC_PROBLEMS:
            raise Exception(f"ERROR: Benchmark function {fname} does not exist")
        problem = CEC_PROBLEMS[fname]
        problems[fname] = cec_instance(problem, ndims) 
    return problems


def get_algos(ndims: int): 
    return {
        "LSHADE_SPACMA": partial(
            lshade_spacma,
            ndims=ndims,
            popsize=18*ndims,
            f=.5,
            cr=.5,
            p=.11,
            h=5,
            r_arc=1.4,
            p_ensemble=.5, 
            learning_rate=.8,
            stop_condition=get_stop_condition(ndims)
        ),
        "NLSHADE": partial(
            nlshade,
            ndims=ndims,
            popsize=18*ndims, 
            f=.5, 
            cr=.5, 
            r_arc=2.6, 
            stop_condition=get_stop_condition(ndims)
        ), 
        "EBDE": partial(
            ebde, 
            ndims=ndims,
            popsize=100, 
            f=.5, 
            cr=.5, 
            h=100, 
            p=2/100, 
            stop_condition=ops.stop_condition.max_evaluations(max_evals=ndims*10000)
        ) 
    }

if __name__ == "__main__":
    argsize = len(sys.argv)
    if argsize < 3:
        print(
            f"ERROR: Benchmark function not provided"
            if argsize < 2 else
            f"ERROR: Number of dimensions not provided"
        )
        sys.exit(1)
    fnames = sys.argv[1] 
    ndims = int(sys.argv[2])
    fnames_part = fnames.replace(",", "_")
    expname = "experiment_competitors" if argsize < 4 else sys.argv[3] 
    print(f"Running experiment {expname} on {fnames} for {ndims} dimensions")
    Experiment(
        filename=f"{expname}_{fnames_part}_{ndims}d.csv",
        algos=get_algos(ndims),
        problems=get_problems(fnames, ndims),
        max_experiments=51,
    ).run() 
    sys.exit(0)
        
