from __future__ import annotations
from core import (
    create_optimizer,
    UpdateMethod,
    Params,
    ParamGenerator,
    ensemble,
    PubSub,
)
from typing import List
import operators as ops
from functools import partial
import numpy as np
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional
from core.stop_condition import StopCondition
from lpsge.psgewr import PSGEWR, EvalFn

def _build_update_strategy(picker, mutation, crossover, reparation, convergence_booster):
    return UpdateMethod(
        picker=picker,
        mutation=mutation,
        crossover=crossover,
        reparation=reparation,
        selection=ops.selection.elitist(),
        convergence_booster=convergence_booster
    )


def _build_param_generator(init_val, get_strategy, repair_strategy, learning_strategy):
    return lambda id: ParamGenerator(
        id=id,
        initial_value=init_val,
        get_strategy=get_strategy,
        repair_strategy=repair_strategy,
        learning_strategy=learning_strategy,
    )

class _Optimizer: 
    def __init__(self, events, stop_condition, init_strategy: str = "random", s_lambda = .5, seed=None):
        self._events = events
        self._stop_condition = stop_condition
        self._params = None
        self._init_strategy = init_strategy
        self._s_lambda = s_lambda
        self._seed = seed

    def __call__(self, resize_strategy, update_strategy, archive, popsize, params, init_strategy: str = None, s_lambda = None, seed=None):
        if init_strategy is not None:
            self._init_strategy = init_strategy
        if s_lambda is not None: 
            self._s_lambda = s_lambda
        if seed is not None:
            self._seed = seed
        optimizer = create_optimizer(
            init_strategy=self._get_init_strategy(popsize),
            resize_strategy=resize_strategy,
            update_strategy=[update_strategy],
            archive=archive,
            events=self._events,
            params=params,
            stop_condition=self._stop_condition
        )
        def callback(fit_fn, lb, ub, ndims):
            res = optimizer(fit_fn, lb, ub, ndims)
            del self._events
            del self._params
            return res
        return callback
    
    def _get_init_strategy(self, popsize):
        if self._init_strategy == "opposite":
            return ops.population.init.opposite(popsize, seed=self._seed)
        if self._init_strategy == "semi-opposite":
            return ops.population.init.semi_opposite(popsize, seed=self._seed, s_lambda=self._s_lambda)
        return ops.population.init.random(popsize, seed=self._seed)


class _Picker:
    def __init__(self, events):
        self.default = ops.picker.default
        self.fode = partial(ops.picker.fode, events=events)


class _Mutation:
    def __init__(self, events):
        self.rand_1 = partial(ops.mutation.rand_y, diff_size=1)
        self.rand_2 = partial(ops.mutation.rand_y, diff_size=2)
        self.best_1 = partial(ops.mutation.best_y, diff_size=1)
        self.best_2 = partial(ops.mutation.best_y, diff_size=2)
        self.rand_to_best_1 = partial(ops.mutation.rand_to_best_y, diff_size=1)
        self.rand_to_best_2 = partial(ops.mutation.rand_to_best_y, diff_size=2)
        self.current_to_rand_1 = partial(ops.mutation.current_to_rand_y, diff_size=1)
        self.current_to_best_1 = partial(ops.mutation.current_to_best_y, diff_size=1)
        self.current_to_pbest_1 = ops.mutation.current_to_pbest
        self.current_to_pbestw_1 = ops.mutation.current_to_pbestw
        self.current_to_ord_best_1 = ops.mutation.current_to_ord_best
        self.current_to_ord_pbest_1 = ops.mutation.current_to_ord_pbest
        self.cma = partial(ops.mutation.cma, events=events)


class _Croosover:
    def __init__(self, events):
        self.binomial = ops.crossover.binomial
        self.binomial_with_cauchy_perturbation = (
            ops.crossover.binomial_with_cauchy_perturbation
        )
        self.exponential = ops.crossover.exponential
        self.multiple_exponential = ops.crossover.multiple_exponential
        self.shuffled_exponential = ops.crossover.shuffled_exponential
        self.all = ops.crossover.all


class _Reparation:
    def __init__(self, events):
        self.middle_from_parent = ops.reparation.middle_from_parent
        self.clip = ops.reparation.clip


class _Linear:
    def __init__(self, events):
        self.eval = partial(ops.params.eval.linear, events=events)

class _Random:
    def __init__(self, events):
        self.uniform = ops.params.random.uniform
        self.boolean = ops.params.random.boolean
        self.range = partial(ops.params.random.range, events=events)


def get_locals(stop_condition, ndims, _events = None, init_strategy: str = "random", seed=None):
    events = _events if _events is not None else PubSub()
    return {
        "np": np,
        "Ignore": None,
        "ndims": ndims,
        "ensemble": ensemble,
        "Params": lambda m: Params(memory=m,events=events),
        "popsize": None,
        "lpsr": ops.population.lpsr,
        "archive": ops.archive.fixed,
        "picker": _Picker(events=events),
        "mutation": _Mutation(events=events),
        "crossover": _Croosover(events=events),
        "reparation": _Reparation(events=events),
        "linear": _Linear(events=events),
        "random": _Random(events=events),
        "update_strategy": _build_update_strategy,
        "param_strategy": _build_param_generator,
        "create_optimizer": _Optimizer(events=events, stop_condition=stop_condition, init_strategy=init_strategy, seed=seed),
        "jade": ops.params.jade,
        "shade": ops.params.shade,
        "ilshade": ops.params.ilshade,
        "jso": ops.params.jso,
        "fode": ops.params.fode,
        "lshade": ops.params.lshade,
        "dbshade": ops.params.dbshade,
        "nshade": ops.params.nshade,
        "lshadespa": ops.params.lshadespa,
        "booster": ops.booster,
    }

def auto_de_eval(individual: str, problem: ProblemInstance, invalid_fitness: float=float("inf"))->float:
    fit = None
    #start_time = time.time()
    locals = get_locals(stop_condition=problem.stop_condition, ndims=problem.ndims)
    try:
        parts = individual.split("\\n")
        for part in parts:
            cpart = compile(part, f'algo{uuid.uuid4()}', "exec", dont_inherit=True)
            exec(cpart, locals ) #
        fit = locals["optimizer"](problem.problem, problem.lb, problem.ub, problem.ndims)[1]
    except Exception as e:
        print(f"Failed: {repr(e)}")
        print(individual)
        fit = invalid_fitness
    #print("algo run took %s seconds" % (time.time() - start_time))
    return fit

def auto_de_evals(individual: str, problems: List[ProblemInstance])->List[float]:
    return [auto_de_eval(individual, problem) for problem in problems]

@dataclass
class ProblemInstance:
   problem: Callable[[List[float]], float]
   ndims: int
   lb: float
   ub: float
   global_optimum: float
   stop_condition: StopCondition
   name: Optional[str] = None

def search_optimizer(
    problems: List[ProblemInstance],   
    parameters_file, 
    popsize: int, 
    pmin: int = 2,
    elitesize: int = 2,
    rmethod:str ="kruns", 
    nruns: int = 5
):
    start_time = time.time()
    print(f"Starting search at {start_time}")
    res = PSGEWR(
        evaluation_function=EvalFn(
            fn=partial(auto_de_evals, problems=problems),
            optimal=[problem.global_optimum for problem in problems]
        ),
        parameters_file=parameters_file,
        popsize=popsize, 
        pmin=pmin,
        elitesize=elitesize,
        rmethod=rmethod, 
        nruns=nruns
    ).run()
    res["runtime"] = time.time() - start_time
    return res


def run_auto_de(individual: str, problem: ProblemInstance, invalid_fitness: float=float("inf"))->float:
    locals = get_locals(stop_condition=problem.stop_condition, ndims=problem.ndims)
    parts = individual.split("\\n")
    for part in parts:
        cpart = compile(part, f'algo{uuid.uuid4()}', "exec", dont_inherit=True)
        exec(cpart, locals ) #
    return locals["optimizer"](problem.problem, problem.lb, problem.ub, problem.ndims)
