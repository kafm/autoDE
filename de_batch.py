from typing import List
from autode.core import (
    create_optimizer,
    Params,
    ensemble,
    StopCondition,
    PubSub
)
from autode.core.update_method import BatchUpdateMethod as UpdateMethod
import autode.operators as ops
from opfunu.cec_based.cec2014 import *
import numpy as np
from opfunu.cec_based.cec2020 import *
from opfunu.cec_based.cec2017 import *
from opfunu.cec_based.cec2015 import *
from cec2017.functions import f27, f3
import time

def de_canonical(fit_fn, lb, ub, ndims, popsize, stop_condition, f=0.5, cr=0.6):
    events = PubSub()
    params = Params(events=events)
    search = create_optimizer(
        init_strategy=ops.population.init.random_batch(popsize),
        events=events,
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.rand_y(params.static(f)),
                crossover=ops.crossover.binomial(params.static(cr)),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)


def jade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, p=.1, c=.1, include_archive=False):
    events = PubSub()
    params = Params(events=events)
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.jade.cauchy(initial_value=f, learning_rate=c)),
                    p=p,
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jade.normal(initial_value=cr, learning_rate=c))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(popsize),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)


def shade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, c=.1, include_archive=False):
    events = PubSub()
    params = Params(events=events, memory=ops.params.shade.memory(popsize))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.shade.cauchy(initial_value=f)),
                    p=params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.shade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(popsize),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

def lshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=6, p=.11, r_arc=2.6, include_archive=True):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.lshade.cauchy(initial_value=f)),
                    p=p,#params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                    include_archive=include_archive
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

def ilshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=6, p_min=.1, p_max=.2, r_arc=2.6, include_archive=False):
    events = PubSub()
    params = Params(events=events, memory=ops.params.ilshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.ilshade.cauchy(initial_value=f)),
                    p=params.adaptative(ops.params.eval.linear(events=events,low=p_min, high=p_max)),
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.ilshade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

def jso(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=6, p_min=.1, p_max=.2, r_arc=2.6, include_archive=False):
    events = PubSub()
    params = Params(events=events, memory=ops.params.ilshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbestw(
                    f=params.adaptative(ops.params.jso.cauchy(initial_value=f)),
                    p=params.adaptative(ops.params.eval.linear(events=events,low=p_min, high=p_max)),
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jso.normal(initial_value=cr))
                ),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

#dish variant
def dbshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=6, p_min=.1, p_max=.2, r_arc=2.6, include_archive=False):
    events = PubSub()
    params = Params(events=events, memory=ops.params.ilshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_pbestw(
                    f=params.adaptative(ops.params.dbshade.cauchy(initial_value=f)),
                    p=params.adaptative(ops.params.eval.linear(events=events,low=p_min, high=p_max)),
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jso.normal(initial_value=cr))
                ),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

#lshade variant
def nshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, r_arc=2.6):
    events = PubSub()
    params = Params(events=events, memory=ops.params.nshade.memory())
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
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

def ede(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, include_archive=False):
    events = PubSub()
    params = Params(events=events, memory=ops.params.shade.memory(popsize))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_ord_best_y(
                    params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                    include_archive=include_archive
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(popsize),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)    

def ebde(fit_fn, lb, ub, ndims, popsize, stop_condition, h=None, f=.5, cr=.5, p=.11): #, include_archive=False
    events = PubSub()
    params = Params(events=events, memory=ops.params.shade.memory(popsize if h is None else h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.current_to_ord_pbest(
                    params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                    p=p,
                    #p=params.adaptative(ops.params.random.uniform(low=p, high=0.2)),
                    include_archive=True
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.jade.normal(initial_value=cr))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)    

def ebede(fit_fn, lb, ub, ndims, stop_condition, popsize=None, h=5, pmin=4, f=.5, cr=.5, p=.11, p_ede=.1, r_arc=1.4, include_archive=True): #, include_archive=False
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    if popsize is None:
        popsize = 18*ndims
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=pmin),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.either(
                    ops.mutation.current_to_pbest(
                        f=params.adaptative(ops.params.lshade.cauchy(initial_value=f)),
                        p=p,
                        include_archive=include_archive
                    ),
                    ops.mutation.current_to_ord_pbest(
                        params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                        p=p_ede,
                        include_archive=include_archive
                    ), 
                    params.adaptative(ops.params.random.boolean(low=.2,high=.8, weighted=True, learning_rate=.8))
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

def elshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=5, p=.11, r_arc=2.6, include_archive=True):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.either(
                    ops.mutation.current_to_pbest(
                        f=params.adaptative(ops.params.lshade.cauchy(initial_value=f)),
                        p=p,
                        include_archive=include_archive
                    ),
                    ops.mutation.current_to_ord_best_y(
                        params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                        include_archive=include_archive
                    ), 
                    params.adaptative(ops.params.random.boolean(low=.2,high=.8, weighted=True, learning_rate=.1))
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

def eblshade(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=5, p=.11, r_arc=2.6, include_archive=True):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.either(
                    ops.mutation.current_to_pbest(
                        f=params.adaptative(ops.params.lshade.cauchy(initial_value=f)),
                        p=p,
                        include_archive=include_archive
                    ),
                    ops.mutation.current_to_ord_pbest(
                    params.adaptative(ops.params.jade.cauchy(initial_value=f)),
                    p=p,
                    include_archive=include_archive
                    ), 
                    params.adaptative(ops.params.random.boolean(low=.2,high=.8, weighted=True, learning_rate=.1))
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

def cma(fit_fn, lb, ub, ndims, popsize, stop_condition):
    events = PubSub()
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ops.mutation.cma(events=events),
                reparation=ops.reparation.clip(),
                selection=ops.selection.elitist(),
            )
        ],
        stop_condition=stop_condition
    )
    return search(fit_fn, lb, ub, ndims)

def lshade_spacma(fit_fn, lb, ub, ndims, stop_condition, popsize=None,pmin=4, f=.5, cr=.5, h=5, p=.11, r_arc=1.4, p_ensemble=.5, learning_rate=.8):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    if popsize is None: 
        popsize = 18*ndims
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=pmin),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.either(
                    ops.mutation.current_to_pbest(
                        f=params.adaptative(ops.params.lshadespa.cauchy(initial_value=f)),
                        #f=params.adaptative(ops.params.lshadespa.cauchy_bck(initial_value=f)),
                        #p=params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                        p=p, #params.adaptative(ops.params.random.uniform(low=0.1459, high=0.4765)),
                        include_archive=True
                    ),
                    ops.mutation.cma(events=events), 
                    params.adaptative(ops.params.random.boolean(low=.2,high=.8, p=p_ensemble, weighted=True, learning_rate=learning_rate))
                ),
                crossover=ops.crossover.binomial(
                    params.adaptative(ops.params.lshadespa.normal(initial_value=cr))
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

def fode(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.8, cr=.5, h=5, p=.11, r_arc=1.4, include_archive=True):
    events = PubSub()
    params = Params(events=events, memory=ops.params.ilshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                picker=ops.picker.fode(events=events),
                mutation=ops.mutation.current_to_pbest(
                    f=params.adaptative(ops.params.fode.cauchy(initial_value=f)),
                    p=p,
                    include_archive=True
                ),
                crossover=ops.crossover.binomial_with_cauchy_perturbation(
                    params.adaptative(ops.params.fode.normal(initial_value=cr))
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

def imode(fit_fn, lb, ub, ndims, popsize, stop_condition, f=.5, cr=.5, h=5, p=.11, r_arc=2.6):
    events = PubSub()
    params = Params(events=events, memory=ops.params.lshade.memory(h))
    search = create_optimizer(
        events=events,
        init_strategy=ops.population.init.random_batch(popsize),
        resize_strategy=ops.population.lpsr(pmin=4),
        update_strategy=[
            UpdateMethod(
                mutation=ensemble.choice(
                    [
                        ops.mutation.current_to_pbest(
                            f=params.adaptative(ops.params.lshadespa.cauchy(initial_value=f)),
                            p=p,#params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                            include_archive=True
                        ),  
                        ops.mutation.current_to_pbest(
                            f=params.adaptative(ops.params.lshadespa.cauchy(initial_value=f)),
                            p=p,#params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                            include_archive=False
                        ),   
                        ops.mutation.rand_to_pbest(
                            f=params.adaptative(ops.params.lshadespa.cauchy(initial_value=f)),
                            p=p,#params.adaptative(ops.params.random.uniform(low=2/popsize, high=0.2)),
                            include_archive=False
                        ),  
                    ],
                    params.adaptative(ops.params.random.range(3,events,pmin=.1,pmax=.9))
                    #params.adaptative(ops.params.random.boolean(low=.2,high=.8, weighted=True, learning_rate=.8))
                ),
                # crossover=ops.crossover.binomial(
                #          params.adaptative(ops.params.lshade.normal(initial_value=cr))
                #      ),
                crossover=ensemble.either(
                    ops.crossover.binomial(
                        params.adaptative(ops.params.lshade.normal(initial_value=cr))
                    ),
                    ops.crossover.exponential(
                        params.adaptative(ops.params.lshade.normal(initial_value=cr))
                    ),
                    params.adaptative(ops.params.random.boolean(p=.4, learning_rate=0))
                ),
                reparation=ops.reparation.middle_from_parent(),
                selection=ops.selection.elitist(),
                convergence_booster=ops.booster.imode()
            )
        ],
        stop_condition=stop_condition,
        archive=ops.archive.fixed(round(popsize*r_arc)),
        params=params,
    )
    return search(fit_fn, lb, ub, ndims)

def madde():
    pass

def sphere(_=None) -> float:
    return lambda x: np.sum(np.square(x)), -100, 100 #5.12


def schwefel(_=None):
    def callback(x):
        n = len(x)
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    return callback, -500, 500


def f12014(ndims):
    return F12014(ndims).evaluate, -100, 100

def f22014(ndims):
    return F22014(ndims).evaluate, -100, 100

def f172014(ndims):
    return F172014(ndims).evaluate, -100, 100

def f232014(ndims):
    return F232014(ndims).evaluate, -100, 100

def f292014(ndims):
    return F292014(ndims).evaluate, -100, 100

def f52015(ndims):
    return F52015(ndims).evaluate, -100, 100

def f12020(ndims):
    return F12020(ndims).evaluate, -100, 100

def f12017(ndims):
    return F12017(ndims).evaluate, -100, 100

def f52017(ndims):
    return F52017(ndims).evaluate, -100, 100

def f42017(ndims):
    return F42017(ndims).evaluate, -100, 100

def f62017(ndims):
    return F62017(ndims).evaluate, -100, 100

def f292017(ndims):
    return F292017(ndims).evaluate, -100, 100

def f272017(ndims):
    return F272017(ndims).evaluate, -100, 100


ndims = 50
popsize = ndims*18 # 6*ndims*ndims #imode 6*ndims**2 100 shade #ndims*18 #ndims*12  #ilshade=ndims*12 #lshade=round(ndims*18)   #jso=int(25*np.log(ndims)*np.sqrt(ndims)) #40
max_gen = 1500
max_evals = ndims*10000
#max_gen = 1500
#problem, lb, ub = f272017(ndims)
problem, lb, ub = f27, -100, 100
bias = 2700

start_time = time.time()

# solution, fit = nshade(
#     fit_fn=problem,
#     lb=lb,
#     ub=ub,
#     ndims=ndims,
#     popsize=18*ndims, 
#     f=.5, 
#     cr=.5, 
#     r_arc=2.6, 
#     stop_condition=ops.stop_condition.max_evaluations(max_evals=ndims*10000)
# )

# solution, fit = lshade_spacma(
#     popsize=popsize,
#     h=5,
#     fit_fn=problem,
#     lb=lb,
#     ub=ub,
#     ndims=ndims,
#     stop_condition=ops.stop_condition.max_evaluations(max_evals=ndims*10000)
# )

#mealpy.SHADE
#seed = 304417 # np.random.randint(low=0, high=1000000000)
#print(seed)
#np.random.seed(seed)
solution = None
fit = float("inf")
stop_condition = StopCondition(
            max_evals=ndims*10000, 
            #callback=lambda ctx: (ctx.best()[1] - bias) <= 1e-8
)
for i in range(1):
    _solution, _fit = lshade_spacma(
        fit_fn=problem,
        lb=lb,
        ub=ub,
        ndims=ndims,
        stop_condition=stop_condition
    )
    print(_fit)
    if _fit < fit:
        solution = _solution
        fit = _fit

#solution,fit = jade_test()

print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# res = mealpy.evolutionary_based.SHADE.L_SHADE(epoch=max_gen, pop_size=popsize).solve({
#     "obj_func": problem,
#     "bounds": FloatVar(lb=[lb, ] * ndims, ub=[ub, ] * ndims,), 
#     "minmax": "min",
#     "log_to": None,
# })
# print("--- %s seconds ---" % (time.time() - start_time))
# print(f"Benchmark: {res.target.fitness}")

print(solution, fit)
