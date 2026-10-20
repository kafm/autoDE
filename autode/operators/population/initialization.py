import numpy as np
from typing import List
from autode.core import InitPopulationCallback, SearchContext, Individual, Population


def create_individual(ctx: SearchContext, pos: List[float], index: int) -> Individual:
    fit = ctx.evaluate(pos)
    return Individual(pos=pos, fit=fit, index=index,id=index)


def random(popsize: int, seed=None) -> InitPopulationCallback:
    def callback(ctx: SearchContext) -> Population:
        rnd = np.random if seed is None else np.random.RandomState(seed)
        return Population([
            create_individual(
                ctx, 
                rnd.uniform(low=ctx.lb(), high=ctx.ub(), size=ctx.ndims()),
                i
            )
            for i in range(popsize)
        ])

    return callback


def random_batch(popsize: int, seed=None) -> InitPopulationCallback:
    """Generate positions together and evaluate the initial population once.

    population_size lets the optimizer infer a generation-based evaluation
    budget before initialization. The full population must fit in that budget.
    """
    if isinstance(popsize, bool) or not isinstance(popsize, (int, np.integer)) or popsize <= 0:
        raise ValueError("Population size must be a positive integer")

    def callback(ctx: SearchContext) -> Population:
        max_evals = ctx.max_evals()
        if max_evals is None:
            raise ValueError("Evaluation budget must be initialized before batch initialization")
        if max_evals - ctx.num_evals() < popsize:
            raise ValueError("Evaluation budget is smaller than the initial population")
        rnd = np.random if seed is None else np.random.RandomState(seed)
        positions = rnd.uniform(
            low=ctx.lb(), high=ctx.ub(), size=(popsize, ctx.ndims())
        )
        fits = ctx.evaluate_many(positions)
        return Population([
            Individual(pos=pos, fit=float(fit), index=i, id=i)
            for i, (pos, fit) in enumerate(zip(positions, fits))
        ])

    callback.population_size = popsize
    return callback
