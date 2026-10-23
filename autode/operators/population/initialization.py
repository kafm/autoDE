import numpy as np
from autode.core import InitPopulationCallback, SearchContext, Individual, Population


def create_individual(ctx: SearchContext, index: int) -> Individual:
    pos = ctx.random.uniform(low=ctx.lb(), high=ctx.ub(), size=ctx.ndims())
    fit = ctx.evaluate(pos)
    return Individual(pos=pos, fit=fit, index=index,id=index)


def random(popsize: int) -> InitPopulationCallback:
    def callback(ctx: SearchContext) -> Population:
        return Population([
            create_individual(ctx, i)
            for i in range(popsize)
        ])

    return callback


def random_batch(popsize: int) -> InitPopulationCallback:
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
        positions = ctx.random.uniform(
            low=ctx.lb(), high=ctx.ub(), size=(popsize, ctx.ndims())
        )
        fits = ctx.evaluate_many(positions)
        return Population([
            Individual(pos=pos, fit=float(fit), index=i, id=i)
            for i, (pos, fit) in enumerate(zip(positions, fits))
        ])

    callback.population_size = popsize
    return callback
