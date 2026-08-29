import numpy as np
from typing import List, Callable
from core import InitPopulationCallback, SearchContext, GenerationContext, Individual, Population


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

def opposite(popsize: int, seed=None) -> InitPopulationCallback:
    def callback(ctx: SearchContext) -> Population:
        individuals:List[Individual] = []
        rnd = np.random if seed is None else np.random.RandomState(seed)
        for i in range(popsize):
            x = rnd.uniform(low=ctx.lb(), high=ctx.ub(), size=ctx.ndims())
            bounds_sum = np.add(ctx.lb(), ctx.ub())
            x_opposite = np.subtract(bounds_sum, x)
            individuals.append(create_individual(ctx, x_opposite, i))
        return Population(individuals)

    return callback

def semi_opposite(popsize: int, seed=None, s_lambda: float = .5) -> InitPopulationCallback:
    def callback(ctx: SearchContext) -> Population:
        individuals:List[Individual] = []
        rnd = np.random if seed is None else np.random.RandomState(seed)
        for i in range(popsize):
            x = rnd.uniform(low=ctx.lb(), high=ctx.ub(), size=ctx.ndims())
            bounds_sum = np.add(ctx.lb(), ctx.ub())
            x_opposite = np.subtract(bounds_sum, x)
            x_semi_opposite = (1 - s_lambda)*x + s_lambda*x_opposite
            individuals.append(create_individual(ctx, x_semi_opposite, i))
        return Population(individuals)

    return callback