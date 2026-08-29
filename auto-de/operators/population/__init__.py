from core import (
    ResizePopulationCallback,
    GenerationContext,
    Population,
    Param,
)
from . import initialization as init

def lpsr(
    pmin: int = 4,
) -> ResizePopulationCallback:

    def callback(population: Population, ctx: GenerationContext):
        pinit = population.init_size()
        new_popsize = round(
            (pmin - pinit)/ctx.parent.max_evals() * ctx.parent.num_evals() + pinit
        )
        #new_population_size = round((4 - init_size) / max_evals * num_evals + init_size)
        # round(
        #     population.init_size()
        #     - (ctx.parent.num_evals() / ctx.parent.max_evals())
        #     * (population.init_size() - pmin)
        # )
        #print(f"New popsize {new_popsize}")
        if new_popsize < pmin:
            new_popsize = pmin
        if new_popsize == ctx.popsize:
            return population
        best_individuals = ctx.picker.get_k_best(new_popsize)
        population.set_individuals(best_individuals)

    return callback
