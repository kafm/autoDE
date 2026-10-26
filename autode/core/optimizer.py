from typing import List, Tuple, Optional, Dict
from copy import copy
from .context import *
from .params import Params
from .stop_condition import StopCondition
from .events import *
from .update_method import UpdateMethod


class Optimizer:
    def __init__(
        self,
        init_strategy: InitPopulationCallback,
        update_strategy: List[UpdateMethod],
        stop_condition: StopCondition,
        events: PubSub,
        neighborhood_strategy: Optional[NeighborhoodCallback] = None,
        resize_strategy: Optional[ResizePopulationCallback] = None,
        params: Optional[Params] = None,
        archive: Optional[Archive] = None,
        seed: Optional[int] = None
    ):
        self._init_strategy = init_strategy
        self._neighborhood_strategy = neighborhood_strategy
        self._resize_strategy = resize_strategy
        self._update_strategy = update_strategy
        self._stop_condition = copy(stop_condition)
        self._params = params if params else Params(events=events)
        self._events = events
        self._rng = np.random.default_rng(seed)
        self._archive = archive
        if self._archive is not None:
            self._archive.set_rng(self._rng)

    def search(
        self,
        eval_fn: FitnessFunction,
        lb: Bounds,
        ub: Bounds,
        ndims: NumDimensions,
    ) -> Tuple[List[float], float]:
        # Budget inference belongs to this search, not to the caller's
        # configuration or a previous search using this optimizer.
        stop_condition = copy(self._stop_condition)
        initial_size = getattr(self._init_strategy, "population_size", None)
        if initial_size is not None:
            self._assert_stop_condition(initial_size, stop_condition)
        ctx = SearchContext(
            eval_fn=eval_fn,
            lb=lb,
            ub=ub,
            ndims=ndims,
            stop_condition=stop_condition.__call__,
            max_evals=stop_condition.max_evals,
            max_gen=stop_condition.max_gen,
            rng=self._rng
        )
        population = self._init_strategy(ctx)
        init_size = population.size()
        self._assert_stop_condition(init_size, stop_condition)
        gen_ctx = None
        while ctx.hasNext():
            gen_ctx = GenerationContext(
                population=population,
                parent=ctx,
                previous=gen_ctx,
                archive=self._archive
            )
            self._events.publish(GENERATION_START_EVT, gen_ctx)
            # gen_ctx.previous and self._params.update(gen_ctx)
            population.set_neighborhood(self._get_neighborhood(ctx))
            for strategy in self._update_strategy:
                strategy.update(
                    self._events,
                    gen_ctx,
                    neighbors_callback=population.get_neighbors,
                )
            # TODO how to proceed when we have more than one update in the pipeline? Shoud the ranking be updated?
            self._events.publish(GENERATION_ENDED_EVT, gen_ctx)
            self.resize_population(population, gen_ctx)
            self._archive and self._archive.resize(population.size())
            self._events.publish(GENERATION_FINALIZED_EVT, gen_ctx)
            ctx.next()

        return ctx.best()

    def resize_population(
        self, population: Population, ctx: GenerationContext
    ) -> Population:
        self._resize_strategy and self._resize_strategy(population, ctx)

    def _get_neighborhood(
        self, ctx: GenerationContext
    ) -> Optional[Dict[int, List[int]]]:
        if self._neighborhood_strategy:
            return self._neighborhood_strategy(ctx)
        return None

    def _assert_stop_condition(self, psize: int, stop_condition: StopCondition):
        stop_condition.infer_max_evals_from_popsize_if_not_set(psize)

    def params(self) -> Params:
        return self._params
