from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass, replace
from functools import partial
from .context import *
from .params import Params
from .stop_condition import StopCondition
from .events import *


class Optimizer:
    def __init__(
        self,
        init_strategy: InitPopulationCallback,
        update_strategy: List["UpdateMethod"],
        stop_condition: StopCondition,
        events: PubSub,
        neighborhood_strategy: Optional[NeighborhoodCallback] = None,
        resize_strategy: Optional[ResizePopulationCallback] = None,
        params: Optional[Params] = None,
        archive: Optional[Archive] = None,
    ):
        self._init_strategy = init_strategy
        self._neighborhood_strategy = neighborhood_strategy
        self._resize_strategy = resize_strategy
        self._update_strategy = update_strategy
        self._stop_condition = stop_condition
        self._params = params if params else Params(events=events)
        self._archive = archive
        self._events = events

    def search(
        self,
        eval_fn: FitnessFunction,
        lb: Bounds,
        ub: Bounds,
        ndims: NumDimensions,
    ) -> Tuple[List[float], float]:
        ctx = SearchContext(
            eval_fn=eval_fn,
            lb=lb,
            ub=ub,
            ndims=ndims,
            stop_condition=self._stop_condition.__call__,
            max_evals=self._stop_condition.max_evals,
            max_gen=self._stop_condition.max_gen,
        )
        population = self._init_strategy(ctx)
        init_size = population.size()
        self._assert_stop_condition(init_size)
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

    def _assert_stop_condition(self, psize: int):
        self._stop_condition.infer_max_evals_from_popsize_if_not_set(psize)

    def params(self) -> Params:
        return self._params

@dataclass
class EvaluationResult:
    survivor: Individual
    archived: Optional[Individual] = None

@dataclass
class ConvergenceBooster:
    method: ConvergenceBoosterCallback
    condition: ConvergenceBoosterConditionCallback

    def applies(self, ctx: GenerationContext)->bool:
        return self.condition(ctx)
    
    def boost(self,ctx: GenerationContext, individual: Individual)->Individual:
        return self.method(ctx, individual)

@dataclass
class UpdateMethod:
    mutation: MutationCallback
    selection: SelectionCallback
    picker: Optional[PickerCallback] = None
    reparation: Optional[ReparationCallback] = None
    crossover: Optional[CrossoverCallback] = None
    filter: Optional[FilterNeighborsCallback] = None
    convergence_booster: Optional[ConvergenceBooster] = None

    def __post_init__(self):
        if self.picker is None:
            self.picker = self._default_picker
    
    def _default_picker(self, ctx: UpdateContext)->Picker:
        return DefaultPicker(
            individuals=ctx.neighbors,
            ranking=ctx.ranking,
            archived=ctx.parent.archive.all() if ctx.parent.archive else []
        )

    def update(
        self,
        events: PubSub,
        ctx: GenerationContext,
        neighbors_callback: Callable[[int], Tuple[List[Individual], List[int]]],
    ):
        new_individuals = ctx.population.get()
        individuals = self.filter(ctx) if self.filter else ctx.population.get()
        archived_individuals = []
        for individual in individuals:
            if not ctx.parent.hasNext():
                break
            neighbors, ranking = neighbors_callback(individual.index)
            update_ctx = UpdateContext(
                individual, neighbors=neighbors, ranking=ranking, parent=ctx,
                picker_impl=self.picker
            )
            events.publish(UPDATED_START_EVT, individual)
            new_individual,archived = self._get_new_individual(update_ctx, events)
            # trial = self.repair(update_ctx, self.mutation(update_ctx))
            # if self.crossover:
            #     trial = self.crossover(update_ctx, trial)
            # res = self.evaluate(update_ctx, trial)
            # new_individual = res.survivor
            # archived = res.archived
            new_individuals[new_individual.index] = new_individual
            events.publish(UPDATED_END_EVT, new_individual)
            archived and archived_individuals.append(archived)
        ctx.population.set_individuals(new_individuals, refactor_indices=False)
        ctx.archive and len(archived_individuals) > 0 and ctx.archive.put(archived_individuals)
        self._boost_if_applies(ctx)

    def _boost_if_applies(self, ctx: GenerationContext):
        best = ctx.picker.get_best()
        if self.convergence_booster and self.convergence_booster.applies(ctx):
            trial, fit = self.convergence_booster.boost(ctx, best)
            if trial is not None:
                res = self.evaluate(ctx, best, trial, fit)
                ctx.population.replace_individual(res.survivor)

    def _get_new_individual(self, ctx: UpdateContext, events: PubSub)->Tuple[Individual,Optional[Individual]]:
        events.publish(UPDATED_START_EVT, ctx.individual)
        trial = self.repair(ctx, self.mutation(ctx))
        if self.crossover:
            trial = self.crossover(ctx, trial)
        fit = ctx.parent.evaluate(trial)
        res = self.evaluate(ctx.parent, ctx.individual, trial, fit)
        return res.survivor, res.archived

    def repair(self, ctx: UpdateContext, solution: List[float]) -> List[float]:
        if self.reparation:
            return self.reparation(ctx, solution)
        return solution

    def evaluate(self, ctx: GenerationContext, individual: Individual, solution: List[float], fit: float) -> EvaluationResult:
        improved = fit < individual.fit
        improvement = individual.fit - fit if improved else 0
        trials = individual.trials + 1 if not improved else 0
        delta = solution - individual.pos
        parent: Individual = replace(
            individual,
            improved=improved,
            trials=trials,
            improvement=improvement,
            delta=delta,
        )
        offspring: Individual = replace(
            individual,
            pos=solution,
            fit=fit,
            improved=improved,
            trials=trials,
            improvement=improvement,
            delta=delta,
        )
        if self.selection(
            SelectionContext(individual=parent, trial=offspring, parent=ctx)
        ):
            return EvaluationResult(survivor=offspring,archived=parent)
        return EvaluationResult(survivor=parent)