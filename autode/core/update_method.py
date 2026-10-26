"""Deferred UpdateMethod using GenerationContext.evaluate_many.

Use this UpdateMethod with the existing Optimizer. The objective must support
both scalar evaluations (initialization/boosters) and batches of positions.
All trials are generated before one evaluate_many call; selection follows after
fitness values are available. Stopping conditions are observed between batches.
"""

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple
import numpy as np

from .context import (
    ConvergenceBoosterCallback,
    ConvergenceBoosterConditionCallback,
    CrossoverCallback,
    FilterNeighborsCallback,
    GenerationContext,
    Individual,
    MutationCallback,
    PickerCallback,
    ReparationCallback,
    SelectionCallback,
    SelectionContext,
    UpdateContext,
)
from .events import PubSub, UPDATED_START_EVT, UPDATED_END_EVT
from .population import DefaultPicker, Picker


@dataclass
class EvaluationResult:
    survivor: Individual
    archived: Optional[Individual] = None


@dataclass
class ConvergenceBooster:
    method: ConvergenceBoosterCallback
    condition: ConvergenceBoosterConditionCallback

    def applies(self, ctx: GenerationContext) -> bool:
        return self.condition(ctx)

    def boost(self, ctx: GenerationContext, individual: Individual) -> Individual:
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

    def _default_picker(self, ctx: UpdateContext) -> Picker:
        return DefaultPicker(
            individuals=ctx.neighbors,
            ranking=ctx.ranking,
            archived=ctx.parent.archive.all() if ctx.parent.archive else [],
            rng=ctx.random,
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
                individual,
                neighbors=neighbors,
                ranking=ranking,
                parent=ctx,
                picker_impl=self.picker,
            )
            events.publish(UPDATED_START_EVT, individual)
            new_individual, archived = self._get_new_individual(update_ctx, events)
            new_individuals[new_individual.index] = new_individual
            events.publish(UPDATED_END_EVT, new_individual)
            archived and archived_individuals.append(archived)
        ctx.population.set_individuals(new_individuals, refactor_indices=False)
        ctx.archive and len(archived_individuals) > 0 and ctx.archive.put(
            archived_individuals
        )
        self._boost_if_applies(ctx)

    def _boost_if_applies(self, ctx: GenerationContext):
        # Generation-level picking uses the committed survivors, independently
        # of the per-individual picker configured for mutation (e.g. FODE).
        ctx.picker = DefaultPicker(
            individuals=ctx.population.get(),
            ranking=ctx.population.ranking(),
            archived=ctx.archive.all() if ctx.archive else [],
            rng=ctx.random,
        )
        if self.convergence_booster is None or not ctx.parent.hasNext():
            return
        best = ctx.picker.get_best()
        if self.convergence_booster and self.convergence_booster.applies(ctx):
            trial, fit = self.convergence_booster.boost(ctx, best)
            if trial is not None:
                res = self.evaluate(ctx, best, trial, fit)
                ctx.population.replace_individual(res.survivor)

    def _get_new_individual(
        self, ctx: UpdateContext, events: PubSub
    ) -> Tuple[Individual, Optional[Individual]]:
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

    def evaluate(
        self,
        ctx: GenerationContext,
        individual: Individual,
        solution: List[float],
        fit: float,
    ) -> EvaluationResult:
        improved = fit < individual.fit
        improvement = individual.fit - fit if improved else 0
        trials = individual.trials + 1 if not improved else 0
        delta = solution - individual.pos
        if type(individual) is Individual:
            attributes = dict(
                id=individual.id, index=individual.index, archived=individual.archived,
                improved=improved, trials=trials, improvement=improvement, delta=delta,
            )
            parent = Individual(pos=individual.pos, fit=individual.fit, **attributes)
            offspring = Individual(pos=solution, fit=fit, **attributes)
        else:
            # Preserve dataclass subclass fields and initialization behavior.
            parent = replace(
                individual, improved=improved, trials=trials,
                improvement=improvement, delta=delta,
            )
            offspring = replace(
                individual, pos=solution, fit=fit, improved=improved,
                trials=trials, improvement=improvement, delta=delta,
            )
        if self.selection(
            SelectionContext(individual=parent, trial=offspring, parent=ctx)
        ):
            return EvaluationResult(survivor=offspring, archived=parent)
        return EvaluationResult(survivor=parent)


class BatchUpdateMethod(UpdateMethod):
    """Publish update starts before trial generation and ends after selection.

    Starts precede the batch evaluation; ends follow it. Pair events by the
    payload's individual ID, not adjacency. End subscribers must use the payload
    rather than the last active parameter context. Selection uses SelectionContext.
    """

    def update(
        self,
        events: PubSub,
        ctx: GenerationContext,
        neighbors_callback: Callable[[int], Tuple[List[Individual], List[int]]],
    ):
        if not ctx.parent.hasNext():
            return
        individuals = list(self.filter(ctx) if self.filter else ctx.population.get())
        remaining = max(0, ctx.parent.max_evals() - ctx.parent.num_evals())
        individuals = individuals[:remaining]
        new_individuals = list(ctx.population.get())
        trials = []
        for individual in individuals:
            neighbors, ranking = neighbors_callback(individual.index)
            update_ctx = UpdateContext(
                individual,
                neighbors=neighbors,
                ranking=ranking,
                parent=ctx,
                picker_impl=self.picker,
            )
            events.publish(UPDATED_START_EVT, individual)
            trial = self.repair(update_ctx, self.mutation(update_ctx))
            if self.crossover:
                trial = self.crossover(update_ctx, trial)
            # Own each position even if an operator reuses its output buffer.
            trials.append(np.array(trial, dtype=float, copy=True))

        num_evals_before = ctx.parent.num_evals()
        fits = ctx.evaluate_many(trials)
        evaluated_count = ctx.parent.num_evals() - num_evals_before
        archived_individuals = []
        # evaluate_many pads unevaluated trials with infinity. Do not select or
        # increment failure counters for those trials, even with custom selection.
        for individual, trial, fit in zip(
            individuals[:evaluated_count],
            trials[:evaluated_count],
            fits[:evaluated_count],
        ):
            result = self.evaluate(ctx, individual, trial, float(fit))
            new_individuals[result.survivor.index] = result.survivor
            events.publish(UPDATED_END_EVT, result.survivor)
            if result.archived is not None:
                archived_individuals.append(result.archived)

        ctx.population.set_individuals(new_individuals, refactor_indices=False)
        if ctx.archive and archived_individuals:
            ctx.archive.put(archived_individuals)
        self._boost_if_applies(ctx)
