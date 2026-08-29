from typing import Callable, Dict, Optional, List
import numpy as np
from core import (
    ParamGenerator,
    ParamPerformance,
    ParamLearningStrategy,
    ParamGetStrategy,
    ParamValue,
    ParamMemory,
    ParamValueLearned,
    ParamMemoryEntry,
    GenerationContext,
    AdaptativeParam,
    Individual,
)
from .common import generate_cauchy_bt_0_1, lehmer_mean, normalize_values, manhattan_distance

# TODO accept any neighborhood implementation

class NShade(object):

    @staticmethod
    def memory() -> ParamMemory:
        return _NShadeMemory()

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=NShade.cauchy_get_strategy(scale),
            learning_strategy=NShade.cauchy_learning_strategy(),
        )

    @staticmethod
    def normal(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=NShade.normal_get_strategy(scale),
            learning_strategy=NShade.normal_learning_strategy(),
        )

    @staticmethod
    def cauchy_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: generate_cauchy_bt_0_1(loc=loc, scale=scale)

    @staticmethod
    def cauchy_learning_strategy() -> ParamLearningStrategy:
        def callback(_: ParamValue, results: List[ParamPerformance]):
            values = []
            weights = []
            for result in results:
                if result.individual.improved:
                    values.append(result.value)
                    weights.append(abs(result.individual.improvement))
            value = (
                lehmer_mean(values, weights=normalize_values(weights))
                if len(values) > 0 else
                np.random.uniform(low=.5, high=1)
            )
            return ParamValueLearned(value=value, think_fast=False)

        return callback

    @staticmethod
    def normal_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: np.clip(np.random.normal(loc=loc, scale=scale), 0, 1)

    @staticmethod
    def normal_learning_strategy() -> ParamLearningStrategy:
        def callback(_: ParamValue, results: List[ParamPerformance]):
            values = []
            weights = []
            for result in results:
                if result.individual.improved:
                    values.append(result.value)
                    weights.append(abs(result.individual.improvement))
            value = (
                np.sum(np.multiply(values, normalize_values(weights)))
                if len(values) > 0 else
                np.random.uniform(low=.1, high=1)
            )
            return ParamValueLearned(value=value, think_fast=False)

        return callback


class _NShadeMemory(ParamMemory):
    def __init__(self):
        self._entries: Dict[int, ParamMemoryEntry] = {}

    def start_memorization(self, ctx: GenerationContext, params: List[AdaptativeParam]):
        individuals = ctx.population.get()
        if ctx.previous is None:
            self._entries = {i.id: ParamMemoryEntry() for i in individuals}
            return
        popsize = ctx.popsize
        nsize = round(np.sqrt(popsize))
        entries = {}
        for i in individuals:
            neighbors: List[Individual] = sorted(
                 [j for j in individuals if j.id != i.id],
                 key=lambda j: manhattan_distance(i.pos, j.pos),
             )[:nsize]
            entries[i.id] = {
                param.id(): self._get_param_learnings(param, neighbors)
                for param in params
                if param.memorizable()
            }
        self._entries = entries

    def _get_param_learnings(
        self, param: AdaptativeParam, neighbors: List[Individual]
    ) -> ParamValueLearned:
        values = param.get_collected_values()
        return param.get_learnings([values[i.id] for i in neighbors if i.id in values])

    def get(self, individuals: List[Individual]) -> Dict[int, ParamMemoryEntry]:
        return self._entries