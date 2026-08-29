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
from .common import generate_cauchy_bt_0_1, lehmer_mean, normalize_values


class LShade(object):

    @staticmethod
    def memory(size: int) -> ParamMemory:
        return _LShadeMemory(size)

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=LShade.cauchy_get_strategy(scale),
            learning_strategy=LShade.cauchy_learning_strategy(),
        )

    @staticmethod
    def normal(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=LShade.normal_get_strategy(scale),
            learning_strategy=LShade.normal_learning_strategy(),
        )

    @staticmethod
    def cauchy_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: generate_cauchy_bt_0_1(loc=loc, scale=scale)

    @staticmethod
    def cauchy_learning_strategy() -> ParamLearningStrategy:
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            values = []
            weights = []
            for result in results:
                if result.individual.improved:
                    values.append(result.value)
                    weights.append(abs(result.individual.improvement))
            value = (
                lehmer_mean(values, weights=normalize_values(weights))
                if len(values) > 0
                else loc
            )
            return ParamValueLearned(value=value, think_fast=False)

        return callback

    @staticmethod
    def normal_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: np.clip(np.random.normal(loc=loc, scale=scale), 0, 1)

    @staticmethod
    def normal_learning_strategy() -> ParamLearningStrategy:
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            values = []
            weights = []
            for result in results:
                if result.individual.improved:
                    values.append(result.value)
                    weights.append(abs(result.individual.improvement))
            value = loc
            think_fast = False
            if len(values) > 0:
                if max(values) == 0:
                    value = 0
                    think_fast = True
                else:
                    value = np.sum(np.multiply(values, normalize_values(weights))) #np.average(values, weights=normalize_values(weights)
            return ParamValueLearned(value=value, think_fast=think_fast)

        return callback


class _LShadeMemory(ParamMemory):
    def __init__(self, size: int):
        self._size = size
        self._entries = [ParamMemoryEntry() for _ in range(size)]
        self._k = 0

    def end_memorization(self, ctx: GenerationContext, params: List[AdaptativeParam]):
        if ctx.population.has_improvements():
            new_entry_values = {}
            old_entry = self._entries[self._k]
            new_entry_values = {
                param.id(): self._get_param_learnings(
                    param, old_entry.get(param.id(), param.initial_value())
                )
                for param in params
                if param.memorizable()
            }
            self._entries[self._k] = ParamMemoryEntry(values=new_entry_values)
            k = self._k + 1
            self._k = k if k < self._size else 0

    def _get_param_learnings(
        self, param: AdaptativeParam, memory_value: Optional[ParamValueLearned]
    ) -> ParamValueLearned:
        if memory_value and memory_value.think_fast:
            return memory_value
        return param.get_learnings()

    def get(self, individuals: List[Individual]) -> Dict[int, ParamMemoryEntry]:
        size = len(individuals)
        indices = np.random.randint(low=0, high=self._size, size=size)
        return {individuals[i].id: self._entries[indices[i]] for i in range(size)}
