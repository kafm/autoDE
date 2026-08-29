from typing import Dict, List, Optional, Callable
from core import (
    ParamMemory,
    ParamMemoryEntry,
    ParamValueLearned,
    ParamGenerator,
    GenerationContext,
    AdaptativeParam,
    Individual,
    ParamValue,
    ParamPerformance,
    ParamGetStrategy, 
    ParamLearningStrategy,
    ParamRepairStrategy
)
import numpy as np
from .common import generate_cauchy_bt_0_1, lehmer_mean, normalize_values

class IlShade(object):

    @staticmethod
    def memory(size: int) -> ParamMemory:
        return _IlShadeMemory(size)

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=IlShade.cauchy_get_strategy(scale),
            learning_strategy=IlShade.cauchy_learning_strategy(),
            repair_strategy=IlShade.cauchy_repair_strategy()
        )

    @staticmethod
    def normal(
        initial_value: int = 0.8, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=IlShade.normal_get_strategy(scale),
            learning_strategy=IlShade.normal_learning_strategy(),
            repair_strategy=IlShade.normal_repair_strategy()
        )

    @staticmethod
    def cauchy_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: generate_cauchy_bt_0_1(loc=loc, scale=scale)
    
    @staticmethod
    def cauchy_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_gen()
            max_g = ctx.parent.max_gen()
            if g < .25*max_g:
                return min(value,.7)
            elif g < .5*max_g:
                return min(value,.8)
            elif g < .75*max_g:
                return min(value,.9)
            return value
        return callback

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
                (lehmer_mean(values, weights=normalize_values(weights)) + loc)/2
                if len(values) > 0
                else loc
            )
            return ParamValueLearned(value=value, think_fast=False)

        return callback

    @staticmethod
    def normal_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: np.clip(np.random.normal(loc=loc, scale=scale), 0, 1)
    
    @staticmethod
    def normal_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_gen()
            max_g = ctx.parent.max_gen()
            if g < .25*max_g:
                return max(value,.5)
            elif g < .5*max_g:
                return max(value,.25)
            return value
        return callback

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
                    value = (np.average(values, weights=normalize_values(weights)) + loc)/2
            return ParamValueLearned(value=value, think_fast=think_fast)

        return callback

class _IlShadeMemory(ParamMemory):
    def __init__(self, size: int, fixed_value: float = 0.9):
        self._size = size
        self._fixed_index = size - 1
        self._entries = [
            (
                _FixedParamMemoryEntry(fixed_value)
                if i == self._fixed_index
                else ParamMemoryEntry()
            )
            for i in range(size)
        ]
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
            self._k = k if k < self._fixed_index else 0

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

class _FixedParamMemoryEntry(ParamMemoryEntry):
    def __init__(self, value):
        self._value = value

    def get(self, key: int, default_value: Optional[ParamValue] = None) -> ParamValueLearned:
        return ParamValueLearned(value=self._value)

    def set(self, key: int, value: float):
        pass

    def contains(self, key: int) -> bool:
        return True
