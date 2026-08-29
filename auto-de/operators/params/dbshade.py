from typing import Callable, List, Optional
from core import (
    ParamGenerator,
    ParamValue,
    ParamValueLearned,
    ParamLearningStrategy, 
    ParamPerformance,
    ParamRepairStrategy
)
from .common import lehmer_mean, normalize_values
from .ilshade import IlShade
from .jso import Jso
import numpy as np

class DbShade(object):

    #JSO Variant (DISH)
    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=IlShade.cauchy_get_strategy(scale),
            learning_strategy=DbShade.cauchy_learning_strategy(),
            repair_strategy=Jso.cauchy_repair_strategy()
        )
    
    #JSO Variant (DISH)
    @staticmethod
    def normal(
        initial_value: int = 0.8, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return Jso.normal(initial_value=initial_value, scale=scale)

    @staticmethod
    def cauchy_learning_strategy() -> ParamLearningStrategy:
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            values = []
            weights = []
            for result in results:
                if result.individual.improved:
                    values.append(result.value)
                    weights.append(np.sqrt(np.sum(np.square(result.individual.delta))))
            value = (
                lehmer_mean(values, weights=normalize_values(weights))
                if len(values) > 0
                else loc
            )
            return ParamValueLearned(value=value, think_fast=False)

        return callback
    

