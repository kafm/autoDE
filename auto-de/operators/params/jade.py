from typing import Callable, Dict, Optional, List
import numpy as np
from core import (
    ParamGenerator,
    ParamPerformance,
    ParamLearningStrategy,
    ParamGetStrategy,
    ParamValue,
    ParamValueLearned
)
from .common import generate_cauchy_bt_0_1, lehmer_mean


class Jade(object):

    @staticmethod
    def cauchy(
        initial_value: int = 0.5,
        scale: float = 0.1,
        learning_rate: float = 0.1,
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=Jade.cauchy_get_strategy(scale),
            learning_strategy=Jade.cauchy_learning_strategy(learning_rate),
        )

    @staticmethod
    def normal(
        initial_value: int = 0.5, learning_rate: float = 0.1, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=Jade.normal_get_strategy(scale),
            learning_strategy=Jade.normal_learning_strategy(learning_rate),
        )

    @staticmethod
    def cauchy_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: generate_cauchy_bt_0_1(loc=loc, scale=scale)

    @staticmethod
    def cauchy_learning_strategy(learning_rate: float = 0.1) -> ParamLearningStrategy:
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            improved_values = [
                result.value for result in results if result.individual.improved
            ]
            value = (
                (1 - learning_rate) * loc + learning_rate * lehmer_mean(
                    improved_values
                )
                if len(improved_values) > 0 else 
                (1 - learning_rate) * loc + learning_rate * .5
            )
            return ParamValueLearned(value=value, think_fast=False)
            
        return callback

    @staticmethod
    def normal_get_strategy(scale: float = 0.1) -> ParamGetStrategy:
        return lambda loc: np.clip(np.random.normal(loc=loc, scale=scale),0,1)

    @staticmethod
    def normal_learning_strategy(learning_rate: float = 0.1) -> ParamLearningStrategy:
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            improved_values = [
                result.value for result in results if result.individual.improved
            ]
            value = (
                (1 - learning_rate) * loc + learning_rate * np.mean(improved_values)
                if len(improved_values) > 0 else 
                (1 - learning_rate) * loc + learning_rate * .5
            )
            return ParamValueLearned(value=value, think_fast=False)
        
        return callback