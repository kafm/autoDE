from typing import Callable, List
from autode.core import (
    ParamGenerator,
    GenerationContext,
    ParamValue,
    ParamRepairStrategy,
    ParamLearningStrategy,
    ParamPerformance,
    ParamValueLearned,
    PubSub,
    GENERATION_START_EVT,
)
import numpy as np
from .lshade import LShade
from .common import lehmer_mean, normalize_values


class LShadeSpa(object):

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=LShade.cauchy_get_strategy(scale),
            learning_strategy=LShade.cauchy_learning_strategy(),
            repair_strategy=LShadeSpa.cauchy_repair_strategy(),
        )

    @staticmethod
    def normal(
        initial_value: int = 0.8, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=LShade.normal_get_strategy(scale),
            learning_strategy=LShadeSpa.normal_learning_strategy(),
        )

    @staticmethod
    def normal_learning_strategy() -> ParamLearningStrategy:
        """Match the weighted Lehmer CR update in the authors' MATLAB code."""
        def callback(loc: ParamValue, results: List[ParamPerformance]):
            successful = [r for r in results if r.individual.improved]
            if not successful:
                return ParamValueLearned(value=loc, think_fast=False)
            values = [r.value for r in successful]
            if max(values) == 0:
                return ParamValueLearned(value=0, think_fast=True)
            weights = [abs(r.individual.improvement) for r in successful]
            return ParamValueLearned(
                value=lehmer_mean(values, weights=normalize_values(weights)),
                think_fast=False,
            )

        return callback

    @staticmethod
    def cauchy_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_evals()
            max_g = ctx.parent.max_evals()
            if g < 0.5 * max_g:
                return 0.45 + 0.1 * np.random.rand()
            return value

        return callback
