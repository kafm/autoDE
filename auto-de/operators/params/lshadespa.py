from typing import Callable, List
from core import (
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
            learning_strategy=LShade.normal_learning_strategy(),
        )

    @staticmethod
    def cauchy_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_evals()
            max_g = ctx.parent.max_evals()
            if g < 0.5 * max_g:
                return 0.45 + 0.1 * np.random.rand()
            return value

        return callback
