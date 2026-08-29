from typing import Callable
from core import (
    ParamGenerator,
    GenerationContext,
    ParamValue,
    ParamRepairStrategy
)
from .ilshade import IlShade

class Fode(object):

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=IlShade.cauchy_get_strategy(scale),
            learning_strategy=IlShade.cauchy_learning_strategy(),
            repair_strategy=Fode.cauchy_repair_strategy()
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
            repair_strategy=Fode.normal_repair_strategy()
        )
    
    @staticmethod
    def cauchy_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            max_evals = ctx.parent.max_evals()
            num_evals = ctx.parent.num_evals()
            if num_evals < .6*max_evals:
                return min(value,.7)
            return value
        return callback
    
    @staticmethod
    def normal_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            max_evals = ctx.parent.max_evals()
            num_evals = ctx.parent.num_evals()
            if num_evals < .25*max_evals:
                return max(value,.7)
            elif num_evals < .5*max_evals:
                return max(value,.6)
            return value
        return callback
