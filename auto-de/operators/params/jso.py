from typing import Callable
from core import (
    ParamGenerator,
    GenerationContext,
    ParamValue,
    ParamRepairStrategy
)
from .ilshade import IlShade

class Jso(object):

    @staticmethod
    def cauchy(
        initial_value: int = 0.5, scale: float = 0.1
    ) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=initial_value,
            get_strategy=IlShade.cauchy_get_strategy(scale),
            learning_strategy=IlShade.cauchy_learning_strategy(),
            repair_strategy=Jso.cauchy_repair_strategy()
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
            repair_strategy=Jso.normal_repair_strategy()
        )
    
    @staticmethod
    def cauchy_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_gen()
            max_g = ctx.parent.max_gen()
            if g < .6*max_g:
                return min(value,.7)
            return value
        return callback
    
    @staticmethod
    def normal_repair_strategy() -> ParamRepairStrategy:
        def callback(ctx: GenerationContext, value: ParamValue) -> ParamValue:
            g = ctx.parent.num_gen()
            max_g = ctx.parent.max_gen()
            if g < .25*max_g:
                return max(value,.7)
            elif g < .5*max_g:
                return max(value,.6)
            return value
        return callback
