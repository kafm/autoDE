from typing import Callable
from core import ParamGenerator
from core import (
    ParamGenerator,
    GenerationContext,
    ParamValue,
    PubSub,
    GENERATION_START_EVT,
)


class Evaluation:
   
    @staticmethod
    def linear(events: PubSub, low: float = 0, high: float = 1) -> Callable[[int], ParamGenerator]:
        generator = _EvaluationLinear(events, low=low, high=high)
        return lambda id: ParamGenerator(
            id=id, initial_value=None, memorizable=False, get_strategy=lambda _: generator.get()
        )


class _EvaluationLinear:
    def __init__(
        self,
        events: PubSub,
        low: float = 0,
        high: float = 1,
    ):
        self._low = low
        self._high = high
        self._value = low
        self._unsub_gen_start = events.subscribe(GENERATION_START_EVT, self._update)

    def __del__(self):
        self._unsub_gen_start()

    def get(self) -> ParamValue:
        return self._value

    def _update(self, ctx: GenerationContext):
        max_evals = ctx.parent.max_evals()
        num_evals = ctx.parent.num_evals()
        #print(f"Going to update p {self._value} max_evals={max_evals}, num_evals={num_evals}")
        self._value = (
            (self._high - self._low) / max_evals
        ) * num_evals + self._low 
