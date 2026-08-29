from dataclasses import dataclass
from abc import ABC
from typing import Optional, Union, List, Dict, Callable, Any
from .context import *
from .events import *
import uuid

ParamValue = Union[int, float, bool]

ParamValues = Dict[int, ParamValue]


@dataclass
class ParamPerformance:
    value: ParamValue
    individual: Individual


@dataclass
class ParamValueLearned:
    value: ParamValue
    think_fast: bool = False


class Param:
    def __init__(self, value: ParamValue):
        self._value = value

    def set_value(self, value: ParamValue):
        self._value = value

    def get_value(self) -> ParamValue:
        return self._value

    def __call__(self) -> ParamValue:
        return self._value


class AdaptativeParam(Param):
    def __init__(self, generator: "ParamGenerator"):
        super().__init__(generator.initial_value)
        self._values: Dict[int, ParamPerformance] = {}
        self._ctx_individual: Individual = None
        self._ctx: GenerationContext = None
        self._generator = generator
        self._think_fast = False
        self._ref = uuid.uuid4()

    def initial_value(self) -> ParamValue:
        return self._generator.initial_value

    def memorizable(self) -> bool:
        return self._generator.memorizable

    def get_value(self) -> ParamValue:
        return self._value

    def id(self) -> int:
        return self._generator.id

    def start_collection(self, ctx: GenerationContext):
        self._ctx = ctx
        self._values = {}

    def get_learnings(
        self, values: Optional[List[ParamPerformance]] = None
    ) -> ParamValueLearned:
        return self._generator.get_learnings(
            self._value, values if values else self._values.values()
        )

    def self_learn(self):
        self.set_value(self.get_learnings())

    def set_value(self, learned_value: ParamValueLearned):
        self._value = learned_value.value
        self._think_fast = learned_value.think_fast

    def get_collected_values(self) -> Dict[int, ParamPerformance]:
        return self._values

    def __call__(self) -> ParamValue:
        value = self._generator.get(self._ctx, self._value, self._think_fast)
        if self._ctx_individual:
            self._values[self._ctx_individual.id] = ParamPerformance(
                individual=self._ctx_individual, value=value
            )
        return value

    def set_context_individual(self, individual: Individual):
        self._ctx_individual = individual

    def collect_performace(self, individual: Individual):
        if individual.id in self._values:
            self._values[self._ctx_individual.id].individual = individual


ParamLearningStrategy = Callable[
    [ParamValue, List[ParamPerformance]], ParamValueLearned
]
ParamGetStrategy = Callable[..., ParamValue]
ParamRepairStrategy = Callable[
    [GenerationContext, ParamValue], ParamValue
]  # TODO should pass search context


class Params:
    def __init__(
        self,
        events: PubSub,
        memory: Optional["ParamMemory"] = None,
    ):
        self._adaptative_params: List[AdaptativeParam] = []
        self._adaptative_params_cnt = 0
        self._memory: ParamMemory = memory
        self._individual_memory: Dict[int, ParamMemoryEntry] = {}
        self._unsub_gen_start = events.subscribe(GENERATION_START_EVT, self._start_memorization)
        self._unsub_gen_ended = events.subscribe(GENERATION_ENDED_EVT, self._end_memorization)
        self._unsub_update_start =events.subscribe(UPDATED_START_EVT, self._set_context_individual)
        self._unsub_updaten_ended =events.subscribe(UPDATED_END_EVT, self._collect_performace)

    def __del__(self):
        self._unsub_gen_start()
        self._unsub_gen_ended()
        self._unsub_update_start()
        self._unsub_updaten_ended()

    def static(self, value: float) -> "Param":
        return Param(value)

    def adaptative(
        self, generator_callback: Callable[[Any], "ParamGenerator"]
    ) -> Param:
        param = AdaptativeParam(
            generator=generator_callback(self._adaptative_params_cnt) # #uuid.uuid4()
        )
        self._adaptative_params.append(param)
        self._adaptative_params_cnt += 1
        return param

    def _set_context_individual(self, individual: Individual):
        for param in self._adaptative_params:
            memory_value = self._get_memory(individual, param)
            if memory_value is not None:
                param.set_value(memory_value)
            param.set_context_individual(individual)

    def _get_memory(
        self, individual: Individual, param: AdaptativeParam
    ) -> Optional[ParamValueLearned]:
        individual_memory = (
            self._individual_memory.get(individual.id, None)
            if param.memorizable()
            else None
        )
        return (
            individual_memory.get(param.id(), param.initial_value())
            if individual_memory
            else None
        )

    def _collect_performace(self, individual: Individual):
        for param in self._adaptative_params:
            param.collect_performace(individual)

    def _start_memorization(self, ctx: GenerationContext):
        if self._memory:
            self._memory.start_memorization(ctx, self._adaptative_params)
            self._individual_memory = self._memory.get(ctx.population.get())
        for param in self._adaptative_params:
            param.start_collection(ctx)

    def _end_memorization(self, ctx: GenerationContext):
        if self._memory:
            self._memory.end_memorization(ctx, self._adaptative_params)
        else:
            self._self_learn_params()

    def _self_learn_params(self):
        for param in self._adaptative_params:
            param.self_learn()


@dataclass
class ParamGenerator:
    id: int
    initial_value: ParamValue
    get_strategy: ParamGetStrategy
    learning_strategy: Optional[ParamLearningStrategy] = None
    repair_strategy: Optional[ParamRepairStrategy] = None
    memorizable: bool = True

    def get(
        self, ctx: GenerationContext, value: ParamValue, think_fast: bool
    ) -> ParamValue:
        return self._repair(
            ctx, value if think_fast else self.get_strategy(value)
        )

    def _repair(self, ctx: GenerationContext, value: ParamValue) -> ParamValue:
        return self.repair_strategy(ctx, value) if self.repair_strategy else value

    def get_learnings(
        self, value: ParamValue, colllection: Dict[int, ParamPerformance]
    ) -> ParamValueLearned:
        if self.learning_strategy:
            return self.learning_strategy(value, colllection)
        return ParamValueLearned(value=self.initial_value, think_fast=False)


class ParamMemoryEntry:
    def __init__(self, values: Optional[Dict[int, ParamValueLearned]] = None):
        self._values: ParamValues = values if values else {}

    def get(
        self, key: int, default_value: Optional[ParamValue] = None
    ) -> ParamValueLearned:
        try:
            return self._values[key]
        except KeyError:
            if default_value is not None:
                value = ParamValueLearned(value=default_value)
                self.set(key, value)
                return value
            raise KeyError()

    def set(self, key: int, value: float):
        self._values[key] = value

    def contains(self, key: int) -> bool:
        return key in self._values


class ParamMemory(ABC):

    def start_memorization(self, ctx: GenerationContext, params: List[AdaptativeParam]):
        pass

    def end_memorization(self, ctx: GenerationContext, params: List[AdaptativeParam]):
        pass

    def get(self, individuals: List[Individual]) -> Dict[int, ParamMemoryEntry]:
        raise NotImplementedError
