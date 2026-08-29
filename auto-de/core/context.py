from typing import Callable, List, Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
import numbers
import numpy as np
from .population import Individual, Archive, Picker, Population, DefaultPicker


class SearchContext:
    def __init__(
        self,
        eval_fn: Callable[[List[float]], float],
        stop_condition: Callable[["SearchContext"], bool],
        max_evals: Callable[[], int],
        max_gen: Callable[[], int],
        lb: Union[float, List[float]],
        ub: Union[float, List[float]],
        ndims: int,
    ):
        self._eval_fn = eval_fn
        self._stop_condition = stop_condition
        self._fit = float("inf")
        self._gfit = self._fit
        self._best = None
        self._trials = 0
        self._max_trials = 0
        self._num_evals = 0
        self._num_gen = 0
        self._max_gen = max_gen
        self._max_evals = max_evals
        self._ndims = ndims
        self._lb = [lb for _ in range(ndims)] if isinstance(lb, numbers.Number) else lb
        self._ub = [ub for _ in range(ndims)] if isinstance(ub, numbers.Number) else ub
        # if not stop_condition.valid():
        #     raise Exception("No stop condition definition set")
        if len(self._ub) != len(self._lb) or len(self._ub) != ndims:
            raise Exception(
                "Lower or Upper bounds size don't match with the number of dimensions"
            )

    def evaluate(self, solution: List[float]) -> float:
        fit = self._eval_fn(solution)
        if fit < self._fit:
            self._best = solution
            self._fit = fit
        self._num_evals += 1
        return fit

    def hasNext(self) -> bool:
        return not self._stop_condition(self)

    def next(self):
        if self._fit < self._gfit:
            self._trials = 0
        else:
            self._trials += 1
            self._max_trials = max(self._trials, self._max_trials)
        self._gfit = self._fit
        self._num_gen += 1
        

    def best(self) -> Tuple[List[float], float]:
        return self._best, self._fit

    def num_evals(self) -> int:
        return self._num_evals

    def num_gen(self) -> int:
        return self._num_gen
    
    def num_trials(self) -> int:
        return self._trials
    
    def max_trials(self) -> int: 
        return self._max_trials

    def max_evals(self) -> int:
        return self._max_evals()

    def max_gen(self) -> int:
        return self._max_gen()

    def ndims(self) -> int:
        return self._ndims

    def lb(self) -> List[float]:
        return self._lb

    def ub(self) -> List[float]:
        return self._ub

    def _get_bounds(
        lb: Union[float, List[float]], ub: Union[float, List[float]], ndims: int
    ) -> Tuple[List[float], List[float]]:
        _lb = [lb for _ in range(ndims)] if isinstance(lb, numbers.Number) else lb
        _ub = [ub for _ in range(ndims)] if isinstance(ub, numbers.Number) else ub
        if len(_ub) != len(_lb) or len(_ub) != ndims:
            raise f"Lower or Upper bounds size don't match with the number of dimensions"
        return _lb, _ub

@dataclass
class GenerationContext:
    population: Population
    parent: SearchContext
    picker: Picker 
    archive: Optional[Archive] = None
    previous: Optional["GenerationContext"] = None
    popsize: int = field(init=False)
    picker: Picker = field(init=False)

    def __post_init__(self):
        self.popsize = self.population.size()
        self.picker = DefaultPicker(
            individuals=self.population.get(),
            ranking=self.population.ranking(),
            archived=self.archive.all() if self.archive else []
        )

    def evaluate(self, solution: List[float]) -> float:
        return self.parent.evaluate(solution)


@dataclass
class UpdateContext:
    individual: Individual
    neighbors: List[Individual]
    ranking: List[int]  # InitVar[List[int]]
    parent: GenerationContext
    picker_impl: InitVar["PickerCallback"]
    neighborhood_size: int = field(init=False)
    picker: Picker = field(init=False)

    def __post_init__(self, picker_impl: "PickerCallback"):
        self.neighborhood_size = len(self.neighbors)
        self.picker = picker_impl(self) 


@dataclass
class SelectionContext:
    individual: Individual
    trial: Individual
    parent: GenerationContext


InitPopulationCallback = Callable[[SearchContext], Population]
ResizePopulationCallback = Callable[[Population, SearchContext], None]
NeighborhoodCallback = Callable[[GenerationContext], Dict[int, List[int]]]
MutationCallback = Callable[[UpdateContext], List[float]]
ReparationCallback = Callable[[UpdateContext, List[float]], List[float]]
SelectionCallback = Callable[[SelectionContext], bool]
ConvergenceBoosterConditionCallback = Callable[[GenerationContext], bool]
ConvergenceBoosterCallback = Callable[[GenerationContext, Individual], Tuple[List[float], float]]
PickerCallback = Callable[[UpdateContext], Picker]
CrossoverCallback = Callable[[UpdateContext, List[float]], List[float]]
FilterNeighborsCallback = Optional[Callable[[GenerationContext], List[Individual]]]
FitnessFunction = Callable[[List[float]], float]
Bounds = Union[float, List[float]]
NumDimensions = int