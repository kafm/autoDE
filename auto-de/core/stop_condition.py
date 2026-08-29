from typing import Optional, Callable
from sys import maxsize as maxint
from .context import SearchContext


class StopCondition:
    def __init__(
        self,
        max_evals: Optional[int] = None,
        max_gen: Optional[int] = None,
        callback: Optional[Callable[[SearchContext], bool]] = None,
    ):
        self._max_gen = max_gen 
        self._max_evals = max_evals
        self._callback = callback
        self._is_valid = True if self._max_gen or self._max_evals or callback else False

    def valid(self) -> bool:
        return self._is_valid

    def infer_max_evals_from_popsize_if_not_set(self, popsize: int):
        if self._max_evals:
            return
        if self._max_gen:
            self._max_evals = self._max_gen * popsize
        else:
            self._max_evals = maxint
            self._max_gen = maxint

    def max_gen(self, popsize: Optional[int] = None) -> int:
        if self._max_gen:
            return self._max_gen
        if popsize and self._max_evals and self._max_evals < maxint:
             return self._max_evals / popsize
        return maxint

    def max_evals(self) -> int:
        return self._max_evals

    def __call__(self, ctx: SearchContext) -> bool:
        res = self._callback and self._callback(ctx)
        if not res:
            res = (
                self._max_gen
                and ctx.num_gen() >= self._max_gen
                or self._max_evals
                and ctx.num_evals() >= self._max_evals
            )
        return res
