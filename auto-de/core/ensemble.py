from typing import Callable, List, TypeVar, Optional
from core import (
    MutationCallback,
    CrossoverCallback,
)

T = TypeVar("T", MutationCallback, CrossoverCallback)


def blank_fn(*args, **kwargs):
    pass


class Ensemble:

    @staticmethod
    def either(
        a: T,
        b: T,
        condition: Callable,
    ) -> T:
        _a = blank_fn if a is None else a
        _b = blank_fn if b is None else b

        def callback(*args, **kwargs):
            return _a(*args, **kwargs) if condition() else _b(*args, **kwargs)

        return callback

    @staticmethod
    def choice(
        choices: List[T], p: Optional[Callable[[], Optional[List[float]]]] = None
    ) -> T:
        _p = blank_fn if p is None else p

        def callback(*args, **kwargs):
            return choices[_p()](*args, **kwargs)

        return callback
