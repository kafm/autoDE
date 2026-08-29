from dataclasses import dataclass, replace
from typing import Optional, List
import numpy as np
from core import Archive, Individual


def fixed(
    size: int, allow_resizing: bool = True, overflow_removal_method: str = "random"
):
    if overflow_removal_method == "random":
        return _ArchiveWithRandomRemoval(size, allow_resizing=allow_resizing)
    raise NotImplementedError(
        f"Archive overflow removal method {overflow_removal_method} not implemented."
    )


# Jade
@dataclass
class _ArchiveWithRandomRemoval(Archive):
    def __init__(self, maxsize: int, allow_resizing: bool = True):
        self._maxsize: Optional[int] = maxsize
        self._allow_resizing = allow_resizing
        self._items: List[Individual] = []

    def all(self) -> List[Individual]:
        return self._items

    def put(self, individuals: List[Individual]):
        for i in individuals:
            self._items.append(replace(i, improved=False, archived=True))
        self.resize()

    def resize(self, new_size: Optional[int] = None):
        if new_size and self._allow_resizing:
            self._maxsize = new_size
        if self._maxsize < len(self._items):
            self._items = np.random.choice(self._items, size=self._maxsize).tolist()
