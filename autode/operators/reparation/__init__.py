from typing import List
from autode.core import UpdateContext, ReparationCallback
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _repair_middle(solution, parent, lower, upper):
    repaired = solution.copy()
    for i in range(len(solution)):
        if solution[i] < lower[i]:
            repaired[i] = (parent[i] + lower[i]) / 2
        elif solution[i] > upper[i]:
            repaired[i] = (parent[i] + upper[i]) / 2
    return repaired

def clip()->ReparationCallback:
    def callback(ctx: UpdateContext, solution: List[float]):
        lb = ctx.parent.parent.lb()
        ub = ctx.parent.parent.ub()
        return np.clip(solution, lb, ub)
    return callback

def middle_from_parent()->ReparationCallback:
    def callback(ctx: UpdateContext, solution: List[float]):
        search = ctx.parent.parent
        lb, ub = search.lb(), search.ub()
        parent = ctx.individual.pos
        return _repair_middle(solution, parent, np.asarray(lb), np.asarray(ub))
    return callback
