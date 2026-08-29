from typing import List
from core import UpdateContext, ReparationCallback
import numpy as np

def clip()->ReparationCallback:
    def callback(ctx: UpdateContext, solution: List[float]):
        lb = ctx.parent.parent.lb()
        ub = ctx.parent.parent.ub()
        return np.clip(solution, lb, ub)
    return callback

def middle_from_parent()->ReparationCallback:
    def callback(ctx: UpdateContext, solution: List[float]):
        lb = ctx.parent.parent.lb()
        ub = ctx.parent.parent.ub()
        ndims = ctx.parent.parent.ndims()
        parent = ctx.individual.pos
        rsolution = solution.copy()
        for i in range(ndims):
            if solution[i] < lb[i]:
                rsolution[i] = (parent[i] + lb[i])/2
            elif solution[i] > ub[i]:
                rsolution[i] = (parent[i] + ub[i])/2
        return rsolution
    return callback