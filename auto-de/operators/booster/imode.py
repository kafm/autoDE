from typing import Tuple, List
from dataclasses import replace
from core import GenerationContext, Individual
from .sqp import sqp 
import numpy as np

class ImodeConvergenceBooster:
    def __init__(self, pmin: float = .0001, pmax: float = .1, piter: float = 0.02, pstart: float = .85):
        self._pmin = pmin
        self._pmax = pmax
        self._p = pmax
        self._piter = piter
        self._pstart = pstart
        self._sqp = sqp(piter)


    def applies(self, ctx: GenerationContext)->bool:
        max_evals = ctx.parent.max_evals()
        num_evals = ctx.parent.num_evals()
        return num_evals >= max_evals*self._pstart and np.random.uniform() <= self._p
    
    def boost(self,ctx: GenerationContext, individual: Individual)->Tuple[List[float], float]:
        pos,fit = self._sqp(ctx, individual)
        self._p = self._pmax if fit < individual.fit else self._pmin
        return pos, fit
            




