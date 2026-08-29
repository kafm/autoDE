from typing import List, Tuple
from scipy.optimize import Bounds, minimize
from core import (
    MutationCallback,
    GenerationContext,
    Individual
)

#fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None
def sqp(piter: float = 0.02) -> MutationCallback:
    return SqpWrapper(piter)


class SqpWrapper:
    def __init__(self, piter: float):
        self._piter = piter
        self._bounds = None
        self._inited = False

    def __call__(self, ctx: GenerationContext, individual: Individual) -> Tuple[List[float], float]:
        lb = ctx.parent.lb()
        ub = ctx.parent.ub()
        max_evals = ctx.parent.max_evals()
        num_evals = ctx.parent.num_evals()
        fes = min(self._piter * max_evals, max_evals - num_evals)
        ev_wrapped = EvalFn(ctx.parent.evaluate, fes)
        try:
            minimize(ev_wrapped,
                individual.pos,
                tol=1e-8,
                method="L-BFGS-B",
                bounds=Bounds(lb=lb, ub=ub)#,
                #options={"maxfun": fes},
            )
        except StopIteration as e:
            pass
        return ev_wrapped.pos(),ev_wrapped.fit()

        
class EvalFn: 
    def __init__(self, fn, max_evals):
        self._fn = fn
        self._num_evals = 0
        self._max_evals = max_evals
        self._pos = None
        self._fit = float("inf")

    def __call__(self, x):
        if self._num_evals >= self._max_evals:
            raise StopIteration()
        self._num_evals += 1
        fit = self._fn(x)
        if fit < self._fit or self._pos is None:
            self._fit = fit
            self._pos = x
        return fit
    
    def pos(self)->List[float]:
        return self._pos
    
    def fit(self)->float:
        return self._fit