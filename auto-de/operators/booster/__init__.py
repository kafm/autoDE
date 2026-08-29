from .imode import ImodeConvergenceBooster as _ImodeConvergenceBooster
from core import ConvergenceBooster

def imode(pmin: float = .0001, pmax: float = .1, piter: float = 0.02)->ConvergenceBooster:
    booster = _ImodeConvergenceBooster(pmin=pmin, pmax=pmax,piter=piter)
    return ConvergenceBooster(
        method=booster.boost,
        condition=booster.applies
    )