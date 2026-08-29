from typing import List, Optional
from core import (
    Param,
    UpdateContext,
    CrossoverCallback,
    GenerationContext,
    events,
    GENERATION_START_EVT,
)
import scipy.stats as ss
import numpy as np


def all() -> CrossoverCallback:
    return lambda _, mx: mx


def binomial(cr: Param) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        pos_new = np.where(
            np.random.uniform(size=ndims) < _cr, mx, ctx.individual.pos
        )  # TODO check impact of <= or < in cr
        j = np.random.randint(low=0, high=ndims)
        pos_new[j] = mx[j]
        return pos_new

    return callback


def binomial_with_cauchy_perturbation(
    cr: Param, scale: float = 0.1, jump_rate: float = 0.2
) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        R = np.random.randint(low=0, high=ndims)
        y = ctx.individual.pos.copy()
        for i in range(ndims):
            ri = np.random.rand()
            if ri < _cr or i == R:
                y[i] = mx[i]
            elif np.random.rand() < jump_rate:
                y[i] = ss.cauchy.rvs(loc=y[i], scale=scale)
        return y

    return callback


def exponential(cr: Param) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        y = ctx.individual.pos.copy()
        n = np.random.randint(low=0, high=ndims)
        j = n
        for _ in range(ndims):
            y[j] = mx[j]
            if np.random.rand() < _cr:
                break
            j = (j + 1) % ndims
        return y

    return callback


def multiple_exponential(cr: Param, t: int = 10) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        y = ctx.individual.pos.copy()
        em = t * _cr
        es = t * (1 - _cr)
        cr_m = em / (em + 1)
        cr_s = es / (es + 1)
        n = np.random.randint(low=0, high=ndims)
        mutation_enabled = True
        k = 0
        while k < ndims:
            if mutation_enabled:
                while k < ndims and np.random.rand() < cr_m:
                    y[n] = mx[n]
                    n = (n + 1) % ndims
                    k += 1
                mutation_enabled = False
            else:
                while k < ndims and np.random.rand() < cr_s:
                    #y[n] = ctx.individual.pos[n]
                    n = (n + 1) % ndims
                    k += 1
                mutation_enabled = True
        return y

    return callback


def shuffled_exponential(cr: Param) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        y = ctx.individual.pos.copy()
        s = np.arange(ndims)
        np.random.shuffle(s)
        for j in s:
            y[j] = mx[j]
            if np.random.rand() < _cr:
                break
        return y

    return callback


class CovarianceMatrix:
    def __init__(self):
        self._value: List[List[float]] = None
        self._unsub_gen_start = events.subscribe(GENERATION_START_EVT, self.compute)

    def compute(self, ctx: GenerationContext):
        values = [i.pos for i in ctx.population.get()]
        C = np.cov(np.array(values).T)
        self._value = np.linalg.eigh(C)[1]

    def __call__(self):
        return self._value

    def __del__(self):
        self._unsub_gen_start()


def eigen_bin(cr: Param) -> CrossoverCallback:
    C = CovarianceMatrix()
    bin_cr = binomial(cr)

    def callback(ctx: UpdateContext, mx: List[float]):
        ndims = len(mx)
        xt = np.zeros(ndims)
        vt = np.zeros(ndims)
        target = ctx.individual.pos
        for j in range(ndims):
            for k in range(ndims):
                xt[j] += C()[k][j] * target[k]
                vt[j] += C()[k][j] * mx[k]

        ut = bin_cr(xt, vt)
        u = np.zeros(ndims)
        for j in range(ndims):
            for k in range(ndims):
                u[j] += C()[j][k] * ut[k]
        return u

    return callback
