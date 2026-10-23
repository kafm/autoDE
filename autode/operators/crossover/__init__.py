from typing import List
from autode.core import (
    Param,
    UpdateContext,
    CrossoverCallback,
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
            ctx.random.uniform(size=ndims) < _cr, mx, ctx.individual.pos
        ) 
        j = ctx.random.integers(low=0, high=ndims)
        pos_new[j] = mx[j]
        return pos_new

    return callback


def binomial_with_cauchy_perturbation(
    cr: Param, scale: float = 0.1, jump_rate: float = 0.2
) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        R = ctx.random.integers(low=0, high=ndims)
        y = ctx.individual.pos.copy()
        for i in range(ndims):
            ri = ctx.random.random()
            if ri < _cr or i == R:
                y[i] = mx[i]
            elif ctx.random.random() < jump_rate:
                y[i] = ss.cauchy.rvs(loc=y[i], scale=scale, random_state=ctx.random)
        return y

    return callback


def exponential(cr: Param) -> CrossoverCallback:
    def callback(ctx: UpdateContext, mx: List[float]):
        _cr = cr()
        ndims = len(mx)
        y = ctx.individual.pos.copy()
        n = ctx.random.integers(low=0, high=ndims)
        j = n
        for _ in range(ndims):
            y[j] = mx[j]
            if ctx.random.random() < _cr:
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
        n = ctx.random.integers(low=0, high=ndims)
        mutation_enabled = True
        k = 0
        while k < ndims:
            if mutation_enabled:
                while k < ndims and ctx.random.random() < cr_m:
                    y[n] = mx[n]
                    n = (n + 1) % ndims
                    k += 1
                mutation_enabled = False
            else:
                while k < ndims and ctx.random.random() < cr_s:
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
        ctx.random.shuffle(s)
        for j in s:
            y[j] = mx[j]
            if ctx.random.random() < _cr:
                break
        return y

    return callback
