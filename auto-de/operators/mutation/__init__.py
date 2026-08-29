from typing import List, Union, Callable
import numpy as np
from core import Param, UpdateContext, MutationCallback, Individual
from .cma import cma


def difference_vector(
    orig: List[Individual], dest: List[Individual], f: float
) -> List[float]:
    ndims = len(orig[0].pos)
    res = np.zeros(ndims)
    for i in range(len(orig)):
        res += f * (orig[i].pos - dest[i].pos)
    return res


def prepend_individual(
    individuals: List[Individual], individual: Individual
) -> List[Individual]:
    return np.insert(individuals, 0, individual)


def rand_y(
    f: Param, diff_size: Union[int, Param] = 1, include_archive: bool = False
) -> MutationCallback:
    _diff_size = diff_size if callable(diff_size) else lambda: diff_size

    def callback(ctx: UpdateContext) -> List[float]:
        y = _diff_size()
        a, b, c = np.split(
            ctx.picker.get_random(size=y * 2 + 1, include_archive=include_archive),
            [1, y + 1],
        )
        return a[0].pos + difference_vector(b, c, f())

    return callback


def best_y(
    f: Param, diff_size: Union[int, Param] = 1, include_archive: bool = False
) -> MutationCallback:
    _diff_size = diff_size if callable(diff_size) else lambda: diff_size

    def callback(ctx: UpdateContext) -> List[float]:
        y = _diff_size()
        best = ctx.picker.get_best()
        b, c = np.split(
            ctx.picker.get_random(
                size=y * 2,
                include_archive=include_archive,
                ignore=[best],
            ),
            [y],
        )
        return best.pos + difference_vector(b, c, f())

    return callback


def current_to_rand_y(
    f: Param, diff_size: Union[int, Param] = 1, include_archive: bool = False
) -> MutationCallback:
    _diff_size = diff_size if callable(diff_size) else lambda: diff_size

    def callback(ctx: UpdateContext) -> List[float]:
        y = _diff_size()
        current = ctx.individual
        a, b, c = np.split(
            ctx.picker.get_random(
                size=y * 2 + 1, include_archive=include_archive, ignore=[current]
            ),
            [1, y + 1],
        )
        return current.pos + difference_vector(
            prepend_individual(b, a), prepend_individual(c, current), f()
        )

    return callback


def rand_to_best_y(
    f: Param,
    p: Union[int, Param] = 0.5,
    diff_size: Union[int, Param] = 1,
    include_archive: bool = False,
) -> MutationCallback:
    _diff_size = diff_size if callable(diff_size) else lambda: diff_size
    _p = p if callable(p) else lambda: p

    def callback(ctx: UpdateContext) -> List[float]:
        y = _diff_size()
        current = ctx.individual
        best = ctx.picker.get_best()
        a, b, c = np.split(
            ctx.picker.get_random(
                size=y * 2 + 1, include_archive=include_archive, ignore=[current, best]
            ),
            [1, y + 1],
        )
        l = _p()
        return (
            l * current.pos
            + (1 - l) * best.pos
            + difference_vector(
                prepend_individual(b, a), prepend_individual(c, current), f()
            )
        )

    return callback


def current_to_best_y(
    f: Param, diff_size: Union[int, Param] = 1, include_archive: bool = False
) -> MutationCallback:
    _diff_size = diff_size if callable(diff_size) else lambda: diff_size

    def callback(ctx: UpdateContext) -> List[float]:
        y = _diff_size()
        current = ctx.individual
        best = ctx.picker.get_best()
        b, c = np.split(
            ctx.picker.get_random(
                size=y * 2, include_archive=include_archive, ignore=[current, best]
            ),
            [y],
        )
        return current.pos + difference_vector(
            prepend_individual(b, best), prepend_individual(c, current), f()
        )

    return callback


def current_to_pbest(
    f: Param,
    p: Union[int, Param] = 0.05,
    include_archive: bool = False,
) -> MutationCallback:
    _p = p if callable(p) else lambda: p

    def callback(ctx: UpdateContext) -> List[float]:
        current = ctx.individual
        pbest = ctx.picker.get_random_best(p=_p())
        b = ctx.picker.get_random(ignore=[current, pbest])[0]
        c = ctx.picker.get_random(
            include_archive=include_archive, ignore=[current, pbest, b]
        )[0]
        _f = f()
        return current.pos + _f * (pbest.pos - current.pos) + _f * (b.pos - c.pos)

    return callback


def current_to_pbestw(
    f: Param, p: Union[int, Param] = 0.05, include_archive: bool = False
) -> MutationCallback:
    _p = p if callable(p) else lambda: p

    def callback(ctx: UpdateContext) -> List[float]:
        current = ctx.individual
        pbest = ctx.picker.get_random_best(p=_p())
        b = ctx.picker.get_random(ignore=[current, pbest])[0]
        c = ctx.picker.get_random(
            include_archive=include_archive, ignore=[current, pbest, b]
        )[0]
        _f = f()
        _fw = _f
        max_evals = ctx.parent.parent.max_evals()
        num_evals = ctx.parent.parent.num_evals()
        if num_evals < 0.2 * max_evals:
            _fw *= 0.7
        elif num_evals < 0.4 * max_evals:
            _fw *= 0.8
        else:
            _fw *= 1.2
        return current.pos + _fw * (pbest.pos - current.pos) + _f * (b.pos - c.pos)

    return callback


def current_to_ord_best(f: Param, include_archive: bool = False) -> MutationCallback:
    def callback(ctx: UpdateContext) -> List[float]:
        current = ctx.individual
        individuals = ctx.picker.get_random(size=2,ignore=[current])
        individuals += ctx.picker.get_random(
            include_archive=include_archive, ignore=[current] + individuals
        )
        obest, omedian, oworse = sorted(individuals, key=lambda i: i.fit)
        return current.pos + difference_vector([obest, omedian], [current, oworse], f())

    return callback


def current_to_ord_pbest(
    f: Param, p: Union[int, Param] = 0.05, include_archive: bool = False
) -> MutationCallback:
    _p = p if callable(p) else lambda: p

    def callback(ctx: UpdateContext) -> List[float]:
        current = ctx.individual
        pbest = ctx.picker.get_random_best(p=_p())
        omedian = ctx.picker.get_random(ignore=[current, pbest])[0]
        oworse = ctx.picker.get_random(
            include_archive=include_archive, ignore=[current, pbest, omedian]
        )[0]
        if omedian.fit > oworse.fit:
            b = omedian
            omedian = oworse
            oworse = b
        # omedian, oworse = ctx.picker.get_random_ordered(
        #     size=2, include_archive=include_archive
        # )
        return current.pos + difference_vector([pbest, omedian], [current, oworse], f())

    return callback


def rand_to_pbest(
    f: Param,
    p: Union[int, Param] = 0.05,
    include_archive: bool = False,
) -> MutationCallback:
    _p = p if callable(p) else lambda: p

    def callback(ctx: UpdateContext) -> List[float]:
        current = ctx.individual
        pbest = ctx.picker.get_random_best(p=_p())
        b = ctx.picker.get_random(ignore=[current, pbest])[0]
        c = ctx.picker.get_random(
            include_archive=include_archive, ignore=[current, pbest, b]
        )[0]
        _f = f()
        return _f * b.pos + (pbest.pos - c.pos)

    return callback


# def current_to_pbest_fo(
#     f: Param,
#     p: Union[int, Param] = 0.05,
#     include_archive: bool = False,
# ) -> MutationCallback:
#     _p = p if callable(p) else lambda: p

#     def callback(ctx: UpdateContext) -> List[float]:
#         current = ctx.picker.get_by_id(ctx.individual.id)
#         pbest = ctx.picker.get_random_best(p=_p())
#         b = ctx.picker.get_random(ignore=[current, pbest])[0]
#         c = ctx.picker.get_random(
#             include_archive=include_archive, ignore=[current, pbest, b]
#         )[0]
#         _f = f()
#         return current.pos + _f * (pbest.pos - current.pos) + _f * (b.pos - c.pos)

#     return callback


# def current_to_pbest_y(f: Param, diff_size: Union[int, Param] = 1, p: Union[int, Param] = 0.05)->MutationCallback:
#     _p = p if isinstance(p, Param) else lambda: p
#     def callback(ctx: UpdateContext)->MutationCallback:
#         current = ctx.individual
#         pbest = ctx.picker.get_random_best(p=_p())
#         r1 = ctx.picker.get_random()[0]
#         r2 = ctx.picker.get_random(include_archive=True)[0]
#         return add_step(current, difference_vector([pbest, r1], [current, r2], f()))
#         #_f = f()
#         #return current.pos + _f * (pbest.pos - current.pos) + _f * (r1.pos - r2.pos)

#     return callback
