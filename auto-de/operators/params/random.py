from typing import List, Callable, Optional, Dict, Tuple
import numpy as np
from core import (
    ParamGenerator,
    ParamValue,
    ParamPerformance,
    ParamValueLearned,
    Individual,
    PubSub,
    GENERATION_START_EVT,
    GenerationContext,
)
from scipy.spatial import distance


class Random:

    @staticmethod
    def uniform(low: float = 0, high: float = 1) -> Callable[[int], ParamGenerator]:
        return lambda id: ParamGenerator(
            id=id,
            initial_value=None,
            memorizable=False,
            get_strategy=lambda _: np.random.uniform(low=low, high=high),
        )

    @staticmethod
    def boolean(
        low: float = 0,
        high: float = 1,
        p: float = 0.5,
        # TODO instead of bellow to params, we should receive a learning_strategy as with range
        weighted: bool = False,
        learning_rate: float = 0.1,
    ) -> Callable[[int], ParamGenerator]:
        _low = min(low, high)
        _high = max(low, high)
        return lambda id: ParamGenerator(
            id=id,
            initial_value=p,
            memorizable=True,
            get_strategy=lambda p: True if np.random.uniform() < p else False,
            learning_strategy=(
                _BooleanLearning(_low, _high, learning_rate, weighted)
                if learning_rate and learning_rate > 0
                else None
            ),
        )

    @staticmethod
    def range(
        size: int,
        events: PubSub,
        pmin: float = 0.1,
        pmax: float = 0.9,
        p: Optional[List[float]] = None,
        learnable: bool = True,
    ) -> Callable[[int], ParamGenerator]:
        _p = p if p else [1 / size for _ in range(size)]
        r = _Range(np.arange(size), _p, events, pmin=pmin, pmax=pmax)
        return lambda id: ParamGenerator(
            id=id,
            initial_value=_p,
            memorizable=True,
            get_strategy=lambda _: r.get(),
            learning_strategy=r.learn if learnable else None,
        )


class _Range:
    def __init__(
        self,
        values: List[int],
        p: List[float],
        events: PubSub,
        pmin: float = 0.1,
        pmax: float = 0.9,
    ):
        self._values = values
        self._size = len(values)
        self._p = p
        self._pmin = pmin
        self._pmax = pmax
        self._generated = None
        self._gen_size = 0
        self._gen_index = -1
        self._budget = np.zeros((self._size,))
        self._unsub_gen_start = events.subscribe(
            GENERATION_START_EVT, self._restart_generation
        )

    def get(self) -> int:
        if self._generated is None:
            self._generated = self._generate_values()
        self._gen_index += 1
        return self._generated[self._gen_index]
    
    def _generate_values(self)->List[int]:
        allocation = np.array([]) if self._gen_size < self._size else self._values.copy()
        remaining = self._gen_size - len(allocation) 
        if remaining > 0: 
            remaining_allocation = np.random.choice(self._values, p=self._p, size=remaining)
            allocation = np.concatenate((allocation, remaining_allocation))
        np.random.shuffle(allocation)
        return allocation
        

    def _restart_generation(self, ctx: GenerationContext):
        self._gen_size = ctx.popsize
        self._gen_index = -1
        self._generated = None

    def learn(
        self, _: ParamValue, results: List[ParamPerformance]
    ) -> ParamValueLearned:
        best, inds = self._get_groups(results)
        dops = []
        total_dops = 0
        qrs = []
        total_qrs = sum([i.fit for i in best if i is not None])
        for i in range(self._size):
            if best[i]:
                dop = (
                    (1 / self._size) * sum([distance.euclidean(ind.pos, best[i].pos) for ind in inds[i]])
                )
                dops.append(dop)
                total_dops += dop
                qrs.append(1 - best[i].fit / total_qrs)
            else:
                dops.append(0)
                qrs.append(0)
        if total_dops == 0:
            total_dops = 1
        irvs = [qrs[i] + (dops[i] / total_dops) for i in range(self._size)]
        total_irvs = sum(irvs)
        p = [max(self._pmin, min(self._pmax, irv / total_irvs)) for irv in irvs]
        #print(p)
        self._p = np.divide(p, sum(p))
        return ParamValueLearned(value=p)

    def _get_groups(
        self, results: List[ParamPerformance]
    ) -> Tuple[List[Individual], List[List[Individual]]]:
        best: List[Individual] = [None for _ in range(self._size)]
        inds: List[List[Individual]] = [[] for _ in range(self._size)]
        indices: Dict[int, int] = {i: self._values[i] for i in range(self._size)}
        for result in results:
            if result.value in indices: #result.individual.improved and 
                index = indices[result.value]
                ind = result.individual
                inds[index].append(ind)
                if best[index] is None or ind.fit < best[index].fit:
                    best[index] = ind
        return best, inds

    def __del__(self):
        self._unsub_gen_start and self._unsub_gen_start()


class _BooleanLearning:
    def __init__(self, low: float, high: float, learning_rate: float, weighted: bool):
        self._low = low
        self._high = high
        self._learning_rate = learning_rate
        self._weighted = weighted

    def __call__(
        self, old_p: ParamValue, results: List[ParamPerformance]
    ) -> ParamValueLearned:
        p_true = 0
        total = 0
        for result in results:
            if result.individual.improved:
                delta = result.individual.improvement if self._weighted else 1
                if result.value:
                    p_true += delta
                total += delta
        p = p_true / total if total > 0 else 0
        p = np.clip(p, self._low, self._high)
        p = (1 - self._learning_rate) * old_p + self._learning_rate * p
        return ParamValueLearned(value=p)
