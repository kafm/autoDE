from typing import List, Dict, Optional
from dataclasses import replace
from core import (
    Individual,
    GenerationContext,
    UpdateContext,
    PubSub,
    GENERATION_START_EVT,
    OPTIMIZATION_END_EVT,
    PickerCallback,
    Picker,
    MAX_PICKER_ITER,
)
import numpy as np
import math


def fode(events: PubSub, size: int = 6, rate: float = 0.995, greediness: float = 3) -> PickerCallback:
    fo = FractionalOrderReferences(size=size, rate=rate, k=greediness)
    unsub = []

    def unsub_all(*args, **kwargs):
        for u in unsub:
            u()

    unsub.append(events.subscribe(GENERATION_START_EVT, fo.update))
    unsub.append(events.subscribe(OPTIMIZATION_END_EVT, unsub_all))

    return lambda ctx: FodePicker(
        individuals=fo.get(ctx),
        ranking=fo.ranking(ctx),
        archived=fo.get_archived(),
        p=fo.get_p(),
    )


# 1202.6163671027402
class FractionalOrderReferences:
    def __init__(self, size: int, rate: float, k: float):
        self._size = size
        self._rate = rate
        self._k = k
        self._individuals: List[Individual] = []
        self._archived: List[Individual] = []
        self._ranking: List[Individual] = []
        self._p: List[float] = []
        self._individuals_map: Dict[int, List[List[float]]] = {}
        self._ranking_map: Dict[int, List[List[float]]] = {}
        self._archive_map: Dict[int, List[List[float]]] = {}
        self._ndims = None

    def update(self, ctx: GenerationContext):
        if self._ndims is None:
            self._ndims = ctx.parent.ndims()
        population = ctx.population
        individuals = []
        ranking = []
        for i in range(ctx.popsize):
            individuals.append(self._get_fo_individual(population.get_by_index(i)))
            ranking.append(self._get_fo_ranking(population.get_by_rank(i), i))
        self._individuals = population.get()
        self._ranking = [
            population.get_by_rank(i) for i in range(ctx.population.size())
        ]
        if ctx.archive:
            archived = ctx.archive.all()
            self._archived = [
                self._get_fo_archive(archived[i], i) for i in range(len(archived))
            ]
        self._update_p(population.ranking())

    def _update_p(self, ranking: List[int]):
        n = len(ranking)
        p = np.zeros(n)
        for i in range(n):
            p[ranking[i]] = self._k * (n - i) + 1
        self._p = p / np.sum(p)

    def _get_fo_ranking(self, individual: Individual, rank: int):
        if rank not in self._ranking_map:
            self._ranking_map[rank] = []
        foi: List[List[float]] = self._ranking_map[rank]
        foi.insert(0, individual.pos)
        len(foi) > self._size and foi.pop()
        return replace(individual, pos=self._get_fo_pos(foi))

    def _get_fo_individual(self, individual: Individual) -> Individual:
        if individual.index not in self._individuals_map:
            self._individuals_map[individual.index] = []
        foi: List[List[float]] = self._individuals_map[individual.id]
        foi.insert(0, individual.pos)
        len(foi) > self._size and foi.pop()
        # return individual
        return replace(individual, pos=self._get_fo_pos(foi))

    def _get_fo_archive(self, individual: Individual, index: int) -> Individual:
        if index not in self._archive_map:
            self._archive_map[index] = []
        foi: List[List[float]] = self._archive_map[index]
        foi.insert(0, individual.pos)
        len(foi) > self._size and foi.pop()
        return replace(individual, pos=self._get_fo_pos(foi))
        # return individual

    def _get_fo_pos(self, foi: List[List[float]]) -> List[float]:
        res = np.zeros((self._ndims,))
        rate = self._rate
        for i in range(len(foi)):
            ri = i + 1
            res += 1 / math.factorial(ri) * rate * foi[i]
            rate = rate * (ri - self._rate)
        return res

    # TODO to implement neighborhood definition. For every neighborhood a different fo must be implemented
    def get(self, ctx: UpdateContext) -> List[Individual]:
        if ctx.neighborhood_size == ctx.parent.popsize:
            return self._individuals
        raise NotImplementedError

    # TODO to implement neighborhood definition. For every neighborhood a different fo must be implemented
    def ranking(self, ctx: UpdateContext) -> List[Individual]:
        if ctx.neighborhood_size == ctx.parent.popsize:
            return self._ranking
        raise NotImplementedError

    def get_archived(self) -> List[Individual]:
        return self._archived

    def get_p(self) -> List[float]:
        return self._p


class FodePicker(Picker):
    def __init__(
        self,
        individuals: List[Individual],
        ranking: List[Individual],
        p: List[float],
        archived: List[Individual] = [],
    ):
        self._individuals = individuals
        self._size = len(individuals)
        self._archived = archived
        self._ranking = ranking
        self._p = p

    def get_best(self) -> Individual:
        return self._ranking[0]

    def get_k_best(self, k: int) -> List[Individual]:
        return self._ranking[:k]

    def get_random_best(self, p: int = 1) -> Individual:
        high = np.clip(int(self._size * p), 2, self._size)
        return self._ranking[np.random.randint(low=0, high=high)]

    def get_random(
        self,
        size: int = 1,
        include_archive: bool = False,
        ignore: Optional[List[Individual]] = [],
    ) -> List[Individual]:
        ignore_indices = [j.index for j in ignore]
        archive_size = len(self._archived)
        use_archive = include_archive and archive_size > 0
        candidates = (
            self._individuals + self._archived if use_archive else self._individuals
        )
        candidates_size = len(candidates)
        popindices = range(self._size)
        picked = []
        iter_cnt = 0
        while len(picked) < size:
            if iter_cnt > MAX_PICKER_ITER:
                raise Exception(
                    "Max iteraction reached without finding different individuals"
                )
            index = (
                np.random.randint(low=self._size, high=candidates_size)
                if use_archive and np.random.rand() < 0.5
                else np.random.choice(popindices, p=self._p)
            )
            if index not in ignore_indices:
                picked.append(candidates[index])
                ignore_indices.append(index)
                if (
                    len(ignore_indices) >= candidates_size
                ):  # restart ignore indices because we've reached all unique possibilities
                    ignore_indices = []
            iter_cnt += 1
        return picked

    def get_random_ordered(
        self, size: int = 1, include_archive: bool = False
    ) -> List[Individual]:
        candidates = self.get_random(size=size, include_archive=include_archive)
        return sorted(candidates, key=lambda i: i.fit)

    def get_median(self) -> Individual:
        return self._ranking[self._size // 2]

    def get_worse(self) -> Individual:
        return self._ranking[-1]

    def get_at(self, index: int) -> Individual:
        return self._individuals[index]

    def get_many_at(self, indices: List[int]) -> Individual:
        return [self._individuals[i] for i in indices]

    def get_by_id(self, id: int) -> Individual:
        for i in self._individuals:
            if i.id == id:
                return i

    def ranking(self, n: Optional[int] = None) -> List[int]:
        return [self._ranking[i].index for i in range(n)]
