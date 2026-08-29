from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import numpy as np
import warnings

MAX_PICKER_ITER = 1000

@dataclass
class Individual:
    pos: List[float]
    fit: float
    id: int
    index: int
    trials: int = 0
    improved: bool = True
    improvement: float = 0
    delta: Optional[List[float]] = None
    archived: bool = False


@dataclass
class Archive(ABC):

    @abstractmethod
    def all(self) -> List[Individual]:
        pass

    @abstractmethod
    def put(individuals: List[Individual]):
        pass

    @abstractmethod
    def resize(self, new_size: int):
        pass


class Population:
    def __init__(self, individuals: List[Individual] = []):
        self._individuals = individuals
        self._size = len(individuals)
        self._init_size = self._size
        self._ranking = self.rank(individuals)
        self._neighborhood = None

    def set_individuals(
        self, individuals: List[Individual], refactor_indices: bool = True
    ):
        self._size = len(individuals)
        self._individuals = (
            individuals
            if not refactor_indices
            else [replace(individuals[i], index=i) for i in range(self._size)]
        )
        self._ranking = self.rank(individuals)

    def replace_individual(self, individual: Individual):
        self._individuals[individual.index] = individual
        if self._individuals[individual.index].fit != individual.fit:
            self._ranking = self.rank(self._individuals)

    def set_neighborhood(self, neighbors_matrix: Optional[Dict[int, List[int]]] = None):
        if neighbors_matrix:
            neighborhoods = {}
            for i, neighbors in neighbors_matrix.items():
                individuals = [self._individuals[n] for n in neighbors]
                neighborhoods[i] = (individuals, self.rank(individuals))
            self._neighborhoods = neighborhoods

    def rank(self, individuals: List[Individual]) -> List[int]:
        return sorted(range(len(individuals)), key=lambda i: individuals[i].fit)

    def get_neighbors(self, id: int) -> Tuple[List[Individual], List[int]]:
        if self._neighborhood:
            return self._neighborhood[id]
        return self._individuals, self._ranking

    def get(self) -> List[Individual]:
        return self._individuals

    def get_by_index(self, index: int) -> Individual:
        return self._individuals[index]
    
    def get_by_rank(self, rank: int) -> Individual:
        return self._individuals[self._ranking[rank]]

    def size(self) -> int:
        return self._size

    def init_size(self) -> int:
        return self._init_size

    def ranking(self) -> List[int]:
        return self._ranking
    
    def get_ranked(self) ->List[Individual]: 
        return [self._individuals[i] for i in self._ranking]
    

    def has_improvements(self) -> bool:
        for i in self._individuals:
            if i.improved:
                return True
        return False


class Picker(ABC):

    def get_best(self) -> Individual:
        raise NotImplementedError

    def get_k_best(self, k: int) -> List[Individual]:
        raise NotImplementedError

    def get_random_best(self, p: int = 1) -> Individual:
        raise NotImplementedError

    def get_random(
        self,
        size: int = 1,
        include_archive: bool = False,
        ignore: Optional[List[Individual]] = [],
    ) -> List[Individual]:
        raise NotImplementedError

    def get_random_ordered(
        self, size: int = 1, include_archive: bool = False
    ) -> List[Individual]:
        raise NotImplementedError

    def get_median(self) -> Individual:
        raise NotImplementedError

    def get_worse(self) -> Individual:
        raise NotImplementedError

    def get_at(self, index: int) -> Individual:
        raise NotImplementedError

    def get_many_at(self, indices: List[int]) -> Individual:
        raise NotImplementedError
    
    def get_by_id(self, id: int) -> Individual:
        raise NotImplementedError

    def ranking(self, n: Optional[int] = None) -> List[int]:
        raise NotImplementedError
    

class DefaultPicker(Picker):
    def __init__(
        self,
        individuals: List[Individual],
        ranking: List[int],
        archived: List[Individual] = [],
    ):
        self._individuals = individuals
        self._size = len(individuals)
        self._archived = archived
        self._ranking = ranking

    def get_best(self) -> Individual:
        return self._individuals[self._ranking[0]]
    
    def get_k_best(self, k: int)-> List[Individual]:
        return [self._individuals[self._ranking[i]] for i in range(k)]

    def get_random_best(self, p: int = 1) -> Individual:
        high = np.clip(int(self._size * p), 2, self._size)
        return self._individuals[self._ranking[np.random.randint(low=0, high=high)]]

    # def get_random(
    #     self, 
    #     size: int = 1, 
    #     include_archive: bool = False, 
    #     ignore: Optional[List[Individual]] = []
    # )->List[Individual]:
    #     ignore_indices = [j.index for j in ignore]    
    #     candidates = self._individuals + self._archived if include_archive else self._individuals
    #     picked = []
    #     candidates_size = len(candidates)
    #     iter_cnt = 0
    #     while len(picked) < size:
    #         if iter_cnt > MAX_PICKER_ITER:
    #             #raise Exception(f"Max iteraction {iter_cnt} reached without finding {size} different individuals from pool of {candidates_size} ignoring {ignore_indices} and picked {len(picked)}")
    #             #Ignore indicies due to large number of attempts
    #             warnings.warn(f"Will pick random without ensuring uniqueness due to Max iteraction {iter_cnt} reached without finding {size} different individuals from pool of {candidates_size} ignoring {ignore_indices} and picked {len(picked)}")
    #             ignore_indices = []
    #         index = np.random.randint(low=0,high=candidates_size)
    #         candidate = candidates[index]
    #         if candidate.index not in ignore_indices:
    #             picked.append(candidate)
    #             ignore_indices.append(index)
    #             if len(ignore_indices) >= candidates_size: #restart ignore indices because we've reached all unique possibilities
    #                 ignore_indices = []
    #         iter_cnt += 1
    #     return picked

    def get_random(
        self, 
        size: int = 1, 
        include_archive: bool = False, 
        ignore: Optional[List[Individual]] = None
    )->List[Individual]:
        _ignore = [i for i in ignore] if ignore else []
        candidates = self._individuals + self._archived if include_archive else self._individuals
        picked = []
        candidates_size = len(candidates)
        iter_cnt = 0
        while len(picked) < size:
            if iter_cnt > MAX_PICKER_ITER:
                #raise Exception(f"Max iteraction {iter_cnt} reached without finding {size} different individuals from pool of {candidates_size} ignoring {ignore_indices} and picked {len(picked)}")
                #Ignore indicies due to large number of attempts
                warnings.warn(f"Will pick random without ensuring uniqueness due to Max iteraction {iter_cnt} reached without finding {size} different individuals from pool of {candidates_size} ignoring {len(_ignore)} and picked {len(picked)}")
                _ignore = []
            index = np.random.randint(low=0,high=candidates_size)
            candidate = candidates[index]
            if self._solution_is_different(candidate, _ignore):
                picked.append(candidate)
                _ignore.append(candidate)
                if len(picked) == size:
                    break
                if len(_ignore) >= candidates_size: #restart ignore indices because we've reached all unique possibilities
                    _ignore = []
            iter_cnt += 1
        return picked
    
    def _solution_is_different(self, individual: Individual, others: List[Individual])->bool:
        for other in others: 
            if not np.any(individual.pos - other.pos):
                #print(np.any(individual.pos - other.pos))
                return False 
        return True
    # def _solution_is_different(self, solutions: List[Individual], candidate: Individual)->bool:
    #     if len(solutions) == 0: 
    #         return True
    #     for solution in solutions:
    #         if (solution.archived or solution.id != candidate.id): # and np.any(solution.pos - candidate.pos):
    #             return True
    #     return False

    def get_random_ordered(
        self, size: int = 1, include_archive: bool = False
    ) -> List[Individual]:
        candidates = self.get_random(size=size, include_archive=include_archive)
        return sorted(candidates, key=lambda i: i.fit)

    def get_median(self) -> Individual:
        index = self._ranking[self._size // 2]
        return self._individuals[index]

    def get_worse(self) -> Individual:
        index = self._ranking[-1]
        return self._individuals[index]

    def get_at(self, index: int) -> Individual:
        return self._individuals[index]

    def get_many_at(self, indices: List[int]) -> Individual:
        return [self._individuals[i] for i in indices]
    
    def get_by_id(self, id: int) -> Individual:
        for i in self._individuals:
            if i.id == id:
                return i 
            
    def _get_by_id(self, id: int) -> Individual:
        for i in self._individuals:
            if id == i.id:
                return i

    def ranking(self, n: Optional[int] = None) -> List[int]:
        return self._ranking[:n] if n else self._ranking
