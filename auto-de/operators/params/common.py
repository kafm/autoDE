from typing import Optional, List
import scipy.stats as ss
from scipy.spatial import distance
import numpy as np
from core import ParamValue, Individual, MAX_PICKER_ITER


def normalize_values(w: List[float]) -> List[float]:
    return np.divide(w, np.sum(w))

def lehmer_mean(values: List[float], weights: Optional[List[float]] = None) -> float:
    if weights is None:
        return np.divide(np.sum(np.square(values)), np.sum(values))
    return np.divide(
        np.sum(np.multiply(weights, np.square(values))),
        np.sum(np.multiply(values, weights)),
    )

def manhattan_distance(x: List[float], y: List[float]) -> float:
    return distance.cityblock(x, y)

def generate_cauchy_bt_0_1(loc=0, scale=1, size: Optional[int] = None) -> ParamValue:
    value_list = []
    max_generation = size if size else 1
    num_iter = 0
    while len(value_list) < max_generation:
        value = ss.cauchy.rvs(loc=loc, scale=scale)
        if value > 0:
            value_list.append(value if value < 1 else 1)
        else:
            num_iter += 1
            if num_iter > MAX_PICKER_ITER:
                raise Exception("Maximum iteraction reached without generating cauchy_bt_0_1")
    return value_list if size else value_list[0]


def generate_cauchy(loc=0, scale=1, size: Optional[int] = None) -> ParamValue:
    return (
        ss.cauchy.rvs(loc=loc, scale=scale)
        if size is None
        else [ss.cauchy.rvs(loc=loc, scale=scale) for _ in range(size)]
    )
