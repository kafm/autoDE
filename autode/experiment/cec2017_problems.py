from typing import Dict, Callable, List
from dataclasses import dataclass
from opfunu.cec_based.cec2017 import *


@dataclass
class Benchmark:
    fn: Callable[[List[float]], float]
    bias: float
    lb: float = -100
    ub: float = 100

    def __call__(self, x) -> float:
        return self.fn(x)


def _wrap_benchmark(fn: CecBenchmark) -> Benchmark:
    return Benchmark(fn=fn.evaluate, bias=fn.f_bias)


def get_problems(ndims) -> Dict[str, Benchmark]:
    return {
        "F1": _wrap_benchmark(F12017(ndim=ndims)),
        "F2": _wrap_benchmark(F22017(ndim=ndims)),
        "F3": _wrap_benchmark(F32017(ndim=ndims)),
        "F4": _wrap_benchmark(F42017(ndim=ndims)),
        "F5": _wrap_benchmark(F52017(ndim=ndims)),
        "F6": _wrap_benchmark(F62017(ndim=ndims)),
        "F7": _wrap_benchmark(F72017(ndim=ndims)),
        "F8": _wrap_benchmark(F82017(ndim=ndims)),
        "F9": _wrap_benchmark(F92017(ndim=ndims)),
        "F10": _wrap_benchmark(F102017(ndim=ndims)),
        "F11": _wrap_benchmark(F112017(ndim=ndims)),
        "F12": _wrap_benchmark(F122017(ndim=ndims)),
        "F13": _wrap_benchmark(F132017(ndim=ndims)),
        "F14": _wrap_benchmark(F142017(ndim=ndims)),
        "F15": _wrap_benchmark(F152017(ndim=ndims)),
        "F16": _wrap_benchmark(F162017(ndim=ndims)),
        "F17": _wrap_benchmark(F172017(ndim=ndims)),
        "F18": _wrap_benchmark(F182017(ndim=ndims)),
        "F19": _wrap_benchmark(F192017(ndim=ndims)),
        "F20": _wrap_benchmark(F202017(ndim=ndims)),
        "F21": _wrap_benchmark(F212017(ndim=ndims)),
        "F22": _wrap_benchmark(F222017(ndim=ndims)),
        "F23": _wrap_benchmark(F232017(ndim=ndims)),
        "F24": _wrap_benchmark(F242017(ndim=ndims)),
        "F25": _wrap_benchmark(F252017(ndim=ndims)),
        "F26": _wrap_benchmark(F262017(ndim=ndims)),
        "F27": _wrap_benchmark(F272017(ndim=ndims)),
        "F28": _wrap_benchmark(F282017(ndim=ndims)),
        "F29": _wrap_benchmark(F292017(ndim=ndims)),
        # "F30": _wrap_benchmark(F302017(ndim=ndims), 3000.0),
    }
