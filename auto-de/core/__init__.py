from .context import *
from .optimizer import UpdateMethod, ConvergenceBooster, Optimizer as _Optimizer
from .params import *
from .stop_condition import StopCondition
from .events import PubSub
from .ensemble import Ensemble as ensemble
from .population import Picker, MAX_PICKER_ITER

def create_optimizer(
    init_strategy: InitPopulationCallback,
    update_strategy: List["UpdateMethod"],
    stop_condition: StopCondition,
    events: PubSub,
    neighborhood_strategy: Optional[NeighborhoodCallback] = None,
    resize_strategy: Optional[ResizePopulationCallback] = None,
    params: Optional[Params] = None,
    archive: Optional[Archive] = None,
) -> Callable[
    [FitnessFunction, Tuple[Bounds, Bounds], NumDimensions], Tuple[List[float], float]
]:
    return _Optimizer(
        init_strategy=init_strategy,
        update_strategy=update_strategy,
        neighborhood_strategy=neighborhood_strategy,
        resize_strategy=resize_strategy,
        stop_condition=stop_condition,
        events=events,
        params=params,
        archive=archive,
    ).search
