from .fode import fode

from core import PickerCallback as _PickerCallback, DefaultPicker as _DefaultPicker


def default() -> _PickerCallback:
    return lambda ctx: _DefaultPicker(
        individuals=ctx.neighbors,
        ranking=ctx.ranking,
        archived=ctx.parent.archive.all() if ctx.parent.archive else [], 
    )
