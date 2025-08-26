from .loss import SSIMLoss
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.naneu.nn.modules.tinyinr import TinyCudaINR
else:
    from ...common.importlib import LazyModule
    TinyInr = LazyModule("tinyinr.TinyCudaINR")