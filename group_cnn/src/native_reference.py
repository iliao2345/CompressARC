"""Comparison layers from the retained repository source package."""
from pathlib import Path
from types import SimpleNamespace
import sys

source_root = Path(__file__).resolve().parents[2] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from compressarc.model import multitensor_systems
from compressarc.layers import helper
from compressarc.layers.shift.primitives import ShiftPrimitives, shift_, diagonal_shift_
from compressarc.layers.direction_share.primitives import DirectionSharePrimitives

reference = SimpleNamespace(
    multitensor_systems=multitensor_systems,
    make_directional_layer=helper.make_directional_layer,
    apply_residual=helper.apply_residual,
    only_do_for_certain_shapes=helper.only_do_for_certain_shapes,
    shift_=shift_,
    diagonal_shift_=diagonal_shift_,
    shift=ShiftPrimitives(multitensor_systems.multify),
    direction_share=DirectionSharePrimitives(multitensor_systems.multify),
)
