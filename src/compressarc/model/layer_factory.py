"""Factories for selecting concrete layer implementations."""

from functools import partial

from ..layers.cummax.morphological import MorphologicalMax
from ..layers.cummax.optimisation import MorphologicalMax as OptimisedMax
from ..layers.shift.primitives import ShiftPrimitives
from ..layers.shift.layer import TiedShiftLayer
from ..layers.shift.morphological import UnfoldTiedDirectionalConv
from ..layers.shift.D4_morphological import D4DirectionalConv
from ..layers.cummax.layer import LSELayer
from ..layers.cummax.primitives import CummaxPrimitives
from ..layers.direction_share.primitives import DirectionSharePrimitives
from ..layers.direction_share.vectorized import DirectionShareVectorized
from ..layers.direction_share.packed import DirectionShareVectorizedPacked
from ..layers.direction_share.morphological import D4DirectionShareLayer


def create_triton_cummax(**kwargs):
    """Import the optional CUDA backend only when its YAML selects it."""
    try:
        from ..layers.cummax.triton_op import TritonMorphologicalMax
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError(
                "Triton cummax requires Triton on Linux with a CUDA GPU; "
                "install src/requirements.txt in that environment."
            ) from error
        raise
    return LSELayer(model_type=TritonMorphologicalMax, **kwargs)


def create_triton_hard_cummax(**kwargs):
    """Hard primitive acceleration; separate from the trainable Triton LSE."""
    try:
        from ..layers.cummax.triton_hard import TritonHardCummax
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError("Hard Triton cummax requires Triton on Linux with CUDA.") from error
        raise
    return TritonHardCummax(**kwargs)


def create_triton_hard_grouped_cummax(**kwargs):
    try:
        from ..layers.cummax.triton_hard_grouped import TritonHardCummaxGrouped
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError("Grouped hard cummax requires Triton on Linux with CUDA.") from error
        raise
    return TritonHardCummaxGrouped(**kwargs)


def create_triton_shift(**kwargs):
    try:
        from ..layers.shift.triton_op import TritonTiedDirectionalConv
    except ModuleNotFoundError as error:
        if error.name == "triton":
            raise RuntimeError("Triton shift requires Linux CUDA FP32 and Triton; install src/requirements.txt.") from error
        raise
    return TiedShiftLayer(model_type=TritonTiedDirectionalConv, **kwargs)

class LayerFactory:
    """Construct configured layer implementations from a small registry.

    New implementations can be registered under a name and selected from model
    configuration without changing the model's forward-pass structure.
    """

    def __init__(self):
        self._implementations = {
            "shift": {
                "primitives": ShiftPrimitives, "c4_tied_conv": TiedShiftLayer,
                "d4_tied_conv": partial(TiedShiftLayer, model_type=D4DirectionalConv),
                "tied_conv": TiedShiftLayer,  # Legacy spelling: same C4 implementation.
                "unfold": partial(TiedShiftLayer, model_type=UnfoldTiedDirectionalConv),
                "triton": create_triton_shift,
            },
            "cummax": {
                "primitives": CummaxPrimitives,
                "triton_hard": create_triton_hard_cummax,
                "triton_hard_grouped": create_triton_hard_grouped_cummax,
                "d4": partial(LSELayer, model_type=MorphologicalMax),
                "optimised": partial(LSELayer, model_type=OptimisedMax),
                "triton": create_triton_cummax,
            },
            "direction_share": {"primitives": DirectionSharePrimitives, "vectorized": DirectionShareVectorized, "vectorized_packed": DirectionShareVectorizedPacked, "d4": D4DirectionShareLayer},
        }

    def register(self, layer_name, implementation_name, implementation):
        """Register a layer class for a layer/implementation name pair."""
        if not callable(implementation):
            raise TypeError("implementation must be a class or callable constructor")
        self._implementations.setdefault(layer_name, {})[implementation_name] = implementation

    def create(self, layer_name, implementation_name, **kwargs):
        """Construct the requested implementation, reporting valid choices on error."""
        implementations = self._implementations.get(layer_name, {})
        try:
            implementation = implementations[implementation_name]
        except KeyError as exc:
            choices = ", ".join(sorted(implementations)) or "none registered"
            raise ValueError(
                f"Unknown {layer_name!r} implementation {implementation_name!r}; "
                f"available choices: {choices}"
            ) from exc
        return implementation(**kwargs)

    def create_direction_share(self, implementation_name="primitives", **kwargs):
        """Construct the selected direction-sharing implementation."""
        return self.create("direction_share", implementation_name, **kwargs)

    def create_cummax(self, implementation_name="primitives", **kwargs):
        """Construct the selected cummax implementation."""
        return self.create("cummax", implementation_name, **kwargs)

    def create_shift(self, implementation_name="primitives", **kwargs):
        """Construct the selected shift implementation."""
        return self.create("shift", implementation_name, **kwargs)