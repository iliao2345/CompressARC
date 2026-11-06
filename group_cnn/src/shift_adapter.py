"""Convolution primitives with the original CompressARC shift interface."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_shift_axis(x, dim, amount, pad_value=0):
    """Shift independent lines along any axis, preserving shape/device/dtype.
        args :
            x: input
            dim: axis to shift; for [E, C, H, W, F], H=2 and W=3
            amount: the number of pixel shift 
        
    """
    # x.shape == [ E, C, H, W, F ] F is the feature dim 
    lines = x.movedim(dim, -1) # lines.shape == [ E, C , W , F , H ] if  dim == 2 else dim == 3 [E, C, H, F, W ]
    # dim == 2: lines.shape == [E, C, W, F, H]
    # dim == 3: lines.shape == [E, C, H, F, W]
    length = lines.shape[-1]
    distance = abs(amount)
    if distance > length:
        raise ValueError("shift distance exceeds the axis length")
    if distance == 0:
        return x
    # conv1d uses cross-correlation: a left tap moves values right.
    kernel = x.new_zeros(1, 1, 2 * distance + 1) # kernel.shape == [1, 1, 2 * distance + 1]
    kernel[0, 0, 0 if amount > 0 else -1] = 1
    flat = lines.reshape(-1, 1, length)
    padded = F.pad(flat, (distance, distance), value=pad_value)
    shifted = F.conv1d(padded, kernel)
    return shifted.reshape(lines.shape).movedim(-1, dim) # [E*C*W*F, 1, H] â†’ [E, C, W, F, H] â†’ [E, C, H, W, F]


def conv_shift_(x, dim, masks=None):
    """Same arguments and zero-padded output as layers.shift_."""
    return _conv_shift_axis(x, dim, 1)


def conv_diagonal_shift_(x, dim1, dim2, masks=None, shift_amount=1, pad_value=0):
    """Same arguments and output as layers.diagonal_shift_."""
    for dim in (dim1, dim2):
        x = _conv_shift_axis(x, dim, shift_amount, pad_value)
    return x


def build_shift_layer(reference_layers, cardinal_fn=conv_shift_,
                      diagonal_fn=conv_diagonal_shift_):
    """Build shift(x, residual_weights, masks, **kwargs) using original wrappers.

    Pass the imported CompressARC layers module. Optional primitives let you
    verify your own network inside the exact same surrounding operations.
    """
    directional = reference_layers.make_directional_layer(cardinal_fn, diagonal_fn)
    def residual(dims, x, weights, masks, **kwargs):
        return reference_layers.apply_residual(
            x, weights, lambda projected: directional(dims, projected, masks), **kwargs
        )
    filtered = reference_layers.only_do_for_certain_shapes(
        (1, 1, 1, 1, 1), (1, 0, 1, 1, 1)
    )(residual)
    return reference_layers.multitensor_systems.multify(filtered)


class CompressARCShift(nn.Module):
    """Full shift adapter; residual weights are supplied by the caller.

    Input/output are MultiTensor objects with identical component shapes.
    Convolution kernels are fixed; gradients flow into inputs and weights.
    """
    def __init__(self, reference_layers):
        super().__init__()
        self._shift = build_shift_layer(reference_layers)

    def forward(self, x, residual_weights, masks, *, use_bias=False,
                pre_norm=False, post_norm=False):
        return self._shift(x, residual_weights, masks, use_bias=use_bias,
                           pre_norm=pre_norm, post_norm=post_norm)

def tied_directional_shift(x, masks, model):
    """Map [E, (C), D, H, W, F] to TiedDirectionalConv and back.

    Model cardinal order: up, left, down, right.
    Model diagonal order: down-right, up-right, up-left, down-left.
    This reproduces the original wrapper's even-feature cardinal selection.
    """
    if x.ndim not in (5, 6) or x.shape[-4] != 8 or x.shape[-1] % 2:
        raise ValueError("Expected [E, (C), 8, H, W, even_features]")
    height, width = x.shape[-3:-1]
    valid = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
    valid = valid.reshape(x.shape[0], *([1] * (x.ndim - 4)), height, width, 1)
    masked = x * valid
    # For each original direction slot, its index in the flattened CNN orbit.
    movement_indices = (
        (2, 4, 3, 5, 0, 6, 1, 7),  # first feature half
        (0, 6, 1, 7, 2, 4, 3, 5),  # opposite movements
    )
    halves = []
    for half, mapping in enumerate(movement_indices):
        by_movement = [None] * 8
        for direction, movement in enumerate(mapping):
            # Cardinal directions use even features in BOTH output halves.
            feature_start = 0 if direction % 2 == 0 else half
            grid = masked[..., direction, :, :, feature_start::2]
            by_movement[movement] = grid.movedim(-1, -3)
        # [E, (C), F/2, 8, H, W] -> [E*(C)*F/2, 2, 4, H, W]
        packed = torch.stack(by_movement, dim=-3)
        shifted = model(packed.reshape(-1, 2, 4, height, width))
        unpacked = shifted.reshape(packed.shape)
        restored = [unpacked[..., movement, :, :].movedim(-3, -1)
                    for movement in mapping]
        halves.append(torch.stack(restored, dim=-4))
    return torch.cat(halves, dim=-1)


if __name__ == "__main__":


    generator = torch.Generator().manual_seed(42)


    from src.native_reference import reference

    n_examples = 2 
    n_colors = 3
    n_x = 3
    n_y = 5


    n_features = 4
    projected_features = n_features  # Square identity projections; width must be even.
    system = reference.multitensor_systems.MultiTensorSystem(
        n_examples, n_colors, n_x, n_y, task=None
    )
    x = system.make_multitensor()
    weights = system.make_multitensor()
    active_dims = {(1, 1, 1, 1, 1), (1, 0, 1, 1, 1)}

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    # make_multitensor() creates empty slots; fill them with actual tensors.
    for dims in system:
        x[dims] = rand(*system.shape(dims, extra_dim=n_features))
        if tuple(dims) in active_dims:
            weights[dims] = [
                [torch.eye(n_features, dtype=torch.float64), torch.zeros(projected_features, dtype=torch.float64)],
                [torch.eye(n_features, dtype=torch.float64), torch.zeros(n_features, dtype=torch.float64)],
            ]

    masks = torch.randint(
        0, 2, (n_examples, n_x, n_y, 2), generator=generator
    ).double()
    flags = dict(use_bias=False, pre_norm=False, post_norm=False)
    actual = CompressARCShift(reference)(x, weights, masks, **flags)
    expected = reference.shift(x, weights, masks, **flags)
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims])
    print("PASS: all output components match src/compressarc layers")

    # MultiTensor has no single .shape; each x[dims] is a tensor with a shape.
    print("\nComponent shapes (flags: E, C, D, H, W; features are always last):")
    for dims in system:
        print(f"dims={tuple(dims)}: x.shape={tuple(x[dims].shape)}, "
              f"output.shape={tuple(actual[dims].shape)}")

    # Inspect the full [E, C, D, H, W, F] component.
    # Change this to (1, 0, 1, 1, 1) to inspect the component without colors.
    inspect_dims = (1, 1, 1, 1, 1)
    torch.set_printoptions(precision=3, sci_mode=False, threshold=float("inf"))
    print(f"\nInspecting dims={inspect_dims}")
    print("x.shape:", x[inspect_dims].shape)
    print("x:\n", x[inspect_dims])
    for name, (weight, bias) in zip(("input projection", "output projection"),
                                    weights[inspect_dims]):
        print(f"\n{name} weight.shape: {weight.shape}")
        print("weight:\n", weight)
        print("bias.shape:", bias.shape)
        print("bias:\n", bias)
    print("\nmasks.shape:", masks.shape)
    print("masks:\n", masks)
    print("\noutput.shape:", actual[inspect_dims].shape)
    print("output:\n", actual[inspect_dims])

    print("\nShift branch only (output - x), with residual input removed:")
    print(actual[inspect_dims] - x[inspect_dims])

    # A simple column makes the opposite zero-padded edges easy to see.
    column = torch.arange(1, 5, dtype=torch.float64).reshape(4, 1)
    down = _conv_shift_axis(column, dim=0, amount=1)
    up = _conv_shift_axis(column, dim=0, amount=-1)
    print("\nSimple column: values shift; they do not reflect.")
    print(" Input   Down     Up")
    for source, down_value, up_value in zip(column[:, 0], down[:, 0], up[:, 0]):
        print(f"{source.item():6.0f} {down_value.item():6.0f} {up_value.item():6.0f}")

    # Full-layer direction 0: branch feature 0 and feature 2 both use input
    # feature 0, but shift it down and up respectively in the original wrapper.
    branch = actual[inspect_dims] - x[inspect_dims]
    source_grid = x[inspect_dims][0, 0, 0, :, :, 0]
    combined_mask = 1 - (1 - masks[0, :, :, 0]) * (1 - masks[0, :, :, 1])
    masked_grid = source_grid * combined_mask
    down_grid = branch[0, 0, 0, :, :, 0]
    up_grid = branch[0, 0, 0, :, :, 2]
    torch.testing.assert_close(down_grid, _conv_shift_axis(masked_grid, 0, 1))
    torch.testing.assert_close(up_grid, _conv_shift_axis(masked_grid, 0, -1))
    print("\nActual full-layer example: E=0, C=0, direction=0")
    print("Input feature 0:\n", source_grid)
    print("Combined mask:\n", combined_mask)
    print("Masked input feature 0:\n", masked_grid)
    print("Down: branch feature 0 (top row padded with zeros):\n", down_grid)
    print("Up: branch feature 2 (bottom row padded with zeros):\n", up_grid)
    print("PASS: both branch grids match shifts of the same masked input.")

    # Run the user's 2D convolution on the same identity-projected inputs.
    if __package__:
        from .group_cnn import TiedDirectionalConv
    else:
        from group_cnn import TiedDirectionalConv
    tied_model = TiedDirectionalConv().to(dtype=x[inspect_dims].dtype,
                                         device=x[inspect_dims].device)
    with torch.no_grad():
        for dims in active_dims:
            tied_branch = tied_directional_shift(x[dims], masks, tied_model)
            tied_output = x[dims] + tied_branch
            torch.testing.assert_close(tied_output, expected[dims])
            print(f"\nPASS: TiedDirectionalConv full output matches for dims={dims}")
        tied_branch = tied_directional_shift(x[inspect_dims], masks, tied_model)
    print("TiedDirectionalConv branch shape:", tied_branch.shape)
    print("TiedDirectionalConv down grid (E=0, C=0, D=0, feature=0):\n",
          tied_branch[0, 0, 0, :, :, 0])
    print("TiedDirectionalConv up grid (E=0, C=0, D=0, feature=2):\n",
          tied_branch[0, 0, 0, :, :, 2])
    print("Masking: combined input/output validity multiplies features before convolution.")
    print("Padding: convolution inserts numeric zeros at grid boundaries; no PAD token.")


    torch.testing.assert_close(tied_branch[0, 0, 0, :, :, 0], _conv_shift_axis(masked_grid, 0, 1))
    torch.testing.assert_close(tied_branch[0, 0, 0, :, :, 2], up_grid)
    print("test case pass ")
