import torch
import pytest

from src import shift_, diagonal_shift_, TiedDirectionalConv

@pytest.fixture
def pulse():
    x = torch.zeros((5, 5))
    x[2, 2] = 1.0
    return x


def test_shift_dim0(pulse):
    expected = torch.zeros((5, 5))
    expected[3, 2] = 1.0

    output = shift_(pulse, dim=0, masks=None)

    assert torch.equal(output, expected)



def test_tied_directional_conv(pulse):
    model = TiedDirectionalConv()

    # [B, orbit, direction, H, W]
    # orbit 0 = cardinal
    # orbit 1 = diagonal
    x = torch.zeros((1, 2, 4, 5, 5))

    # pulse in all 8 directional channels
    x[0, :, :, 2, 2] = 1.0

    output = model(x)

    # -----------------------------------
    # Helper for negative shift
    # -----------------------------------
    def opposite_shift(x, dim):
        x = torch.flip(x, dims=[dim])
        x = shift_(x, dim=dim, masks=None)
        return torch.flip(x, dims=[dim])

    # -----------------------------------
    # Cardinal reference
    # -----------------------------------
    cardinal = [
        opposite_shift(pulse, dim=0),         # up
        opposite_shift(pulse, dim=1),         # left
        shift_(pulse, dim=0, masks=None),     # down
        shift_(pulse, dim=1, masks=None),     # right
    ]

    # -----------------------------------
    # Diagonal reference
    # -----------------------------------

    # down-right
    down_right = diagonal_shift_(
        pulse,
        dim1=0,
        dim2=1,
        masks=None,
        shift_amount=1,
    )

    # up-right
    up_right = opposite_shift(pulse, dim=0)
    up_right = shift_(up_right, dim=1, masks=None)

    # up-left
    up_left = diagonal_shift_(
        pulse,
        dim1=0,
        dim2=1,
        masks=None,
        shift_amount=-1,
    )

    # down-left
    down_left = shift_(pulse, dim=0, masks=None)
    down_left = opposite_shift(down_left, dim=1)

    diagonal = [
        down_right,
        up_right,
        up_left,
        down_left,
    ]

    # -----------------------------------
    # Check each individual direction
    # -----------------------------------
    for d, expected in enumerate(cardinal):
        assert torch.equal(
            output[0, 0, d],
            expected
        ), f"Cardinal direction {d} failed"

    for d, expected in enumerate(diagonal):
        assert torch.equal(
            output[0, 1, d],
            expected
        ), f"Diagonal direction {d} failed"

    # -----------------------------------
    # Stack into exactly model's shape
    # -----------------------------------
    cardinal = torch.stack(cardinal)    # [4, H, W]
    diagonal = torch.stack(diagonal)    # [4, H, W]

    expected = torch.stack([
        cardinal,
        diagonal,
    ])                                  # [2, 4, H, W]

    expected = expected.unsqueeze(0)    # [1, 2, 4, H, W]

    assert torch.equal(output, expected)

