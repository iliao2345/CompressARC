# Layer organization

The `layers` package contains the ARC compressor operations. The model owns trainable weights and passes them to the layer calls.

## Shared operations

`helper.py` provides normalization, affine projections, residual application, fixed softmax and SiLU nonlinear layers, shape filtering, and the directional wrapper used by shift and cummax.

`share/primitives.py` contains `MultitensorShare`, with the fixed `share_up`, `share_down`, and shared communication implementation.

## Configurable layer families

`model/layer_factory.py` selects implementations for the layer families that are intended to vary:

- `shift/primitives.py`: `ShiftPrimitives` and its cardinal/diagonal shift functions.
- `cummax/primitives.py`: `CummaxPrimitives` and its cumulative maximum operations.
- `direction_share/primitives.py`: `DirectionSharePrimitives`.

Each family currently registers its `primitives` implementation. The factory can register additional implementations under other names.

## Decoder and output heads

- `latent_decoder.py` contains `channel_layer` and `LatentDecoder` for initial latent sampling and KL collection.
- `output_heads.py` contains `OutputHeads` for the color and x/y size projections, plus `postprocess_mask`.

## Model construction

`model/model.py` keeps the constructor readable and delegates its four weight-setup sections to `model/initialization/latent.py`, `layers.py`, `output.py`, and `symmetry.py`. The forward pass assembles the latent decoder, repeated layer stack, and output heads. `config/default.yaml` holds model dimensions, layer implementations, optimizer, loss, seed, device, and annealing hyperparameters. The `config` package loads it with OmegaConf and supports YAML files and dot-list overrides.

