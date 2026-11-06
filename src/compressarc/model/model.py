"""ARC compressor model configuration, parameter construction, and forward pass."""

from config import load_config
from ..train.layer_profiling import profile_layer
from .initializers import Initializer
from .initialization.latent import initialize_latent_weights
from .initialization.layers import initialize_layer_weights
from .initialization.output import initialize_output_weights
from .initialization.symmetry import apply_weight_symmetry
from .layer_factory import LayerFactory
from .multitensor_systems import multify
from ..layers.helper import nonlinear, normalize, softmax
from ..layers.softmax_fused import softmax_fused
from ..layers.latent_decoder import LatentDecoder
from ..layers.output_heads import OutputHeads
from ..layers.share.primitives import MultitensorShare
from ..layers.share.reuse import ReusedMultitensorShare


class ARCCompressor:
    """The ARC compressor model."""

    def __init__(self, task, config):
        self.multitensor_system = task.multitensor_system
        self.config = config if config is not None else load_config().model
        self.n_layers = self.config.n_layers

        self.initializer = Initializer(
            self.multitensor_system,
            self.channel_dim_fn,
        )

        self._init_latent_weights()
        self._init_layer_weights()
        self._init_output_weights()
        self._apply_symmetry()

        self.weights_list = self.initializer.weights_list
        self._init_layers()

    def channel_dim_fn(self, dims):
        """Return the residual channel width for a multitensor component."""
        return self.config.channel_dim_shared if dims[2] == 0 else self.config.channel_dim_grid

    def _init_latent_weights(self):
        (
            self.multiposteriors,
            self.decode_weights,
            self.target_capacities,
        ) = initialize_latent_weights(self.initializer, self.config)

    def _init_layer_weights(self):
        layer_weights = initialize_layer_weights(self.initializer, self.config)
        for name, weights in layer_weights.items():
            setattr(self, name, weights)

        self.layer_weights = [
            {
                "share_up": self.share_up_weights[index],
                "share_down": self.share_down_weights[index],
                "softmax": self.softmax_weights[index],
                "cummax": self.cummax_weights[index],
                "shift": self.shift_weights[index],
                "direction_share": self.direction_share_weights[index],
                "nonlinear": self.nonlinear_weights[index],
            }
            for index in range(self.n_layers)
        ]

    def _init_output_weights(self):
        self.head_weights, self.mask_weights = initialize_output_weights(
            self.initializer,
            self.channel_dim_fn,
        )

    def _apply_symmetry(self):
        layer_weights = {
            name: getattr(self, name)
            for name in (
                "share_up_weights",
                "share_down_weights",
                "softmax_weights",
                "cummax_weights",
                "shift_weights",
                "direction_share_weights",
                "nonlinear_weights",
            )
        }
        apply_weight_symmetry(
            self.initializer,
            layer_weights,
            self.decode_weights,
        )

    def _init_projected_direction_share(self):
        # The pilot consumed baseline pair-weight initialization, then created
        # separate 8-channel projections after heads and symmetry constraints.
        width = self.config.get("direction_share_dim", 8)
        self.direction_projection_weights = [
            self.initializer.initialize_multiresidual(width, width)
            for _ in range(self.n_layers)
        ]
        for weights in self.direction_projection_weights:
            self.initializer.symmetrize_xy(weights)
        unused = {
            id(tensor) for weights in self.direction_share_weights
            for dims in self.multitensor_system if dims[2]
            for row in weights[dims] for pair in row for tensor in pair
        }
        self.weights_list[:] = [weight for weight in self.weights_list if id(weight) not in unused]
        self.direction_share_weights = self.direction_projection_weights
        for index, weights in enumerate(self.direction_projection_weights):
            self.layer_weights[index]["direction_share"] = weights

    def _init_layers(self):
        if self.config.direction_share_implementation == "d4":
            self._init_projected_direction_share()
        softmax_backend = self.config.get("softmax_implementation", "primitives")
        if softmax_backend not in ("primitives", "fused"):
            raise ValueError(f"Unknown softmax implementation: {softmax_backend!r}")
        self.softmax = softmax_fused if softmax_backend == "fused" else softmax
        factory = LayerFactory()
        self.latent_decoder = LatentDecoder(multify)
        self.output_heads = OutputHeads()
        sharing_backend = self.config.get("share_implementation", "primitives")
        if sharing_backend not in ("primitives", "reuse", "batched", "reuse_fused_norm"):
            raise ValueError("share_implementation must be primitives, reuse, batched or reuse_fused_norm")
        self.share = (MultitensorShare() if sharing_backend == "primitives" else
                      ReusedMultitensorShare(batch=sharing_backend == "batched",
                          max_batch_elements=self.config.get("share_batch_max_elements", 262144),
                          fused_normalization=sharing_backend == "reuse_fused_norm"))
        self._layer_call_options = {}
        for name in ("cummax", "shift", "direction_share"):
            implementation = getattr(self.config, f"{name}_implementation")
            options = {}
            projected = implementation != "primitives" and (name, implementation) not in {
                ("direction_share", "vectorized"), ("direction_share", "vectorized_packed"), ("cummax", "triton_hard"),
                ("cummax", "triton_hard_grouped")}
            if projected:
                options["n_layers"] = self.n_layers
                if name == "cummax":
                    options.update(
                        height=self.multitensor_system.n_x,
                        width=self.multitensor_system.n_y,
                        tau=self.config.get("lse_tau", 0.1),
                        timing=self.config.get("layer_timing", False),
                    )
            if name == "direction_share" and implementation == "d4":
                options["packed"] = self.config.get("direction_share_packed", False)
            layer = factory.create(name, implementation, multify=multify, **options)
            setattr(self, name, layer)
            self._layer_call_options[name] = projected
            if projected:
                anchor = self.weights_list[0]
                layer.to(device=anchor.device, dtype=anchor.dtype)
                self.weights_list.extend(layer.parameters())

    def forward(self):
        with profile_layer("latent_decoder"):
            x, kl_amounts, kl_names = self.latent_decoder(
                self.target_capacities,
                self.decode_weights,
                self.multiposteriors,
            )

        masks = self.multitensor_system.task.masks
        for layer_num in range(self.n_layers):
            weights = self.layer_weights[layer_num]

            with profile_layer(f"depth_{layer_num}/share_up"):
                x = self.share.share_up(x, weights["share_up"])
            with profile_layer(f"depth_{layer_num}/softmax"):
                x = self.softmax(
                    x,
                    weights["softmax"],
                    pre_norm=False,
                    post_norm=False,
                    use_bias=False,
                )
            with profile_layer(f"depth_{layer_num}/cummax"):
                x = self.cummax(
                    x,
                    weights["cummax"],
                    masks,
                    pre_norm=False,
                    post_norm=True,
                    use_bias=False,
                    **({"layer_index": layer_num} if self._layer_call_options["cummax"] else {}),
                )
            with profile_layer(f"depth_{layer_num}/shift"):
                x = self.shift(
                    x,
                    weights["shift"],
                    masks,
                    pre_norm=False,
                    post_norm=True,
                    use_bias=False,
                    **({"layer_index": layer_num} if self._layer_call_options["shift"] else {}),
                )
            with profile_layer(f"depth_{layer_num}/direction_share"):
                x = self.direction_share(
                    x,
                    weights["direction_share"],
                    pre_norm=True,
                    use_bias=False,
                    **({"layer_index": layer_num} if self._layer_call_options["direction_share"] else {}),
                )
            with profile_layer(f"depth_{layer_num}/nonlinear"):
                x = nonlinear(
                    x,
                    weights["nonlinear"],
                    pre_norm=False,
                    post_norm=False,
                    use_bias=False,
                )
            with profile_layer(f"depth_{layer_num}/share_down"):
                x = self.share.share_down(x, weights["share_down"])
            with profile_layer(f"depth_{layer_num}/normalize"):
                x = normalize(x)

        with profile_layer("output_heads"):
            output, x_mask, y_mask = self.output_heads(
                x,
                self.head_weights,
                self.mask_weights,
                self.multitensor_system.task,
            )
        return output, x_mask, y_mask, kl_amounts, kl_names



