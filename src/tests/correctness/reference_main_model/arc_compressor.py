import numpy as np
import torch

import initializers
import layers


np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class ARCCompressor:
    """
    The main model class for the VAE Decoder in our solution to ARC.
    """

    # Define the channel dimensions that all the layers use
    n_layers = 4
    share_up_dim = 16
    share_down_dim = 8
    decoding_dim = 4
    softmax_dim = 2
    cummax_dim = 4
    shift_dim = 4
    nonlinear_dim = 16

    # This function gives the channel dimension of the residual stream depending on
    # which dimensions are present, for every tensor in the multitensor.
    def channel_dim_fn(self, dims):
        return 16 if dims[2] == 0 else 8

    def __init__(self, task, shift_variant='original', direction_share_variant='original',
                 cummax_variant='original'):
        """
        Create a model that is tailored to the given task, and initialize all the weights.
        The weights are symmetrized such that swapping the x and y dimension ordering should
        make the output's dimension ordering also swapped, for the same weights. This may not
        be exactly correct since symmetrizing all operations is difficult.
        Args:
            task (preprocessing.Task): The task which the model is to be made for solving.
        """
        if shift_variant not in ('original', 'projected_conv', 'pure_conv'):
            raise ValueError(f'Unknown shift variant: {shift_variant}')
        self.shift_variant = shift_variant
        if direction_share_variant not in ('original', 'projected_d4', 'pure_d4'):
            raise ValueError(f'Unknown direction-share variant: {direction_share_variant}')
        self.direction_share_variant = direction_share_variant
        if cummax_variant not in ('original', 'projected_lse', 'pure_lse'):
            raise ValueError(f'Unknown cummax variant: {cummax_variant}')
        self.cummax_variant = cummax_variant
        self.multitensor_system = task.multitensor_system

        # Initialize weights
        initializer = initializers.Initializer(self.multitensor_system, self.channel_dim_fn)

        self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
        self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
        initializer.symmetrize_xy(self.decode_weights)
        self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])

        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.direction_share_weights = []
        self.nonlinear_weights = []

        for layer_num in range(self.n_layers):
            self.share_up_weights.append(initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim))
            self.share_down_weights.append(initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim))
            output_scaling_fn = lambda dims: self.softmax_dim * (2 ** (dims[1] + dims[2] + dims[3] + dims[4]) - 1)
            self.softmax_weights.append(initializer.initialize_multiresidual(self.softmax_dim, output_scaling_fn))
            self.cummax_weights.append(initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim))
            self.shift_weights.append(initializer.initialize_multiresidual(self.shift_dim, self.shift_dim))
            self.direction_share_weights.append(initializer.initialize_multidirection_share())
            self.nonlinear_weights.append(initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim))

        self.head_weights = initializer.initialize_head()
        self.mask_weights = initializer.initialize_linear(
            [1, 0, 0, 1, 0], [self.channel_dim_fn([1, 0, 0, 1, 0]), 2]
        )

        # Symmetrize weights so that their behavior is equivariant to swapping x and y dimension ordering
        for weight_list in [
            self.share_up_weights,
            self.share_down_weights,
            self.softmax_weights,
            self.cummax_weights,
            self.shift_weights,
            self.nonlinear_weights,
        ]:
            for layer_num in range(self.n_layers):
                initializer.symmetrize_xy(weight_list[layer_num])

        for layer_num in range(self.n_layers):
            initializer.symmetrize_direction_sharing(self.direction_share_weights[layer_num])

        self.direction_projection_weights = None
        if direction_share_variant == 'projected_d4':
            self.direction_projection_weights = [
                initializer.initialize_multiresidual(8, 8)
                for _ in range(self.n_layers)
            ]
            for weights in self.direction_projection_weights:
                initializer.symmetrize_xy(weights)

        self.weights_list = initializer.weights_list
        if shift_variant != 'original':
            from tied_convolution import TiedDirectionalConv
            # One trainable pair of canonical kernels per depth, shared across
            # colors/features and the two active multitensor components.
            self.tied_convs = [TiedDirectionalConv() for _ in range(self.n_layers)]
            if shift_variant == 'pure_conv':
                # Preserve baseline RNG consumption during initialization, but
                # do not optimize projections absent from this architecture.
                unused = {id(w) for weights in self.shift_weights
                          for dims in self.multitensor_system
                          for pair in weights[dims] for w in pair}
                self.weights_list = [w for w in self.weights_list if id(w) not in unused]
                self.shift_weights = None
            self.weights_list.extend(p for conv in self.tied_convs for p in conv.parameters())

        if direction_share_variant != 'original':
            from direction_share import D4DirectionShare
            unused = {id(w) for weights in self.direction_share_weights
                      for dims in self.multitensor_system if dims[2]
                      for row in weights[dims] for pair in row for w in pair}
            self.weights_list = [w for w in self.weights_list if id(w) not in unused]
            self.direction_share_weights = None
            self.d4_direction_shares = [D4DirectionShare() for _ in range(self.n_layers)]
            self.weights_list.extend(
                parameter for module in self.d4_direction_shares
                for parameter in module.parameters()
            )

        if cummax_variant != 'original':
            from lse import MorphologicalMax
            self.lse_modules = [
                MorphologicalMax(task.n_x, task.n_y) for _ in range(self.n_layers)
            ]
            if cummax_variant == 'pure_lse':
                unused = {id(w) for weights in self.cummax_weights
                          for dims in self.multitensor_system
                          for pair in weights[dims] for w in pair}
                self.weights_list = [w for w in self.weights_list if id(w) not in unused]
                self.cummax_weights = None
            self.weights_list.extend(
                parameter for module in self.lse_modules
                for parameter in module.parameters()
            )


    def forward(self):
        """
        Compute the forward pass of the VAE decoder. Start by using internally stored latents,
        and process from there. Output an [example, color, x, y, channel] tensor for the colors,
        and an [example, x, channel] and [example, y, channel] tensor for the masks.
        Returns:
            Tensor: An [example, color, x, y, channel] tensor, where for every example,
                    input/output (picked by channel dimension), and every pixel (picked
                    by x and y dimensions), we have a vector full of logits for that
                    pixel being each possible color.
            Tensor: An [example, x, channel] tensor, where for every example, input/output
                    (picked by channel dimension), and every x, we assign a score that
                    contributes to the likelihood that that index of the x dimension is not
                    masked out in the prediction.
            Tensor: An [example, y, channel] tensor, used in the same way as above.
            list[Tensor]: A list of tensors indicating the amount of KL contributed by each component
                    tensor in the layers.decode_latents() step.
            list[str]: A list of tensor names that correspond to each tensor in the aforementioned output.
        """
        # Decoding layer
        x, KL_amounts, KL_names = layers.decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )

        for layer_num in range(self.n_layers):
            # Multitensor communication layer
            x = layers.share_up(x, self.share_up_weights[layer_num])

            # Softmax layer
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Directional layers
            if self.cummax_variant == 'original':
                x = layers.cummax(
                    x, self.cummax_weights[layer_num], self.multitensor_system.task.masks,
                    pre_norm=False, post_norm=True, use_bias=False
                )
            else:
                x = layers.lse_cummax(
                    x, self.multitensor_system.task.masks, self.lse_modules[layer_num],
                    None if self.cummax_weights is None else self.cummax_weights[layer_num],
                    projected=self.cummax_variant == 'projected_lse',
                )
            if self.shift_variant == 'original':
                x = layers.shift(
                    x, self.shift_weights[layer_num], self.multitensor_system.task.masks,
                    pre_norm=False, post_norm=True, use_bias=False
                )
            else:
                x = layers.tied_conv_shift(
                    x, self.multitensor_system.task.masks, self.tied_convs[layer_num],
                    None if self.shift_weights is None else self.shift_weights[layer_num],
                    projected=self.shift_variant == 'projected_conv',
                )

            # Directional communication layer
            if self.direction_share_variant == 'original':
                x = layers.direction_share(
                    x, self.direction_share_weights[layer_num], pre_norm=True, use_bias=False
                )
            else:
                x = layers.d4_direction_share(
                    x,
                    self.d4_direction_shares[layer_num],
                    None if self.direction_projection_weights is None
                    else self.direction_projection_weights[layer_num],
                    projected=self.direction_share_variant == 'projected_d4',
                )

            # Nonlinear layer
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Multitensor communication layer
            x = layers.share_down(x, self.share_down_weights[layer_num])

            # Normalization layer
            x = layers.normalize(x)

        # Linear Heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], self.head_weights, use_bias=False)
            + 100 * self.head_weights[1]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], self.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], self.mask_weights, use_bias=True)

        # Postprocessing
        x_mask, y_mask = layers.postprocess_mask(self.multitensor_system.task, x_mask, y_mask)

        return output, x_mask, y_mask, KL_amounts, KL_names

