import numpy as np
import torch
import multitensor_systems


np.random.seed(0)
torch.manual_seed(0)


class Initializer:
    def __init__(self, multitensor_system, channel_dim_fn):
        """
        Initializes weight tensors for a multitensor system.
        Args:
            multitensor_system (MultiTensorSystem): The multitensor system that we want to use
                    for initializing weights.
            channel_dim_fn (function): A function that takes in a dims list of type list[int], and
                    returns an int representing the channel dimension size.
        """
        self.multitensor_system = multitensor_system
        self.channel_dim_fn = channel_dim_fn
        self.weights_list = []

    def initialize_zeros(self, dims, shape):
        """Initializes a weight tensor with zeros."""
        if callable(shape):
            shape = shape(dims)
        zeros = torch.zeros(shape, requires_grad=True)
        self.weights_list.append(zeros)
        return zeros

    def initialize_linear(self, dims, shape):
        """Initializes a linear transformation."""
        if callable(shape):
            shape = shape(dims)
        n_in, n_out = shape

        if callable(n_in):
            n_in = n_in(dims)
        if callable(n_out):
            n_out = n_out(dims)

        # A relaxed multitensor system contains an example-only tensor. Its
        # softmax branch intentionally has zero output channels, which creates
        # a following 0 -> residual-width projection. Keep that no-op finite.
        scale = 0 if n_in == 0 else 1 / np.sqrt(n_in)
        weight = scale * torch.randn(n_in, n_out)
        bias = scale * torch.randn(n_out)
        weight.requires_grad = True
        bias.requires_grad = True

        self.weights_list.extend([weight, bias])
        return [weight, bias]

    def initialize_residual(self, dims, n_in, n_out):
        """Initializes two linear layers that map to and from the residual stream."""
        linear_1 = self.initialize_linear(dims, [self.channel_dim_fn, n_in])
        linear_2 = self.initialize_linear(dims, [n_out, self.channel_dim_fn])
        return [linear_1, linear_2]

    def initialize_posterior(self, dims, channel_dim):
        """Initializes a posterior z distribution for the decoding layer."""
        if callable(channel_dim):
            channel_dim = channel_dim(dims)

        shape = self.multitensor_system.shape(dims, channel_dim)
        mean = 0.01 * torch.randn(shape)
        mean.requires_grad=True
        local_capacity_adjustment = self.initialize_zeros(dims, shape)

        self.weights_list.append(mean)
        return [mean, local_capacity_adjustment]

    def initialize_direction_share(self, dims, _):
        """
        Initializes linear maps for the directional communication layer. Symmetrization
        is to be performed later by symmetrize_direction_sharing().
        """
        channel_dim_fn = self.channel_dim_fn
        return [[self.initialize_linear(dims, [channel_dim_fn, channel_dim_fn]) for _ in range(8)] for _ in range(8)]

    def initialize_head(self):
        """Initializes the linear head while ensuring symmetry wrt swapping x and y."""
        dims = [1, 1, 0, 1, 1]
        head_weights = self.initialize_linear(dims, [self.channel_dim_fn(dims), 2])

        # Ensure symmetry
        head_weights[0].requires_grad = False
        head_weights[0] = torch.stack([head_weights[0][..., 0]] * 2, dim=-1)
        head_weights[0].requires_grad = True

        # Maintain correct weight list order
        self.weights_list[-2] = head_weights[0]
        return head_weights

    # The functions below serve to perform the initializations once per tensor
    # in the multitensor. Functions can also be fed in as arguments instead,
    # and they will be run with dims as an argument, to produce a different
    # argument for every tensor in the multitensor.
    def initialize_multizeros(self, shape):
        return multitensor_systems.multify(self.initialize_zeros)(
            self.multitensor_system.make_multitensor(default=shape)
        )

    def initialize_multilinear(self, shape):
        return multitensor_systems.multify(self.initialize_linear)(
            self.multitensor_system.make_multitensor(default=shape)
        )

    def initialize_multiresidual(self, n_in, n_out):
        return multitensor_systems.multify(self.initialize_residual)(
            n_in, self.multitensor_system.make_multitensor(default=n_out)
        )

    def initialize_multiposterior(self, decoding_dim):
        return multitensor_systems.multify(self.initialize_posterior)(
            self.multitensor_system.make_multitensor(default=decoding_dim)
        )

    def initialize_multidirection_share(self):
        return multitensor_systems.multify(self.initialize_direction_share)(
            self.multitensor_system.make_multitensor()
        )

    def symmetrize_xy(self, multiweights):
        """Ensures xy swap symmetry for weights by enforcing shared values."""
        for dims in self.multitensor_system:
            if dims[3] == 0 and dims[4] == 1:
                multiweights[dims] = multiweights[dims[:3] + [1, 0]]

    def symmetrize_direction_sharing(self, multiweights):
        """
        Ensures xy swap symmetry for weights by enforcing shared values.
        Enforcement of shared values is more complicated since the direction axis
        is involved, which has individual indices assigned to individual directions.
        """

        # For every directional communication linear map, identify one linear map
        # that will serve as the representative map for all reachable maps under
        # the equivariance transformation. Always use that representative map.
        for dims in self.multitensor_system:
            for dir1 in range(8):
                for dir2 in range(8):
                    from_dims = dims
                    from_dir1, from_dir2 = dir1, dir2

                    # Apply the transformations under certain conditions to reduce a map
                    # to the representative map.
                    if dims[3] + dims[4] == 1:
                        from_dims = dims[:3] + [1, 0]
                        if dims[4] == 1:
                            from_dir1 = (2 + from_dir1) % 8
                            from_dir2 = (2 + from_dir2) % 8

                        if from_dir1 > 4 or (from_dir1 in {0, 4} and from_dir2 > 4):
                            from_dir1 = (8 - from_dir1) % 8
                            from_dir2 = (8 - from_dir2) % 8

                        if 2 < from_dir1 < 6 or (from_dir1 in {2, 6} and 2 < from_dir2 < 6):
                            from_dir1 = (4 - from_dir1) % 8
                            from_dir2 = (4 - from_dir2) % 8
                    else:
                        rotation = (from_dir1 // 2) * 2
                        from_dir1 = (from_dir1 - rotation) % 8
                        from_dir2 = (from_dir2 - rotation) % 8

                        if (from_dir2 - from_dir1) % 8 > 4:
                            from_dir2 = (8 + 2 * from_dir1 - from_dir2) % 8

                    # Copy down the representative map for later use.
                    multiweights[dims][dir1][dir2] = multiweights[from_dims][from_dir1][from_dir2]

