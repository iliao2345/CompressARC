import numpy as np
import torch


np.random.seed(0)
torch.manual_seed(0)


NUM_DIMENSIONS = 5  # We have 5 dimensions: examples, colors, directions, x, y

class MultiTensorSystem:
    """
    A system for handling multi-dimensional configurations of 'examples',
    'colors', 'directions', and (x, y) positions. This class can generate
    and iterate through valid dimension combinations.
    """
    def __init__(self, n_examples, n_colors, n_x, n_y, task, constraint_policy='strict'):
        """
        Args:
            n_examples (int): Number of examples.
            n_colors (int): Number of colors.
            n_x (int): Size of the X dimension.
            n_y (int): Size of the Y dimension.
            task: ARC task that the multitensor system is xreated for
        """
        self.n_examples = n_examples
        self.n_colors = n_colors
        self.n_directions = 8
        self.n_x = n_x
        self.n_y = n_y
        self.task = task
        if constraint_policy not in ('strict', 'relaxed'):
            raise ValueError(
                f"Unknown multitensor constraint policy: {constraint_policy!r}"
            )
        self.constraint_policy = constraint_policy
        self.dim_lengths = [self.n_examples, self.n_colors,
                            self.n_directions, self.n_x, self.n_y]
    def dims_valid(self, dims):
        """
        Checks whether a given dimension combination is valid. The strict
        baseline uses these validity rules:
        1. If any of x/y is set (dims[3] or dims[4]), then examples (dims[0]) must also be set.
        2. Sum of dims[1:] cannot be zero (i.e., at least color, direction, or x/y must be set).
        The relaxed ablation accepts every combination except the all-zero one.
        Args:
            dims (list[int]): A list of 0/1 flags indicating which dimensions are included.
        Returns:
            bool: Whether the dimension combination is valid.
        """
        if self.constraint_policy == 'relaxed':
            # Ablation: allow every semantic dimension combination except the
            # dimensionless global channel vector.
            return sum(dims) > 0

        # Strict baseline: if x or y is set, examples must also be set.
        if (dims[3] or dims[4]) and not dims[0]:
            return False
        # At least one of [color, direction, x, y] must be set.
        if sum(dims[1:]) == 0:
            return False
        return True

    def shape(self, dims, extra_dim=None):
        """
        Creates a shape tuple for PyTorch or NumPy based on which dimensions are used.
        Args:
            dims (list[int]): A list of 0/1 flags for each dimension.
            extra_dim (int, optional): An additional dimension to be appended at the end.
        Returns:
            list[int]: The computed shape.
        """
        shape = []
        for dim_index, length in enumerate(self.dim_lengths):
            if dims[dim_index]:
                shape.append(length)
        if extra_dim is not None:
            shape.append(extra_dim)
        return shape

    def _generate_dims_combinations(self):
        """Generate all possible 5-bit dimension combinations (from 0..31)."""
        for i in range(2 ** NUM_DIMENSIONS):
            # For each of the 5 bits in i, compute dims array
            dims = [(i >> bit) & 1 for bit in range(NUM_DIMENSIONS)]
            yield dims

    def __iter__(self):
        """
        Yields valid dims.
        """
        for dims in self._generate_dims_combinations():
            if self.dims_valid(dims):
                yield dims

    def _make_multitensor(self, default, index):
        """
        Recursively creates a nested list (tree-like) of shape [2 x 2 x 2 x 2 x 2]
        (depth = NUM_DIMENSIONS) if `index < NUM_DIMENSIONS`.
        Once index == NUM_DIMENSIONS, returns `default`.
        Args:
            default (Any): The value to return at the leaf of the recursion.
            index (int): Current depth.
        Returns:
            list or default: A nested list structure or the default object if at depth.
        """
        if index == NUM_DIMENSIONS:
            return default
        return [self._make_multitensor(default, index+1) for _ in range(2)]

    def make_multitensor(self, default=None):
        """
        Create a multitensor with a default object to place at every index.
        Args:
            default (Any): The default value to place at all leaves. Default: None
        Returns:
            MultiTensor: A multitensor with the default object at every index.
        """
        return MultiTensor(self._make_multitensor(default, 0), self)


class MultiTensor:
    """
    Wrapper for a nested data structure that can be indexed by a 5-element dims array.
    """

    def __init__(self, data, multitensor_system):
        """
        Args:
            data (nested list): The nested list holding the actual data.
            multitensor_system (MultiTensorSystem): The system this MultiTensor belongs to.
        """
        self.data = data
        self.multitensor_system = multitensor_system

    def __getitem__(self, dims):
        """
        Retrieve the data at a specific 5-dimensional index.
        Args:
            dims (list[int]): 5-element array (0 or 1) indicating path in nested lists.
        Returns:
            Any: The data stored at that nested location.
        """
        d = self.data
        for dim_val in dims:
            d = d[dim_val]
        return d

    def __setitem__(self, dims, value):
        """
        Set the data at a specific 5-dimensional index.
        Args:
            dims (list[int]): 5-element array (0 or 1) indicating path in nested lists.
            value (Any): The value to store.
        """
        d = self.data
        for dim_val in dims[:-1]:
            d = d[dim_val]
        d[dims[-1]] = value


def multify(fn):
    """
    Decorator that applies a function to all valid dimension combinations
    if any arguments are MultiTensor instances.
    """

    def wrapper(*args, **kwargs):

        # Check if we should perform multi-mode or not
        multitensor_system = None
        multi_mode = False

        # Identify if any arg or kwarg is a MultiTensor
        for arg in args:
            if isinstance(arg, MultiTensor):
                multi_mode = True
                multitensor_system = arg.multitensor_system
        if not multi_mode:
            for value in kwargs.values():
                if isinstance(value, MultiTensor):
                    multi_mode = True
                    multitensor_system = value.multitensor_system
                    break

        # If none of the args/kwargs are MultiTensor, just call the function directly
        if not multi_mode:
            return fn(None, *args, **kwargs)

        # We do have MultiTensor arguments, so let's build a new MultiTensor result
        # of the same shape and fill it by iterating over valid dimension combos.
        def iterate_and_assign(multitensor_system, result_data):
            """Helper to iterate over dims and assign function outputs."""

            for dims in multitensor_system:
                # Build per-dims argument list
                new_args = []
                for arg in args:
                    if isinstance(arg, MultiTensor):
                        new_args.append(arg[dims])
                    else:
                        new_args.append(arg)
                # Build per-dims kwargs
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, MultiTensor):
                        new_kwargs[key] = value[dims]
                    else:
                        new_kwargs[key] = value
                # Call the user function on these "scalar" values
                output = fn(dims, *new_args, **new_kwargs)
                # Assign back to the result MultiTensor
                # This goes step by step into result_data
                result_data[dims] = output

        # Create an empty nested list structure
        result_data = multitensor_system.make_multitensor()
        iterate_and_assign(multitensor_system, result_data)

        # Return a MultiTensor wrapping the nested result
        return result_data

    return wrapper
