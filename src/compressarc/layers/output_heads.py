"""Final color and output-size heads for the ARC compressor."""

from .helper import affine


def postprocess_mask(task, x_mask, y_mask):
    """Force mask logits to exclude positions beyond each task's known bounds."""
    x_modifier = x_mask.new_zeros((task.n_examples, task.n_x, 2))
    y_modifier = y_mask.new_zeros((task.n_examples, task.n_y, 2))

    for example_num in range(task.n_examples):
        max_x = max(task.shapes[example_num][0][0], task.shapes[example_num][1][0])
        max_y = max(task.shapes[example_num][0][1], task.shapes[example_num][1][1])
        x_modifier[example_num, max_x:, :] = -1000
        y_modifier[example_num, max_y:, :] = -1000

    return x_mask + x_modifier, y_mask + y_modifier


class OutputHeads:
    """Apply the color and x/y size projections and mask postprocessing."""

    def forward(self, x, head_weights, mask_weights, task):
        """Return color logits and postprocessed x- and y-size logits."""
        output = (
            affine(x[[1, 1, 0, 1, 1]], head_weights, use_bias=False)
            + 100 * head_weights[1]
        )
        x_mask = affine(x[[1, 0, 0, 1, 0]], mask_weights, use_bias=True)
        y_mask = affine(x[[1, 0, 0, 0, 1]], mask_weights, use_bias=True)
        x_mask, y_mask = postprocess_mask(task, x_mask, y_mask)
        return output, x_mask, y_mask

    __call__ = forward