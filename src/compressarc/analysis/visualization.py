import os
import matplotlib.pyplot as plt
import numpy as np
import torch


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


color_list = np.array([
    [0, 0, 0],  # black
    [30, 147, 255],  # blue
    [249, 60, 49],  # red
    [79, 204, 48],  # green
    [255, 220, 0],  # yellow
    [153, 153, 153],  # gray
    [229, 58, 163],  # magenta
    [255, 133, 27],  # orange
    [135, 216, 241],  # light blue
    [146, 18, 49],  # brown
])

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

def plot_problem(logger, fname=None):
    """
    Draw a plot of an ARC-AGI problem.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
        fname (str | None | False): Output path, the default plots/ path, or False to
                return the unsaved Matplotlib figure.
    """

    # Put all the grids beside one another on one grid
    task = logger.task if hasattr(logger, "task") else logger
    n_train = task.n_train
    n_test = task.n_test
    n_examples = task.n_examples
    n_x = task.n_x
    n_y = task.n_y
    pixels = 255+np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],mode_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid
    pixels = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, arrows, and a question mark for unsolved examples.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for example_num in range(n_examples):
        for mode_num, mode in enumerate(('input', 'output')):
            if example_num < n_train:
                subsplit = 'train'
                subsplit_example_num = example_num
            else:
                subsplit = 'test'
                subsplit_example_num = example_num - n_train
            ax.arrow((2*n_y+8)-3-0.5, (2*n_x+2)*example_num+1+n_x-0.5, 6, 0, width=0.5, fc='k', ec='k', length_includes_head=True)
            if subsplit == 'test' and mode == 'output':
                ax.text((2*n_y+8)+4+n_y-0.5, (2*n_x+2)*example_num+1+n_x-0.5, '?', size='xx-large', ha='center', va='center')
                continue
            grid = np.array(task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            for xline in range(grid.shape[0]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]-0.5, (2*n_y+8)*mode_num+4+n_y+grid.shape[1]-0.5),
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(grid.shape[1]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]-0.5, (2*n_x+2)*example_num+1+n_x+grid.shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    plt.axis('off')
    if fname is False:
        return fig
    if fname is None:
        os.makedirs("plots/", exist_ok=True)
        fname = 'plots/' + task.task_name + '_problem.png'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_solution(logger, fname=None):
    """
    Draw a plot of a model's solution to an ARC-AGI problem, and save it in plots/
    Draws four plots: A model output sample, the mean of samples, and the top two most common samples.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
        fname (str | None | False): Output path, the default plots/ path, or False to
                return the unsaved Matplotlib figure.
    """
    task = logger.task if hasattr(logger, "task") else logger
    n_train = task.n_train
    n_test = task.n_test
    n_examples = task.n_examples
    n_x = task.n_x
    n_y = task.n_y

    # Four plotted solutions
    solutions_list = [
            torch.softmax(logger.current_logits, dim=1).cpu().numpy(),
            torch.softmax(logger.ema_logits, dim=1).cpu().numpy(),
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
            ]
    masks_list = [
            (logger.current_x_mask, logger.current_y_mask),
            (logger.ema_x_mask, logger.ema_y_mask),
            None,
            None,
            ]
    solutions_labels = [
            'sample',
            'sample average',
            'guess 1',
            'guess 2',
            ]
    n_plotted_solutions = len(solutions_list)

    # Put all the grids beside one another on one grid
    pixels = 255+np.zeros([n_test, 2*n_x+2, n_plotted_solutions, 2*n_y+8, 3], dtype=np.uint8)
    shapes = []
    for subsplit_example_num in range(n_test):
        subsplit = 'test'
        example_num = subsplit_example_num + n_train
        shapes.append([])

        for solution_num, (solution, masks, label) in enumerate(zip(solutions_list, masks_list, solutions_labels)):
            grid = np.array(solution[subsplit_example_num])  # c, x, y if 'sample' in label else x, y, c
            if 'sample' in label:
                grid = np.einsum('dxy,dc->xyc', grid, color_list[logger.task.colors])  # x, y, c
                if logger.task.in_out_same_size or logger.task.all_out_same_size:
                    x_length = logger.task.shapes[example_num][1][0]
                    y_length = logger.task.shapes[example_num][1][1]
                else:
                    x_length = None
                    y_length = None
                x_start, x_end = logger._best_slice_point(masks[0][subsplit_example_num,:], x_length)
                y_start, y_end = logger._best_slice_point(masks[1][subsplit_example_num,:], y_length)
                grid = grid[x_start:x_end,y_start:y_end,:]  # x, y, c
                grid = np.clip(grid, 0, 255).astype(np.uint8)
            else:
                grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
                grid = convert_color(grid)  # x, y, c

            shapes[subsplit_example_num].append((grid.shape[0], grid.shape[1]))
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[subsplit_example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],solution_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid

    pixels = pixels.reshape([n_test*(2*n_x+2), n_plotted_solutions*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, and labels.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for subsplit_example_num in range(n_test):
        for solution_num in range(n_plotted_solutions):
            subsplit = 'test'
            grid = np.array(solutions_list[solution_num][subsplit_example_num])  # x, y
            shape = shapes[subsplit_example_num][solution_num]
            for xline in range(shape[0]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]-0.5, (2*n_y+8)*solution_num+4+n_y+shape[1]-0.5),
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(shape[1]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]-0.5, (2*n_x+2)*subsplit_example_num+1+n_x+shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    for solution_num, solution_label in enumerate(solutions_labels):
        ax.text((2*n_y+8)*solution_num+4+n_y-0.5, -3, solution_label, size='xx-small', ha='center', va='center')
    plt.axis('off')
    if fname is False:
        return fig
    if fname is None:
        os.makedirs("plots/", exist_ok=True)
        fname = 'plots/' + task.task_name + '_solutions.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_grid(axis, grid, title):
    from matplotlib.colors import ListedColormap

    palette = color_list / 255.0
    axis.set_title(title)
    axis.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    axis.grid(which="minor", color="#555555", linewidth=0.5)
    axis.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    axis.imshow(grid, cmap=ListedColormap(palette), vmin=0, vmax=9, interpolation="nearest")


def plot_predictions(task, first, second):
    """Return a figure showing both decoded predictions for every test input."""
    figure, axes = plt.subplots(
        task.n_test,
        2,
        figsize=(6, max(2.5 * task.n_test, 3)),
        squeeze=False,
    )
    for index in range(task.n_test):
        for column, (solution, title) in enumerate(((first, "Guess 1"), (second, "Guess 2"))):
            grid = np.asarray(solution[index], dtype=np.int64)
            draw_grid(axes[index, column], grid, f"Test {index + 1} {title}")
    figure.tight_layout()
    return figure


def plot_pca_component(component, axis_names, component_number, strength, max_panels):
    """Return a figure showing one latent PCA component."""
    if component.ndim == 1:
        figure, axis = plt.subplots(figsize=(max(4, component.shape[0] * 0.35), 2.5))
        axis.imshow(component[None, :], cmap="gray", vmin=-1, vmax=1, aspect="auto")
        axis.set_yticks([])
        axis.set_xlabel(axis_names[0])
    elif component.ndim == 2:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(component, cmap="gray", vmin=-1, vmax=1, aspect="auto")
        axis.set_ylabel(axis_names[0])
        axis.set_xlabel(axis_names[1])
    else:
        leading_shape = component.shape[:-2]
        panel_count = min(int(np.prod(leading_shape)), max_panels)
        column_count = min(4, panel_count)
        row_count = int(np.ceil(panel_count / column_count))
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(4 * column_count, 3.5 * row_count),
            squeeze=False,
        )
        panels = component.reshape((-1,) + component.shape[-2:])
        for panel_number, axis in enumerate(axes.flat):
            if panel_number >= panel_count:
                axis.axis("off")
                continue
            axis.imshow(panels[panel_number], cmap="gray", vmin=-1, vmax=1, aspect="auto")
            leading_index = np.unravel_index(panel_number, leading_shape)
            axis.set_title(
                ", ".join(
                    f"{name}={index}"
                    for name, index in zip(axis_names[:-2], leading_index)
                )
            )
            axis.set_ylabel(axis_names[-2])
            axis.set_xlabel(axis_names[-1])
    figure.suptitle(f"Component {component_number}; strength={strength:.5g}")
    figure.tight_layout()
    return figure


__all__ = [
    "draw_grid",
    "plot_problem",
    "plot_solution",
    "plot_predictions",
    "plot_pca_component",
]