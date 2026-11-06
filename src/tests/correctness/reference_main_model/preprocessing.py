import json
import numpy as np
import torch
import multitensor_systems

np.random.seed(0)
torch.manual_seed(0)

class Task:
    """
    A class that helps deal with task-specific operations such as preprocessing,
    grid shape handling, solution processing, etc. Sets up the task-specific
    multitensor system to be used to construct the network.
    """
    def __init__(self, task_name, problem, solution, multitensor_constraints='strict'):
        self.task_name = task_name
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test
        self.unprocessed_problem = problem
        self.multitensor_constraints = multitensor_constraints

        self.shapes = self._collect_problem_shapes(problem)
        self._predict_solution_shapes()
        self._construct_multitensor_system(problem)
        self._compute_mask()
        self._create_problem_tensor(problem)

        self.solution = self._create_solution_tensor(solution) if solution else None
        if solution is None:
            self.solution_hash = None

    def _collect_problem_shapes(self, problem):
        """
        Extract input/output shapes for each example.
        """
        shapes = []
        for split_name in ['train', 'test']:
            for example in problem[split_name]:
                in_shape = list(np.array(example['input']).shape)
                out_shape = list(np.array(example['output']).shape) if 'output' in example else None
                shapes.append([in_shape, out_shape])
        return shapes

    def _predict_solution_shapes(self):
        """
        Predict output shapes when not explicitly provided.
        """
        self.in_out_same_size = all(tuple(inp) == tuple(out) for inp, out in self.shapes[:self.n_train])
        self.all_in_same_size = len({tuple(shape[0]) for shape in self.shapes}) == 1
        self.all_out_same_size = len({tuple(shape[1]) for shape in self.shapes if shape[1]}) == 1

        if self.in_out_same_size:
            for shape in self.shapes[self.n_train:]:
                shape[1] = shape[0]
        elif self.all_out_same_size:
            default_shape = self.shapes[0][1]
            for shape in self.shapes[self.n_train:]:
                shape[1] = default_shape
        else:
            max_x, max_y = self._get_max_dimensions()
            for shape in self.shapes[self.n_train:]:
                shape[1] = [max_x, max_y]

    def _get_max_dimensions(self):
        max_x, max_y = 0, 0
        for in_out_pair in self.shapes:
            for shape in in_out_pair:
                if shape:
                    max_x = max(max_x, shape[0])
                    max_y = max(max_y, shape[1])
        return max_x, max_y

    def _construct_multitensor_system(self, problem):
        """
        Build tensor system with appropriate sizes.
        """
        self.n_x = max(shape[i][0] for shape in self.shapes for i in range(2))
        self.n_y = max(shape[i][1] for shape in self.shapes for i in range(2))

        colors = {color 
                  for split in ['train', 'test']
                  for example in problem[split] 
                  for grid in [example['input'], example.get('output', [])]
                  for row in grid
                  for color in row}
        colors.add(0)  # Always include black as background

        self.colors = list(sorted(colors))
        self.n_colors = len(self.colors) - 1

        self.multitensor_system = multitensor_systems.MultiTensorSystem(
            self.n_examples,
            self.n_colors,
            self.n_x,
            self.n_y,
            self,
            constraint_policy=self.multitensor_constraints,
        )

    def _create_problem_tensor(self, problem):
        """
        Convert input/output grids to tensors.
        """
        self.problem = np.zeros((self.n_examples, self.n_colors + 1, self.n_x, self.n_y, 2))
        
        for subsplit, n_examples in [('train', self.n_train), ('test', self.n_test)]:
            for example_num, example in enumerate(problem[subsplit]):
                new_example_num = example_num if subsplit == 'train' else self.n_train + example_num

                for mode in ('input', 'output'):
                    if subsplit == 'test' and mode == 'output':
                        continue

                    grid = self._create_grid_tensor(
                        example.get(mode, np.zeros(self.shapes[new_example_num][1]))
                    )
                    mode_num = 0 if mode == 'input' else 1
                    self.problem[new_example_num, :, :grid.shape[1], :grid.shape[2], mode_num] = grid

        self.problem = torch.from_numpy(np.argmax(self.problem, axis=1)).to(torch.get_default_device())

    def _create_grid_tensor(self, grid):
        return np.array([
            [[1 if self.colors.index(color) == ref_color else 0
              for color in row]
             for row in grid]
            for ref_color in range(self.n_colors + 1)
        ])

    def _create_solution_tensor(self, solution):
        """
        Convert solution grids to tensors for crossentropy evaluation.
        """
        solution_tensor = np.zeros((self.n_test, self.n_colors + 1, self.n_x, self.n_y))
        solution_tuple = ()

        for example_num, grid in enumerate(solution):
            solution_tuple += (tuple(map(tuple, grid)),)
            grid_tensor = self._create_grid_tensor(grid)
            # unfortunately sometimes the solution tensor will be bigger than (n_x, n_y), and in these cases
            # we'll never get the solution.
            min_x, min_y = min(grid_tensor.shape[1], self.n_x), min(grid_tensor.shape[2], self.n_y)
            solution_tensor[example_num, :, :min_x, :min_y] = grid_tensor[:, :min_x, :min_y]

        self.solution_hash = hash(solution_tuple)
        return torch.from_numpy(np.argmax(solution_tensor, axis=1)).to(torch.get_default_device())

    def _compute_mask(self):
        """
        Compute masks for activations and cross-entropies.
        """
        self.masks = np.zeros((self.n_examples, self.n_x, self.n_y, 2))

        for example_num, (in_shape, out_shape) in enumerate(self.shapes):
            for mode_num, shape in enumerate([in_shape, out_shape]):
                if shape:
                    x_mask = np.arange(self.n_x) < shape[0]
                    y_mask = np.arange(self.n_y) < shape[1]
                    self.masks[example_num, :, :, mode_num] = np.outer(x_mask, y_mask)

        self.masks = torch.from_numpy(self.masks).to(torch.get_default_dtype()).to(torch.get_default_device())


def preprocess_tasks(split, task_nums_or_task_names, multitensor_constraints='strict'):
    """
    Preprocess tasks by loading problems and solutions.
    """
    with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)

    solutions = None if split == "test" else json.load(open(f'dataset/arc-agi_{split}_solutions.json'))
    
    task_names = list(problems.keys())
    
    return [Task(task_name,
                 problems[task_name],
                 solutions.get(task_name) if solutions else None,
                 multitensor_constraints=multitensor_constraints)
            for task_name in task_names
            if task_name in task_nums_or_task_names or task_names.index(task_name) in task_nums_or_task_names]
