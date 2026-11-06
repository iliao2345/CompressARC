import os
import sys
import time
import json
import importlib
import gc
import multiprocessing
import traceback
from pathlib import Path

# The analysis package now lives under src/, while this worker still uses the
# legacy flat modules retained in "compress stuff".
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPOSITORY_ROOT / "compress stuff"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(LEGACY_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch

import preprocessing
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
"""
A worker function that solves one puzzle and can be called by a custom multiprocessing runner. It is not a command-line entry point.
"""

def solve_task(task_name, split, time_limit, n_train_iterations, gpu_id, memory_dict, solutions_dict, error_queue):
    """
    Solves a puzzle.
    Args:
        task_name (str): The name of the puzzle to solve.
        split (str): 'training', 'evaluation', or 'test'
        time_limit (float): An end time that will cause training to exit early if reached.
        n_train_iterations (int): The number of iterations to train for.
        gpu_id (int): The GPU number to run the solver on.
        memory_dict (multiprocessing.Dict[str, int]): An inter-process shared dict that we
            can store the amount of memory taken by this job in.
        solutions_dict (multiprocessing.Dict[str, list[Dict[str, list[list[int]]]]]): An
            inter-process shared dict that we can store the solution in.
        error_queue (multiprocessing.Queue[Exception]): An inter-process shared queue to
            put errors in when an exception occurs.
    """

    try:  # Error catching block that puts errors on the error_queue

        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()  # Measure the memory used.

        # Get the task
        with open(LEGACY_ROOT / 'dataset' / f'arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        task = preprocessing.Task(task_name, problems[task_name], None)
        del problems

        # Set up the training
        model = arc_compressor.ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        train_history_logger = solution_selection.Logger(task)
        train_history_logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        train_history_logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))

        # Training loop
        for train_step in range(n_train_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger)
            if time.time() > time_limit:
                break

        # Get the solution
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        del task
        del model
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        # Store the result
        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list

    except Exception as e:  # If error, write to the error queue
        error_queue.put(traceback.format_exc())
