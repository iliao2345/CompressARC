"""Training metric naming and prediction tracking."""

import numpy as np
import torch


class SolutionTracker:
    """Maintain frequent test predictions and an EMA candidate, as in the old logger."""

    ema_decay = 0.97
    uncertainty_weight = 10
    early_step_penalty = 10
    early_step_cutoff = 150
    ema_penalty = 4

    def __init__(self, task):
        self.task = task
        shape = (task.n_test, task.n_colors + 1, task.n_x, task.n_y)
        # Prediction outputs from take_step are detached CPU tensors.
        self.ema_logits = torch.zeros(shape, device="cpu")
        self.ema_x_mask = torch.zeros((task.n_test, task.n_x), device="cpu")
        self.ema_y_mask = torch.zeros((task.n_test, task.n_y), device="cpu")
        self.scores = {}
        self.candidate_records = {}
        self.first_correct_generation_step = None
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

    def best_slice(self, mask, length):
        lengths = [length] if length is not None else range(1, mask.shape[0] + 1)
        best_score, best_start, best_end = None, 0, 0
        for candidate_length in lengths:
            scores = torch.stack([
                -torch.sum(mask[:offset])
                + torch.sum(mask[offset : offset + candidate_length])
                - torch.sum(mask[offset + candidate_length :])
                for offset in range(mask.shape[0] - candidate_length + 1)
            ])
            candidate_score = torch.max(scores)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_start = int(torch.argmax(scores))
                best_end = best_start + candidate_length
        return best_start, best_end

    def decode(self, logits, x_mask, y_mask):
        color_indices = torch.argmax(logits, dim=1)
        uncertainty = torch.logsumexp(logits, dim=1) - torch.amax(logits, dim=1)
        grids, uncertainties = [], []
        for index in range(self.task.n_test):
            fixed = self.task.in_out_same_size or self.task.all_out_same_size
            x_length = self.task.shapes[self.task.n_train + index][1][0] if fixed else None
            y_length = self.task.shapes[self.task.n_train + index][1][1] if fixed else None
            x0, x1 = self.best_slice(x_mask[index], x_length)
            y0, y1 = self.best_slice(y_mask[index], y_length)
            grid = color_indices[index, x0:x1, y0:y1].numpy()
            grids.append(tuple(tuple(self.task.colors[int(color)] for color in row) for row in grid))
            uncertainties.append(float(torch.mean(uncertainty[index, x0:x1, y0:y1])))
        solution = tuple(grids)
        return solution, float(np.mean(uncertainties))

    def update(self, train_step, outputs):
        logits, x_mask, y_mask = outputs
        logits = logits[self.task.n_train :, :, :, :, 1]
        x_mask = x_mask[self.task.n_train :, :, 1]
        y_mask = y_mask[self.task.n_train :, :, 1]
        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * x_mask
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * y_mask
        candidates = [
            (logits, x_mask, y_mask, 0),
            (self.ema_logits, self.ema_x_mask, self.ema_y_mask, -self.ema_penalty),
        ]
        seen_this_step = set()
        for cand_logits, cand_x, cand_y, penalty in candidates:
            solution, uncertainty = self.decode(cand_logits, cand_x, cand_y)
            early_penalty = self.early_step_penalty if train_step < self.early_step_cutoff else 0
            score = -self.uncertainty_weight * uncertainty + penalty - early_penalty
            key = hash(solution)
            previous_score = self.scores.get(key, -np.inf)
            self.scores[key] = float(np.logaddexp(previous_score, score))
            self._update_top_two(key, solution)
            if solution not in self.candidate_records:
                reference = getattr(self.task, "reference_solutions", None)
                truth_hash = getattr(self.task, "solution_hash", None)
                correct = (solution == reference) if reference is not None else (
                    hash(solution) == truth_hash if truth_hash is not None else None)
                self.candidate_records[solution] = {
                    "hash": str(key), "correct": correct,
                    "first_generation_step": train_step + 1,
                    "occurrences": 0,
                }
                if correct and self.first_correct_generation_step is None:
                    self.first_correct_generation_step = train_step + 1
            if solution not in seen_this_step:
                self.candidate_records[solution]["occurrences"] += 1
                seen_this_step.add(solution)

    def ranked_candidates(self):
        # Stable sorting makes equal-score ties follow first appearance.
        return sorted(self.candidate_records,
                      key=lambda grid: self.scores[hash(grid)], reverse=True)

    def candidate_metrics(self):
        """Report current scores/ranks; best means highest-ranked correct candidate now."""
        ranked = self.ranked_candidates()
        payload = {"candidates/cumulative_unique_count": len(ranked),
                   "candidates/top_ranked_score": self.scores[hash(ranked[0])] if ranked else None}
        if getattr(self.task, "solution_hash", None) is not None:
            correct = [(rank, grid) for rank, grid in enumerate(ranked, 1)
                       if self.candidate_records[grid]["correct"]]
            payload.update({
                "candidates/cumulative_correct_count": len(correct),
                "candidates/best_correct_rank": correct[0][0] if correct else None,
                "candidates/best_correct_score": self.scores[hash(correct[0][1])] if correct else None,
            })
        return payload

    def _update_top_two(self, key, solution):
        if self.solution_most_frequent is None:
            self.solution_most_frequent = solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = solution
        first_key = hash(self.solution_most_frequent)
        second_key = hash(self.solution_second_most_frequent)
        if key != first_key and self.scores[key] >= self.scores.get(second_key, -np.inf):
            self.solution_second_most_frequent = solution
            if self.scores[key] >= self.scores.get(first_key, -np.inf):
                self.solution_second_most_frequent = self.solution_most_frequent
                self.solution_most_frequent = solution
