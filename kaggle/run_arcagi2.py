"""Per-task projected-conv training and two-attempt ARC submission export."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data), encoding='utf-8')
    tmp.replace(path)


def valid_grid(grid):
    return (isinstance(grid, (list, tuple)) and 1 <= len(grid) <= 30
            and all(isinstance(row, (list, tuple)) and len(row) == len(grid[0]) for row in grid)
            and 1 <= len(grid[0]) <= 30
            and all(type(v) is int and 0 <= v <= 9 for row in grid for v in row))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Writable source copy containing train.py')
    p.add_argument('--challenges', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--task', action='append')
    p.add_argument('--budget-seconds', type=float, default=6*3600)
    p.add_argument('--per-task-seconds', type=float, default=2*3600)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gpus', type=int, default=1, help='Independent task workers; one process per GPU')
    p.add_argument('--keep-checkpoints', action='store_true')
    p.add_argument('--wandb-mode', choices=['disabled','offline'], default='disabled')
    p.add_argument('--wandb-tag', default='kaggle run 1')
    a = p.parse_args()
    if a.steps < 1 or a.budget_seconds <= 0 or a.per_task_seconds <= 0:
        p.error('steps and time limits must be positive')
    if a.max_tasks is not None and a.max_tasks < 1:
        p.error('max-tasks must be positive')
    if not 1 <= a.gpus <= 4:
        p.error('gpus must be 1..4')
    if a.device.startswith('cuda'):
        import torch
        first_device = int(a.device.split(':')[1]) if ':' in a.device else 0
        if not torch.cuda.is_available() or first_device+a.gpus > torch.cuda.device_count():
            p.error('Requested GPUs are not visible in this runtime')
        devices = [f'cuda:{first_device+i}' for i in range(a.gpus)]
    else:
        devices = [a.device]*a.gpus
    source = a.source.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'submission.json').exists():
        raise SystemExit('Output already contains submission.json; choose a fresh output directory.')
    problems = json.loads(a.challenges.read_text())
    if not isinstance(problems, dict) or not problems:
        raise ValueError('Expected task-ID -> challenge mapping')
    # Even if public task JSON contains answers, never pass test labels to the model.
    problems = copy.deepcopy(problems)
    for task, problem in problems.items():
        if not problem.get('train') or not problem.get('test'):
            raise ValueError(f'Invalid challenge: {task}')
        for example in problem['test']:
            example.pop('output', None)
    dataset = source/'dataset'
    dataset.mkdir(exist_ok=True)
    atomic_json(dataset/'arc-agi_test_challenges.json', problems)
    selected = a.task or list(problems)
    if len(set(selected)) != len(selected) or not set(selected) <= set(problems):
        raise ValueError('Selected task IDs must be distinct and present in challenges')
    if a.max_tasks is not None:
        selected = selected[:a.max_tasks]
    # A valid, explicitly reported fallback for tasks not completed within budget.
    submission = {t:[{'attempt_1':copy.deepcopy(e['input']), 'attempt_2':copy.deepcopy(e['input'])}
                     for e in problem['test']] for t,problem in problems.items()}
    audit = {t:dict(status='fallback_not_attempted', predictions='copy_input_fallback') for t in problems}
    atomic_json(out/'submission.json', submission)
    atomic_json(out/'task_status.json', audit)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', WANDB_MODE=a.wandb_mode)
    if a.wandb_mode == 'offline':
        (out/'wandb').mkdir(exist_ok=True)
        env['WANDB_DIR'] = str(out/'wandb')
        env['WANDB_DATA_DIR'] = str(out/'wandb_data')
        env['WANDB_CACHE_DIR'] = str(out/'wandb_cache')
    manifest = dict(source=str(source),challenges=str(a.challenges),steps=a.steps,seed=a.seed,
                    device=a.device,selected_tasks=selected,budget_seconds=a.budget_seconds,
                    per_task_seconds=a.per_task_seconds,devices=devices,source_metadata=None)
    meta = source.parent/'source_manifest.json'
    if meta.exists():
        manifest['source_metadata'] = json.loads(meta.read_text())
    atomic_json(out/'run_manifest.json', manifest)
    start = time.monotonic()
    state_lock = threading.Lock()
    def run_one(task, device):
        remaining = a.budget_seconds-(time.monotonic()-start)-30
        if remaining <= 0:
            return False
        checkpoint_dir = out/'checkpoints'/task
        checkpoint = checkpoint_dir/f'{task}.pt'
        cmd = [sys.executable,str(source/'train.py'),'--config',str(source/'config/models/final_projected_conv_20.yaml'),
               '--task',task,'--set','training.split=test','--set',f'training.iterations={a.steps}',
               '--set',f'training.seed={a.seed}','--set',f'training.device={device}',
               '--set','logging.wandb='+str(a.wandb_mode != 'disabled').lower(),
               '--set',f'logging.wandb_mode={a.wandb_mode}',
               '--set','logging.wandb_tags='+json.dumps([a.wandb_tag]),'--save-checkpoints','--output-dir',str(checkpoint_dir)]
        print(f'Training {task} on {device}: {a.steps} steps, timeout {min(remaining,a.per_task_seconds):.0f}s',flush=True)
        task_start = time.monotonic()
        log_path = out/f'{task}.log'
        try:
            with log_path.open('w') as log:
                subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                               check=True,timeout=min(remaining,a.per_task_seconds))
            import torch
            payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
            first = payload['selection']['top_1']
            second = payload['selection']['top_2']
            if first is None or len(first) != len(problems[task]['test']):
                raise ValueError('Incorrect number of selected test grids')
            if second is None:
                second = first
            if len(second) != len(first):
                raise ValueError('Incorrect number of second attempts')
            guesses = []
            for grid1, grid2 in zip(first,second):
                if not valid_grid(grid1) or not valid_grid(grid2):
                    raise ValueError('Invalid selected grid')
                guesses.append({'attempt_1':[list(row) for row in grid1], 'attempt_2':[list(row) for row in grid2]})
            result = dict(status='model_completed',predictions='projected_conv',completed_steps=payload['completed_steps'])
            if not a.keep_checkpoints:
                checkpoint.unlink()
        except subprocess.TimeoutExpired:
            result = dict(status='fallback_timeout',predictions='copy_input_fallback')
        except Exception as exc:
            result = dict(status='fallback_error',predictions='copy_input_fallback',error=str(exc),log=str(log_path))
        result.update(elapsed_seconds=time.monotonic()-task_start,device=device)
        with state_lock:
            if result['status'] == 'model_completed':
                submission[task] = guesses
            audit[task] = result
            atomic_json(out/'submission.json', submission)
            atomic_json(out/'task_status.json', audit)
        print(task, result,flush=True)
        return True
    pending = Queue()
    for task in selected:
        pending.put(task)
    def worker(device):
        while True:
            try:
                task = pending.get_nowait()
            except Empty:
                return
            if not run_one(task,device):
                return
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker,device) for device in devices]
        for future in futures:
            future.result()
    assert set(submission) == set(problems)
    for task, attempts in submission.items():
        assert len(attempts) == len(problems[task]['test'])
        for pair in attempts:
            assert set(pair) == {'attempt_1','attempt_2'}
            assert valid_grid(pair['attempt_1']) and valid_grid(pair['attempt_2'])
    n = sum(r['status']=='model_completed' for r in audit.values())
    atomic_json(out/'submission_validation.json',dict(schema_valid=True,tasks=len(problems),
                test_inputs=sum(len(p['test']) for p in problems.values()),model_completed=n,
                fallback_tasks=len(problems)-n,full_model_coverage=n==len(problems),
                submission_sha256=hashlib.sha256((out/'submission.json').read_bytes()).hexdigest()))
    print(f'Model completed {n}/{len(problems)} tasks. Remaining tasks have explicitly logged fallback outputs.',flush=True)
    print('Submission:',out/'submission.json',flush=True)


if __name__ == '__main__':
    main()
