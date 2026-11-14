import json
from pathlib import Path

folder=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# Projected convolution · ARC-AGI-2 Kaggle workflow

Uses the existing ARC-fafo projected-conv model: tied convolution, original direction-share and cummax layers, strict constraints, 4 layers, training seed 0. Each puzzle is trained independently on its provided input/output examples. No test answers are used.

**Setup:** enable a Kaggle GPU accelerator and attach (1) the official ARC-AGI-2 competition data and (2) a private Kaggle Dataset containing `arc_fafo_projected_conv_source.zip` from this package. The ZIP includes code and offline OmegaConf dependencies. Internet is unnecessary during notebook execution. Do not upload API keys.

Run the smoke test first. Full-dataset execution is configured with MAX_TASKS=None and limited by runtime: 1,500 steps per puzzle can exceed the notebook budget. The audit distinguishes trained predictions from copy-input fallbacks. This is an execution workflow, not a claim of benchmark accuracy or a guaranteed full-dataset completion.
''')
code('''from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, time, zipfile
import torch

assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle notebook settings."
print("GPU:", torch.cuda.get_device_name(0))
print("Python:", sys.version, "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
GPU_WORKERS = min(4, torch.cuda.device_count())
print("Independent GPU task workers:", GPU_WORKERS)

# Set explicitly if more than one matching file is attached.
SOURCE_ZIP = None
CHALLENGES_PATH = None
STEPS = 1500
SEED = 0
MAX_TASKS = None           # Use 1 or 2 for a first full-length timing test; None attempts all.
TOTAL_BUDGET_HOURS = 6     # Choose below your competition/session limit; includes setup below.
PER_TASK_LIMIT_HOURS = 2   # A timeout leaves a clearly labeled fallback, not a trained prediction.
KEEP_CHECKPOINTS = False
WORKFLOW_START = time.monotonic()

def choose_one(name, explicit):
    if explicit is not None:
        result = Path(explicit)
        assert result.is_file(), result
        return result
    matches = list(Path('/kaggle/input').rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}; found {matches}. Set the path explicitly.")
    return matches[0]

source_zip = choose_one('arc_fafo_projected_conv_source.zip', SOURCE_ZIP)
challenges_path = choose_one('arc-agi_test_challenges.json', CHALLENGES_PATH)
WORK = Path('/kaggle/working/projected_conv_work')
if WORK.exists():
    raise RuntimeError('Working directory already exists. Start a fresh session or choose a new WORK path.')
WORK.mkdir()
with zipfile.ZipFile(source_zip) as archive:
    for name in archive.namelist():
        target = (WORK/name).resolve()
        if not target.is_relative_to(WORK.resolve()):
            raise ValueError(f'Unsafe ZIP member: {name}')
    archive.extractall(WORK)
SRC = WORK/'src'
print('Data:', challenges_path)
print('Source manifest:', json.loads((WORK/'source_manifest.json').read_text()))
''')
code('''# Keep Kaggle's GPU-enabled PyTorch. Install only small missing dependencies offline.
import yaml, numpy, tqdm, matplotlib
assert importlib.util.find_spec('wandb') is not None, 'Your Kaggle image needs wandb installed; attach offline wheels if absent.'
if importlib.util.find_spec('omegaconf') is None or importlib.util.find_spec('antlr4') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                    str(WORK/'wheels'), 'omegaconf==2.3.0', 'antlr4-python3-runtime==4.9.3'], check=True)
from omegaconf import OmegaConf
print('OmegaConf available. Existing PyTorch retained:', torch.__version__)
problems = json.loads(challenges_path.read_text())
assert isinstance(problems, dict) and problems
assert all(p.get('train') and p.get('test') for p in problems.values())
print('Tasks:', len(problems), 'Test inputs:', sum(len(p['test']) for p in problems.values()))
Path('/kaggle/working/runtime_versions.json').write_text(json.dumps(dict(
    python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
    numpy=numpy.__version__, gpu=torch.cuda.get_device_name(0),
    challenges_sha256=hashlib.sha256(challenges_path.read_bytes()).hexdigest()), indent=2))
''')
md('''## 1. Smoke test
Runs two optimizer steps on one task per GPU worker, exercising GPU preprocessing, projected-conv forward/backward, candidate selection, checkpoint save/read, and submission export. It writes only to `smoke/`; it is not the final submission. If this fails, inspect its task log before proceeding.
''')
code('''SMOKE = WORK/'smoke'
smoke_start = time.monotonic()
subprocess.run([sys.executable, str(WORK/'run_arcagi2.py'), '--source', str(SRC),
                '--challenges', str(challenges_path), '--output', str(SMOKE),
                '--steps', '2', '--seed', str(SEED), '--max-tasks', str(GPU_WORKERS),
                '--budget-seconds', '600', '--per-task-seconds', '540', '--gpus', str(GPU_WORKERS)], check=True)
smoke_status = json.loads((SMOKE/'task_status.json').read_text())
smoke_completed = [t for t,s in smoke_status.items() if s['status']=='model_completed']
if len(smoke_completed) != min(GPU_WORKERS, len(problems)):
    for log in SMOKE.glob('*.log'):
        print(log.read_text()[-12000:])
    raise RuntimeError('Smoke test failed; do not run the full workload yet.')
smoke_result = dict(status='passed', tasks=smoke_completed, gpu_workers=GPU_WORKERS, devices=sorted({smoke_status[t]['device'] for t in smoke_completed}), elapsed_seconds=time.monotonic()-smoke_start)
Path('/kaggle/working/smoke_test_result.json').write_text(json.dumps(smoke_result, indent=2))
print('Smoke test passed:', smoke_result)
''')
md('''## 2. Train and generate guesses
A shared task queue dispatches independent tasks to up to four GPUs. The task order follows the challenges JSON, without performance filtering. Check the actual competition's hardware and runtime limits before choosing the budget. The default 1,500-step configuration is unchanged, but PyTorch/CUDA/hardware can differ from M3 and affect the trajectory. Training starts from scratch for every task; the script does not resume training from inference checkpoints.
''')
code('''RUN = WORK/'predictions'
remaining = TOTAL_BUDGET_HOURS*3600 - (time.monotonic()-WORKFLOW_START) - 120
assert remaining > 60, 'Insufficient remaining budget.'
command = [sys.executable, str(WORK/'run_arcagi2.py'), '--source', str(SRC),
           '--challenges', str(challenges_path), '--output', str(RUN),
           '--steps', str(STEPS), '--seed', str(SEED),
           '--budget-seconds', str(remaining), '--per-task-seconds', str(PER_TASK_LIMIT_HOURS*3600), '--gpus', str(GPU_WORKERS)]
command += ['--wandb-mode', 'offline', '--wandb-tag', 'kaggle run 1', '--keep-checkpoints']
if MAX_TASKS is not None:
    command += ['--max-tasks', str(MAX_TASKS)]
if KEEP_CHECKPOINTS:
    command += ['--keep-checkpoints']
print('Starting:', command)
subprocess.run(command, check=True)
''')
md('''## 3. Validate and publish `submission.json`
There must be two attempts for every test input. Completed tasks use the tracker's final top two, not reference-answer-based selection. Unprocessed, timed-out, or failed tasks use copy-input fallback guesses and are counted explicitly. The fallback only keeps the file structurally complete; it does not indicate that the model processed the task.
''')
code('''submission = json.loads((RUN/'submission.json').read_text())
status = json.loads((RUN/'task_status.json').read_text())
assert set(submission) == set(problems)
for task, examples in problems.items():
    assert len(submission[task]) == len(examples['test'])
    assert all(set(pair)=={'attempt_1','attempt_2'} for pair in submission[task])
sample_candidates = list(challenges_path.parent.glob('*sample_submission*.json'))
if len(sample_candidates) == 1:
    sample = json.loads(sample_candidates[0].read_text())
    assert set(sample) == set(submission), 'Sample submission and challenges task IDs differ.'
    assert all(len(sample[t])==len(submission[t]) for t in sample)
completed = sum(s['status']=='model_completed' for s in status.values())
print(f'MODEL COVERAGE: {completed}/{len(problems)} tasks; FALLBACK: {len(problems)-completed}.')
assert completed > 0, 'No task completed; refusing to publish an all-fallback submission.'
shutil.copy2(RUN/'submission.json', '/kaggle/working/submission.json')
shutil.copy2(RUN/'task_status.json', '/kaggle/working/task_status.json')
shutil.copy2(RUN/'run_manifest.json', '/kaggle/working/run_manifest.json')
shutil.copy2(RUN/'submission_validation.json', '/kaggle/working/submission_validation.json')
print('Validation:', json.loads((RUN/'submission_validation.json').read_text()))
print('Offline logs:', shutil.make_archive('/kaggle/working/kaggle_run_1_logs', 'zip', RUN))
print('Output ready: /kaggle/working/submission.json')
print('Save a notebook version, inspect task_status.json and logs, then use Kaggle submission controls.')
''')
for index,cell in enumerate(cells):cell['id']=f'projected-conv-{index:02d}'
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
(folder/'projected_conv_arcagi2.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
print('Notebook generated')
