"""Create an offline notebook using an already-extracted Kaggle source dataset."""
import argparse
import base64
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--wheels',type=Path,required=True)
a=p.parse_args()
folder=Path(__file__).resolve().parent
nb=json.loads((folder/'projected_conv_arcagi2.ipynb').read_text())
nb['cells'][0]['source']=['''# Projected conv · ARC-AGI-2 · attached source folder

Enable GPU and keep Internet OFF. Attach the official competition data and your `symarc-cide` Dataset. This notebook uses `/kaggle/input/datasets/teddlo/symarc-cide/src`, copies code to writable storage, and includes its runner and small offline dependency wheels. No source ZIP or Git clone is required.

Run the smoke test before full training. MAX_TASKS=None is configured to attempt all tasks. Up to four visible GPUs process independent tasks. Check smoke_test_result.json and submission_validation.json for executed evidence; notebook source alone does not establish successful execution. Full coverage within Kaggle's time limit is not guaranteed; unprocessed tasks receive explicitly audited fallback guesses. Final output is `/kaggle/working/submission.json`.
''']
wheels={w.name:base64.b64encode(w.read_bytes()).decode() for w in sorted(a.wheels.glob('*.whl'))}
assert any(n.startswith('omegaconf-') for n in wheels)
assert any(n.startswith('antlr4_python3_runtime-') for n in wheels)
setup='''from pathlib import Path
import base64, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
import torch

SOURCE_DIR = Path('/kaggle/input/datasets/teddlo/symarc-cide/src')
CHALLENGES_PATH = None  # Set to the exact official ARC-AGI-2 test challenges path if ambiguous.
STEPS = 1500
SEED = 0
MAX_TASKS = None  # Attempt every task; optional 1 for an initial timing experiment.
TOTAL_BUDGET_HOURS = 6
PER_TASK_LIMIT_HOURS = 2
KEEP_CHECKPOINTS = False
WORKFLOW_START = time.monotonic()

assert torch.cuda.is_available(), 'Enable a GPU accelerator in Kaggle settings.'
assert (SOURCE_DIR/'train.py').is_file(), f'Cannot find {SOURCE_DIR}/train.py'
assert (SOURCE_DIR/'config/models/final_projected_conv_20.yaml').is_file(), 'Upload the current ARC-fafo src folder.'
GPU_WORKERS=min(4,torch.cuda.device_count())
print('GPU workers:',GPU_WORKERS,'PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
print('Four-way parallelism requires four visible GPUs; this runtime exposes',torch.cuda.device_count())
if CHALLENGES_PATH is not None:
    challenges_path=Path(CHALLENGES_PATH)
else:
    # Exclude ARC-AGI-1 data bundled with the attached code dataset.
    matches=[f for f in Path('/kaggle/input').rglob('arc-agi_test_challenges.json')
             if not f.is_relative_to(SOURCE_DIR.parent)]
    if len(matches)!=1:
        raise RuntimeError(f'Set CHALLENGES_PATH to the official ARC-AGI-2 test file. Candidates: {matches}')
    challenges_path=matches[0]
assert challenges_path.is_file(), challenges_path
assert not challenges_path.is_relative_to(SOURCE_DIR.parent), 'Use the official competition dataset, not the code dataset.'
WORK=Path('/kaggle/working/projected_conv_work')
if WORK.exists():
    raise RuntimeError('WORK already exists. Start a fresh session or choose a fresh WORK directory.')
WORK.mkdir()
SRC=WORK/'src'
shutil.copytree(SOURCE_DIR,SRC,ignore=shutil.ignore_patterns('dataset','tests','notebook','__pycache__','.pytest_cache','wandb'))
(WORK/'run_arcagi2.py').write_text(__RUNNER__,encoding='utf-8')
wheel_data=__WHEELS__
(WORK/'wheels').mkdir()
for name,encoded in wheel_data.items():
    (WORK/'wheels'/name).write_bytes(base64.b64decode(encoded))
source_manifest=dict(source=str(SOURCE_DIR),files={f.relative_to(SRC).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in SRC.rglob('*') if f.is_file()})
(WORK/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
print('Code copied into:',SRC)
print('Official challenges:',challenges_path)
print('Runner and offline wheels extracted from this notebook.')
'''.replace('__RUNNER__',repr((folder/'run_arcagi2.py').read_text())).replace('__WHEELS__',repr(wheels))
nb['cells'][1]['source']=setup.splitlines(True)
for i,c in enumerate(nb['cells']):c['id']=f'directory-conv-{i:02d}'
out=folder/'projected_conv_arcagi2_directory.ipynb'
out.write_text(json.dumps(nb,indent=2)+'\n')
print(out)
