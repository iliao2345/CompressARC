import argparse,json,os,subprocess,sys,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--step-timing',choices=['off','wall','phases']);a=p.parse_args()
folder=Path(__file__).resolve().parent
manifest=json.loads((folder/'manifest.json').read_text())
if not 0<=a.index<len(manifest['jobs']):p.error('index must be 0..19')
roots=[Path.cwd(),*folder.parents]
repo=next((r for r in roots if (r/'src/train.py').is_file()),None)
if repo is None:raise SystemExit('Run from ARC-fafo root, or place this folder inside ARC-fafo')
j=manifest['jobs'][a.index]
out=repo/'outputs'/folder.name/'checkpoints'/os.environ.get('SLURM_ARRAY_JOB_ID','manual')/j['task']/j['variant']/f"seed_{j['seed']}"
cmd=[sys.executable,str(repo/'src/train.py'),'--config',str(folder/j['config']),'--task',j['task'],'--set','training.device=cuda:0','--save-checkpoints','--output-dir',str(out)]
if a.step_timing is not None:cmd.extend(['--set','training.step_timing='+json.dumps(a.step_timing)])
print(shlex.join(cmd),flush=True)
if not a.dry_run:subprocess.run(cmd,cwd=repo,check=True)
