"""Bundle current ARC-fafo source and offline helpers for a Kaggle Dataset."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root',type=Path,required=True)
p.add_argument('--wheels',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
src=a.repo_root.resolve()/'src'
if not (src/'train.py').is_file():raise SystemExit('Expected ARC-fafo/src/train.py')
wheels=list(a.wheels.glob('*.whl'))
if not any(w.name.startswith('omegaconf-2.3.0') for w in wheels):raise SystemExit('OmegaConf 2.3.0 wheel missing')
if not any(w.name.startswith('antlr4_python3_runtime-4.9.3') for w in wheels):raise SystemExit('ANTLR 4.9.3 wheel missing')
files=[]
for file in sorted(src.rglob('*')):
    rel=file.relative_to(src)
    if file.is_file() and not set(rel.parts)&{'dataset','tests','notebook','__pycache__','.pytest_cache','wandb'} and file.suffix in {'.py','.yaml','.yml','.json','.txt'}:
        files.append((file,'src/'+rel.as_posix()))
manifest=dict(description='Projected-conv source bundle; no bundled ARC dataset, keys, or W&B run files',files={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in files})
try:
    manifest['git_commit']=subprocess.check_output(['git','-C',str(a.repo_root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    manifest['git_tracked_changes']=subprocess.check_output(['git','-C',str(a.repo_root),'diff','--name-only','HEAD','--','src'],text=True,stderr=subprocess.DEVNULL).splitlines()
except Exception:
    manifest['git_commit']=None
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as archive:
    for path,name in files:archive.write(path,name)
    archive.write(Path(__file__).with_name('run_arcagi2.py'),'run_arcagi2.py')
    for wheel in wheels:archive.write(wheel,'wheels/'+wheel.name)
    archive.writestr('source_manifest.json',json.dumps(manifest,indent=2))
print(json.dumps(dict(bundle=str(a.output),source_files=len(files),bytes=a.output.stat().st_size,git_commit=manifest['git_commit'])))
