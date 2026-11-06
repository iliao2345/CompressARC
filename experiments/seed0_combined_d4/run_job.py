"""Run one combined D4 seed-0 model; no model/checkpoint weights saved."""
import argparse,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path

def command(folder,index,device='cuda:0'):
    manifest=json.loads((folder/'manifest.json').read_text())
    if not 0<=index<len(manifest['jobs']):raise ValueError('Index outside remaining manifest')
    job=manifest['jobs'][index]
    repo=folder.parents[1]
    output=repo/'outputs'/folder.name/os.environ.get('SLURM_ARRAY_JOB_ID','manual')/job['task']/job['variant']/f"seed_{job['seed']}"
    cmd=[sys.executable,str(repo/'src/train.py'),'--config',str(folder/job['config']),
         '--task',job['task'],'--set','training.device='+device,
         '--set','logging.save_model=false','--output-dir',str(output)]
    return cmd,job,output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);folder=Path(__file__).resolve().parent
    cmd,job,output=command(folder,a.index,a.device)
    repo=folder.parents[1]
    print(shlex.join(cmd),flush=True)
    if a.dry_run:return cmd
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    output.mkdir(parents=True,exist_ok=True)
    provenance=dict(job=job,code_commit=commit,config_sha256=hashlib.sha256((folder/job['config']).read_bytes()).hexdigest(),
                    model_saving=False,checkpoint_resume=False,command=cmd)
    (output/'execution.json').write_text(json.dumps(provenance,indent=2))
    print('Code commit: '+commit,flush=True)
    cmd+=['--set','experiment.code_commit='+commit]
    subprocess.run(cmd,cwd=repo,check=True)

if __name__=='__main__':main()
