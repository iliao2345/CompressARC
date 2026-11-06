import importlib.util,torch
from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train.optimizer import create_optimizer
from compressarc.train.objective import take_step
spec=importlib.util.spec_from_file_location('before','outputs/objective_before_coverage.py');before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
cfg=load_config(overrides=['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2','model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2','model.softmax_dim=2','model.cummax_dim=2','model.shift_dim=2','model.nonlinear_dim=2'])
problems=[{'train':[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}],'test':[{'input':[[1,0],[0,1]]}]}, {'train':[{'input':[[0,1],[1,0]],'output':[[1],[0],[1]]},{'input':[[1,0]],'output':[[0,1],[1,0]]}],'test':[{'input':[[0,1],[1,0]]}]}]
for problem in problems:
 states=[]
 for step_fn in [before.take_step,take_step]:
  torch.manual_seed(132);task=Task('parity',problem,None);model=ARCCompressor(task,cfg.model);opt=create_optimizer(model.weights_list,cfg.training);rows=[]
  for step in range(3):
   torch.manual_seed(22+step);metrics=step_fn(task,model,opt,step,cfg.training,return_outputs=True)
   rows.append((metrics,[p.detach().clone() for p in model.weights_list],[None if p.grad is None else p.grad.clone() for p in model.weights_list]))
  states.append(rows)
 for old,new in zip(*states):
  for key in old[0]:
   if key=='outputs':
    for x,y in zip(old[0][key],new[0][key]):torch.testing.assert_close(x,y,rtol=0,atol=0)
   else:assert old[0][key]==new[0][key],key
  for xs,ys in zip(old[1:],new[1:]):
   for x,y in zip(xs,ys):
    if x is None:assert y is None
    else:torch.testing.assert_close(x,y,rtol=0,atol=0)
print('Original objective parity: fixed/uncertain shapes x 3 steps; metrics, outputs, gradients and weights bitwise identical.')
