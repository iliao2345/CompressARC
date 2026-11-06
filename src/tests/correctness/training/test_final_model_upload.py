from types import SimpleNamespace
import sys
from pathlib import Path
from compressarc.train.logging import log_model_checkpoint


def test_final_model_artifact_is_attached_with_aliases(monkeypatch,tmp_path):
    path=tmp_path/"task.pt"
    path.write_bytes(b"checkpoint")
    class Artifact:
        def __init__(self,**kwargs): self.kwargs,self.files=kwargs,[]
        def add_file(self,path,**kwargs):
            assert Path(path).read_bytes()==b"checkpoint"
            self.files.append((path,kwargs))
    monkeypatch.setitem(sys.modules,"wandb",SimpleNamespace(Artifact=Artifact))
    calls=[]
    run=SimpleNamespace(id="test-run",log_artifact=lambda artifact,**kwargs:calls.append((artifact,kwargs)))
    log_model_checkpoint(run,path)
    artifact,options=calls[0]
    assert artifact.kwargs=={"name":"model-test-run","type":"model"}
    assert artifact.files==[(str(path),{"name":"task.pt"})]
    assert options=={"aliases":["final","latest"]}


def test_disabled_run_does_not_upload():
    log_model_checkpoint(None,None)
