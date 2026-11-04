"""Build 2000-step full runs for both arms and verify both iteration knobs agree."""
import ast
import json
from pathlib import Path

repo = Path(r"C:\Users\teeds\Documents\GitHub\ARC-fafo")

JOBS = [
    ("kaggle/combined_d4_lse_cuda_graph_full_run.ipynb", "kaggle/lse_2000steps_full_run.ipynb"),
    ("kaggle/combined_d4_seed0_cuda_graph_full_run.ipynb", "kaggle/combined_2000steps_full_run.ipynb"),
]

for source, target in JOBS:
    nb = json.loads((repo / source).read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][1]["source"])
    assert setup.count("STEPS = 1500") == 1, source
    assert setup.count('"iterations": 1500') == 1, source
    setup = setup.replace("STEPS = 1500", "STEPS = 2000").replace('"iterations": 1500', '"iterations": 2000')
    nb["cells"][1]["source"] = setup.splitlines(True)
    out = repo / target
    out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    check = json.loads(out.read_text(encoding="utf-8"))
    body = "".join(check["cells"][1]["source"])
    literals = {n.targets[0].id: n.value for n in ast.parse(body).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    cfg = json.loads(ast.literal_eval(literals["MODEL_CONFIG"]))
    training = cfg["training"]
    print(out.name)
    print("   steps", [l.strip() for l in body.splitlines() if l.startswith("STEPS")][0].split("=")[1].strip(),
          "| config iterations", training["iterations"])
    print("   eager/replay expected:", training["cuda_graph_start"], "/",
          training["iterations"] - training["cuda_graph_start"])
    print("   cummax", cfg["model"]["cummax_implementation"],
          "| reasoning: same arm as the 1500-step run, only the budget changes")
