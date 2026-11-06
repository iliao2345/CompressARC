"""Build a 2-GPU (T4x2/P100x2) variant of the combined-D4 full run."""
import ast
import json
from pathlib import Path

repo = Path(r"C:\Users\teeds\Documents\GitHub\ARC-fafo")
base = repo / "kaggle/combined_d4_seed0_cuda_graph_full_run.ipynb"
nb = json.loads(base.read_text(encoding="utf-8"))
setup = "".join(nb["cells"][1]["source"])

old_assert = "assert GPU_WORKERS == 4, 'Select an L4x4 GPU session for this full-run notebook.'"
assert setup.count(old_assert) == 1
setup = setup.replace(
    old_assert,
    "assert GPU_WORKERS >= 2, 'This two-GPU build needs at least two visible GPUs.'")

old_tasks = "MAX_TASKS = None  # Attempt every task; optional 1 for an initial timing experiment."
assert setup.count(old_tasks) == 1
setup = setup.replace(
    old_tasks,
    "MAX_TASKS = 120  # Paired subset: the LSE arm already covers these same 120 tasks.")

nb["cells"][1]["source"] = setup.splitlines(True)
out = repo / "kaggle/combined_d4_seed0_graph_T4x2_120tasks.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

check = json.loads(out.read_text(encoding="utf-8"))
body = "".join(check["cells"][1]["source"])
literals = {n.targets[0].id: n.value for n in ast.parse(body).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
cfg = json.loads(ast.literal_eval(literals["MODEL_CONFIG"]))
print("file:", out.name)
print("  MAX_TASKS :", [l.strip() for l in body.splitlines() if l.startswith("MAX_TASKS")][0])
print("  worker gate:", [l.strip() for l in body.splitlines() if "GPU_WORKERS >=" in l][0])
print("  cummax    :", cfg["model"]["cummax_implementation"])
print("  graph     :", cfg["training"]["cuda_graph"], "start", cfg["training"]["cuda_graph_start"])
print("  gpus flag :", [l.strip() for l in "".join(check["cells"][8]["source"]).splitlines() if "--gpus" in l])
