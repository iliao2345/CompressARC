"""Build the two deadline-sized run notebooks from the patched base notebook."""
import json
from pathlib import Path

repo = Path(r"C:\Users\teeds\Documents\GitHub\ARC-fafo")
base = repo / "kaggle/combined_d4_lse_cuda_graph_full_run.ipynb"
notebook = json.loads(base.read_text(encoding="utf-8"))

TASKS = 4
MAX_TASKS_OLD = "MAX_TASKS = None  # Attempt every task; optional 1 for an initial timing experiment."
MAX_TASKS_NEW = f"MAX_TASKS = {TASKS}  # One task per GPU worker; sized to finish inside the submission window."


def build(name, cummax, variant, title_replace=None):
    nb = json.loads(json.dumps(notebook))
    cells = nb["cells"]
    setup = "".join(cells[1]["source"])

    assert MAX_TASKS_OLD in setup, "MAX_TASKS line changed"
    setup = setup.replace(MAX_TASKS_OLD, MAX_TASKS_NEW)

    if cummax != "triton":
        old = '"cummax_implementation": "triton",\\n    "lse_tau": 0.1,'
        new = f'"cummax_implementation": "{cummax}",'
        assert setup.count(old) == 1, f"model config block not found for {name}"
        setup = setup.replace(old, new)
        # The embedded runner re-verifies the architecture before training and
        # refuses anything that is not the LSE config.
        runner_old = "\\'cummax_implementation\\': \\'triton\\', \\'lse_tau\\': 0.1"
        assert setup.count(runner_old) == 1, f"runner config gate not found for {name}"
        setup = setup.replace(runner_old, f"\\'cummax_implementation\\': \\'{cummax}\\'")
        label = "triple-d4-conv-sharing-lse"
        assert setup.count(label) >= 1
        setup = setup.replace(label, "triple-d4-conv-sharing-hard")
    # Publish a partial but audited submission instead of nothing when one task
    # fails: an all-fallback submission is still refused.
    gate_old = "if any(audit[task][\\'status\\'] != \\'model_completed\\' for task in selected):"
    gate_new = "if not any(audit[task][\\'status\\'] == \\'model_completed\\' for task in selected):"
    assert setup.count(gate_old) == 1, f"completion gate not found for {name}"
    setup = setup.replace(gate_old, gate_new)
    cells[1]["source"] = setup.splitlines(True)

    preflight = "".join(cells[4]["source"])
    assert "'combined_lse'" in preflight
    cells[4]["source"] = preflight.replace("'--variant','combined_lse'", f"'--variant','{variant}'").splitlines(True)

    if title_replace:
        title = "".join(cells[0]["source"])
        for old, new in title_replace:
            assert old in title
            title = title.replace(old, new)
        cells[0]["source"] = title.splitlines(True)

    out = repo / "kaggle" / name
    out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    check = json.loads(out.read_text(encoding="utf-8"))
    body = "".join(check["cells"][1]["source"])
    import ast

    literals = {node.targets[0].id: node.value for node in ast.parse(body).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    config = json.loads(ast.literal_eval(literals["MODEL_CONFIG"]))
    assert json.loads(ast.literal_eval(literals["MODEL_CONFIG"])) == config
    print(name)
    print("  max_tasks      :", body.split("MAX_TASKS =", 1)[1].split("  #")[0].strip())
    print("  cummax         :", config["model"]["cummax_implementation"])
    print("  lse_tau        :", config["model"].get("lse_tau"))
    print("  direction_share:", config["model"]["direction_share_implementation"],
          "packed=", config["model"].get("direction_share_packed"))
    print("  shift          :", config["model"]["shift_implementation"])
    print("  cuda_graph     :", config["training"]["cuda_graph"], "start", config["training"]["cuda_graph_start"])
    print("  preflight arm  :", "".join(check["cells"][4]["source"]).split("'--variant',")[1].split("]")[0].strip())
    print("  cells          :", len(check["cells"]))
    return out


build("run_a_lse_4tasks.ipynb", "triton", "combined_lse",
      [("+ LSE + CUDA Graphs", "+ LSE + CUDA Graphs [4 tasks]")])
build("run_b_hard_4tasks.ipynb", "triton_hard_grouped", "combined",
      [("+ LSE + CUDA Graphs", "+ hard cummax + CUDA Graphs [4 tasks]")])
