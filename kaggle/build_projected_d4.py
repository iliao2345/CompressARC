"""Build projected_d4 full runs at 1500 and 2000 steps (protocol-matched control)."""
import ast
import json
import re
from pathlib import Path

repo = Path(r"C:\Users\teeds\Documents\GitHub\ARC-fafo")
base = json.loads((repo / "kaggle/combined_d4_seed0_cuda_graph_full_run.ipynb").read_text(encoding="utf-8"))


def sub(text, old, new, count=1):
    found = text.count(old)
    assert found == count, f"expected {count} of {old[:60]!r}, found {found}"
    return text.replace(old, new)


for steps in (1500, 2000):
    nb = json.loads(json.dumps(base))
    setup = "".join(nb["cells"][1]["source"])

    # projected_d4 recipe, matching config/models/final_projected_d4_20.yaml
    setup = sub(setup, '"shift_implementation": "d4_tied_conv",', '"shift_implementation": "primitives",')
    setup = sub(setup, '"cummax_implementation": "triton_hard_grouped",', '"cummax_implementation": "primitives",')
    setup = sub(setup, '"direction_share_packed": true', '"direction_share_packed": false')
    setup = sub(setup, '"variant": "combined-d4-conv-sharing"', '"variant": "projected-d4"')
    setup = sub(setup, '"diagnostics_implementation": "triton"', '"diagnostics_implementation": "reference"')
    setup = sub(setup, "['model']['shift_implementation'] == 'd4_tied_conv'",
                       "['model']['shift_implementation'] == 'primitives'")

    if steps != 1500:
        setup = sub(setup, "STEPS = 1500", f"STEPS = {steps}")
        setup = sub(setup, '"iterations": 1500', f'"iterations": {steps}')

    esc = lambda s: s.replace("'", "\\'")
    gate = re.search(r"expected = \{[^}]*\}", setup)
    assert gate, "runner gate not found"
    new_gate = esc("expected = {'shift_implementation': 'primitives', "
                   "'direction_share_implementation': 'd4', 'direction_share_packed': False, "
                   "'cummax_implementation': 'primitives', 'lse_tau': 0.1}")
    setup = setup[:gate.start()] + new_gate + setup[gate.end():]

    assert setup.count("combined-d4-conv-sharing") >= 1
    setup = setup.replace("combined-d4-conv-sharing", "projected_d4")
    nb["cells"][1]["source"] = setup.splitlines(True)

    out = repo / f"kaggle/projected_d4_{steps}steps_full_run.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    check = json.loads(out.read_text(encoding="utf-8"))
    for cell in check["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    body = "".join(check["cells"][1]["source"])
    literals = {n.targets[0].id: n.value for n in ast.parse(body).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    cfg = json.loads(ast.literal_eval(literals["MODEL_CONFIG"]))
    model, training = cfg["model"], cfg["training"]
    runner = None
    for node in ast.walk(ast.parse(body)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text" and node.args
                and "run_arcagi2" in ast.unparse(node.func)):
            runner = node.args[0].value
    start = runner.index("expected = {")
    expected = ast.literal_eval(runner[start + len("expected = "):runner.index("}", start) + 1])
    mismatch = {k: (model.get(k), v) for k, v in expected.items() if model.get(k) != v}

    print(f"{out.name}")
    print("   arm      :", model["shift_implementation"], "+", model["direction_share_implementation"],
          "packed=", model.get("direction_share_packed"), "| cummax", model["cummax_implementation"])
    print("   budget   : STEPS", [l.strip() for l in body.splitlines() if l.startswith("STEPS")][0].split("=")[1].strip(),
          "| config iterations", training["iterations"],
          "| eager/replay", training["cuda_graph_start"], "/",
          training["iterations"] - training["cuda_graph_start"])
    print("   gate     :", "consistent" if not mismatch else mismatch)
    print("   compiles : yes | diagnostics", training["diagnostics_implementation"],
          "| tasks", [l.strip() for l in body.splitlines() if l.startswith("MAX_TASKS")][0].split("#")[0].strip())
