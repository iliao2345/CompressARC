"""Re-embed the current gate sources and exempt the two files we always replace."""
import ast
import json
from pathlib import Path

repo = Path(r"C:\Users\teeds\Documents\GitHub\ARC-fafo")
SKIP = "('cuda_graph_preflight.py','cuda_graph_lse_validation.py')"

for name in ("projected_d4_1500steps_full_run.ipynb", "projected_d4_2000steps_full_run.ipynb"):
    path = repo / "kaggle" / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    cell = nb["cells"][3]
    lines = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
    text = "".join(lines)
    tree = ast.parse(text)
    nodes = {n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign)}

    def end_offset(node):
        return sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset

    expected = ast.literal_eval(nodes["EXPECTED"].value)
    overlay = ast.literal_eval(nodes["OVERLAY"].value)
    overlay["cuda_graph_preflight.py"] = (repo / "src/cuda_graph_preflight.py").read_text(encoding="utf-8")
    overlay["cuda_graph_lse_validation.py"] = (repo / "src/cuda_graph_lse_validation.py").read_text(encoding="utf-8")

    head = text[: sum(len(line) for line in lines[: nodes["EXPECTED"].lineno - 1])]
    tail = text[end_offset(nodes["OVERLAY"]):]

    guard = "    path=SRC/name\n    if path.exists():"
    assert tail.count(guard) == 1, "guard loop not found"
    tail = tail.replace(guard, f"    path=SRC/name\n    if name in {SKIP}:continue\n    if path.exists():")

    new_text = (head + "EXPECTED=" + repr(expected) + "\nOVERLAY=" + repr(overlay) + tail)
    compile(new_text, "cell3", "exec")
    cell["source"] = new_text.splitlines(True)
    path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    check = json.loads(path.read_text(encoding="utf-8"))
    body = "".join(check["cells"][3]["source"])
    for other in check["cells"]:
        if other["cell_type"] == "code":
            ast.parse("".join(other["source"]))
    again = ast.literal_eval({n.targets[0].id: n for n in ast.parse(body).body
                              if isinstance(n, ast.Assign)}["OVERLAY"].value)
    print(name)
    print("   overlay files :", len(again))
    print("   preflight     :", len(again["cuda_graph_preflight.py"].splitlines()), "lines, matches repo:",
          again["cuda_graph_preflight.py"] == (repo / "src/cuda_graph_preflight.py").read_text(encoding="utf-8"))
    print("   lse gate      :", "matches repo:",
          again["cuda_graph_lse_validation.py"] == (repo / "src/cuda_graph_lse_validation.py").read_text(encoding="utf-8"))
    print("   guard exempts :", SKIP in body)
