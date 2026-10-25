"""Run the notebook's code cells as a script in a scratch folder with QUICK = True.

A cheap way to catch errors before a long run: python tools/dry_run.py <folder containing the split file>
"""
import builtins
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(sys.argv[1])

builtins.display = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: None

nb = json.loads((ROOT / "notebooks" / "rgcn_fraud.ipynb").read_text(encoding="utf-8"))
scope = {}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.startswith("%%writefile"):
        continue
    src = src.replace("QUICK = False", "QUICK = True")
    print(f"\n----- cell {i} -----")
    try:
        exec(compile(src, f"cell{i}", "exec"), scope)
    except Exception:
        print("FAILED in cell", i)
        raise
print("\ndry run finished")
