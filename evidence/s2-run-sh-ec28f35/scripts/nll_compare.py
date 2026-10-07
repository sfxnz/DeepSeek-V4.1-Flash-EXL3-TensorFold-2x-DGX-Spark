"""s2: mean NLL of the 40 G1 passages from the s2 scores (data/tf-dsv41/dev/golden_compare.py's nll(), unchanged),
its repeat, and a bitwise comparison of every G1 / G1rep score file with the s0 receipts' (data/tf-dsv41/receipts/
final/tf_scores, TensorFold 0d91389).   python3 nll_compare.py > nll_compare.txt"""
import importlib.util
import json
from pathlib import Path

import numpy as np

D = Path("/home/sfxnz/projects/data/tf-dsv41")
spec = importlib.util.spec_from_file_location("gc", D / "dev/golden_compare.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)
NEW, OLD = D / "receipts/s2-recipe/tf_scores", D / "receipts/final/tf_scores"
g1 = gc.load("G1_teacher_forcing_short.run0.jsonl")
new, rep, old = gc.nll(g1, NEW, "G1"), gc.nll(g1, NEW, "G1rep"), gc.nll(g1, OLD, "G1")
files = sorted(p.name for p in NEW.glob("G1*.npz"))
same = 0
for f in files:
    a, b = np.load(NEW / f), np.load(OLD / f)
    same += sorted(a.files) == sorted(b.files) and all(a[k].tobytes() == b[k].tobytes() and a[k].dtype == b[k].dtype
                                                       for k in a.files)
print(json.dumps({"passages": len(g1), "tokens_scored": new["tokens_scored"], "mean_nll": new["mean_nll"],
                  "repeat_mean_nll": rep["mean_nll"], "s0_0d91389_mean_nll": old["mean_nll"],
                  "files": len(files), "bit_identical_to_s0": same}, indent=1))
