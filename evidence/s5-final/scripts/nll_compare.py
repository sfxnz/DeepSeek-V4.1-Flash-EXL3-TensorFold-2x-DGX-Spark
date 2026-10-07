"""s5: mean NLL of the 40 G1 passages from the s5 scores (data/tf-dsv41/dev/golden_compare.py's nll(), unchanged), its
repeat, and a comparison of every G1 / G1rep score file with s2's (data/tf-dsv41/receipts/s2-recipe/tf_scores,
TensorFold ec28f35, itself bit-identical to s0's 0d91389): bitwise per array, and the largest absolute difference of
the per-position log-probabilities where they differ.   python3 nll_compare.py > nll_compare.txt"""
import importlib.util
import json
from pathlib import Path

import numpy as np

D = Path("/home/sfxnz/projects/data/tf-dsv41")
spec = importlib.util.spec_from_file_location("gc", D / "dev/golden_compare.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)
NEW, OLD = D / "receipts/s5-recipe/tf_scores", D / "receipts/s2-recipe/tf_scores"
g1 = gc.load("G1_teacher_forcing_short.run0.jsonl")
new, rep, old = gc.nll(g1, NEW, "G1"), gc.nll(g1, NEW, "G1rep"), gc.nll(g1, OLD, "G1")
files = sorted(p.name for p in NEW.glob("G1*.npz"))
same, arrays_differ, max_dlp, top1_moves = 0, {}, 0.0, 0
for f in files:
    a, b = np.load(NEW / f), np.load(OLD / f)
    eq = sorted(a.files) == sorted(b.files) and all(a[k].tobytes() == b[k].tobytes() and a[k].dtype == b[k].dtype
                                                    for k in a.files)
    same += eq
    if not eq:
        for k in a.files:
            if a[k].tobytes() != b[k].tobytes():
                arrays_differ[k] = arrays_differ.get(k, 0) + 1
        max_dlp = max(max_dlp, float(np.max(np.abs(a["lp"] - b["lp"]))))
        top1_moves += int(np.sum(a["top_ids"][:, 0] != b["top_ids"][:, 0]))
print(json.dumps({"passages": len(g1), "tokens_scored": new["tokens_scored"], "mean_nll": new["mean_nll"],
                  "repeat_mean_nll": rep["mean_nll"], "s2_ec28f35_mean_nll": old["mean_nll"],
                  "files": len(files), "bit_identical_to_s2": same, "arrays_that_differ": arrays_differ,
                  "max_abs_dlogprob": max_dlp, "top1_changes": top1_moves}, indent=1))
