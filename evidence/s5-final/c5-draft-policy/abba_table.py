"""Dev only (outside the PR, C5): the ABBA tables from receipts/c5/{1A,2B,3B,4A}: tip (A) vs tip + C5 (B) on the
standard receipts, and within the B boots d=3 vs the default policy (sweep passes forward then reversed)."""
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

R = Path("/home/sfxnz/projects/data/tf-dsv41/receipts/c5")
TAGS = [t for t in ("1A", "2B", "3B", "4A") if (R / t / "finished.txt").exists()]


def jl(p):
    return [json.loads(x) for x in open(p) if x.startswith("{")] if Path(p).exists() else []


def ab(v):
    a = [v[t] for t in v if t.endswith("A")]
    b = [v[t] for t in v if t.endswith("B")]
    return statistics.mean(b) / statistics.mean(a) if a and b else float("nan")


def summary(p):
    text = open(p).read() if Path(p).exists() else ""
    m = re.search(r"SUMMARY (\[.*\]|\{.*\})", text, re.S)
    if not m:
        return []
    x = json.loads(m.group(1))
    return x if isinstance(x, list) else [x]


print("exactness")
for t in TAGS:
    checks = [x for x in jl(R / t / "pairs.jsonl") if "check" in x]
    bc = [x for x in jl(R / t / "bench_concurrent.log") if "alone" in x]
    print(t, "pairs equal", sum(c["equal"] for c in checks), "/", len(checks),
          "| bench_concurrent alone", [(x["alone"]["equal"], x["alone"]["unequal"], x["alone"]["failed"]) for x in bc],
          "serial", [(x["serial"]["equal"], x["serial"]["unequal"], x["serial"]["failed"]) for x in bc])
shas = {t: sorted({x["token_sha"] for x in jl(R / t / "pairs.jsonl") if "token_sha" in x}) for t in TAGS}
print("pairs reply shas equal across boots:", len({json.dumps(v) for v in shas.values()}) == 1,
      {t: hashlib.sha256(" ".join(v).encode()).hexdigest()[:12] for t, v in shas.items()})

print("\nbench_openai median tok/s")
rows = {}
for t in TAGS:
    for x in jl(R / t / "bench_openai.log"):
        if "decode_tps_median" in x:
            rows.setdefault((x["prompt"], x["temperature"]), {})[t] = x["decode_tps_median"]
print("| Prompt | Temp | " + " | ".join(TAGS) + " | B/A |")
print("|---|---|" + "---|" * (len(TAGS) + 1))
for (p, temp), v in rows.items():
    print(f"| {p} | {temp} | " + " | ".join(f"{v.get(t, 0):.1f}" for t in TAGS) + f" | {ab(v):.3f}x |")

print("\nvLLM recipe cells: bench_decode (greedy, 200 tokens) and L.A.I.L (t=0.2, 512 tokens), median tok/s")
cells = {}
for t in TAGS:
    for x in summary(R / t / "bench_decode.log"):
        cells.setdefault(x["phase"], {})[t] = x["median_decode_tok_s"]
    for x in summary(R / t / "lail.log"):
        cells.setdefault("L.A.I.L", {})[t] = x["median_lail_tok_s"]
print("| Cell | " + " | ".join(TAGS) + " | B/A |")
print("|---|" + "---|" * (len(TAGS) + 1))
for c, v in cells.items():
    print(f"| {c} | " + " | ".join(f"{v.get(t, 0):.1f}" for t in TAGS) + f" | {ab(v):.3f}x |")

print("\nprefill_cold tok/s (Engram pages dropped before the run)")
pf = {}
for t in TAGS:
    for x in jl(R / t / "prefill_cold.log"):
        if "tok_s" in x:
            pf.setdefault(x["length"], {})[t] = x["tok_s"]
print("| Tokens | " + " | ".join(TAGS) + " | B/A |")
print("|---|" + "---|" * (len(TAGS) + 1))
for n, v in sorted(pf.items()):
    print(f"| {n:,} | " + " | ".join(f"{v.get(t, 0):.1f}" for t in TAGS) + f" | {ab(v):.3f}x |")

B = [t for t in TAGS if t.endswith("B") and (R / t / "sweep.jsonl").exists()]
if B:
    print(f"\nwithin the B boots ({', '.join(B)}): d=3 vs the default, each pass forward then reversed (ABBA)")
    out = subprocess.run([sys.executable, str(Path(__file__).with_name("table.py")), *B], capture_output=True,
                         text=True).stdout
    print(out)
