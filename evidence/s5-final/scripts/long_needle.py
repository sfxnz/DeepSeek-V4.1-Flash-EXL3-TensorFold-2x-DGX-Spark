"""s2: one long needle prompt through the vLLM sibling's quality_eval needle builder (unmodified, imported), against
the live server. Same document recipe as quality_eval's run_needle (filler paragraphs, one vault-code note at
DEPTH, sized to land in [0.97, 1.0] x (LENGTH - 64) prompt tokens by the server's /tokenize), one depth per call,
greedy, max_tokens 32, thinking off. Prints one JSON line: tokens, prefill_s from the tensorfold block, wall time,
found.

  python3 long_needle.py LENGTH DEPTH [--url http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/tests")
import quality_eval as qe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("length", type=int)
ap.add_argument("depth", type=float)
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
c = qe.Client(a.url, None, timeout=4 * 3600)
seed = a.seed or (a.length // 1024) * 10 + int(a.depth * 10)
name, code = qe.needle_code(seed)
header = f"Archive s2-{seed}. Field notes follow."
sample = "\n\n".join(qe.filler_paragraphs(1, 40))
ratio = c.count(sample) / len(sample)
budget = a.length - 64
n = max(4, int(budget / ratio / 480))
paras = qe.filler_paragraphs(seed, int(n * 1.3))
for _ in range(6):   # land in [0.97, 1.0] x budget, as run_needle does
    ntok = c.count(qe.needle_doc(paras[:n], a.depth, name, code, header))
    if 0.97 * budget <= ntok <= budget:
        break
    n = max(4, int(n * budget / ntok * (0.985 if ntok > budget else 1.0)))
    if n > len(paras):
        paras = qe.filler_paragraphs(seed, int(n * 1.3))
doc = qe.needle_doc(paras[:n], a.depth, name, code, header)
q = f"{doc}\n\nWhat is the vault code assigned to {name}? Reply with only the code."
print(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "length": a.length, "depth": a.depth,
                  "doc_tokens": ntok}), flush=True)
t0 = time.time()
try:
    out = c.chat(q, max_tokens=32)
except Exception as exc:  # noqa: BLE001
    print(json.dumps({"length": a.length, "depth": a.depth, "error": f"{type(exc).__name__}: {exc}"[:400],
                      "s": round(time.time() - t0, 1)}))
    sys.exit(1)
s = out.get("tensorfold") or {}
text = out["choices"][0]["message"].get("content") or ""
row = {"length": a.length, "depth": a.depth, "prompt_tokens": out["usage"]["prompt_tokens"],
       "cached": out["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
       "completion_tokens": out["usage"]["completion_tokens"], "prefill_s": s.get("prefill_s"),
       "prefill_tok_s": round(out["usage"]["prompt_tokens"] / s["prefill_s"], 1) if s.get("prefill_s") else None,
       "wall_s": round(time.time() - t0, 1), "found": code in text.upper(), "code": code, "answer": text.strip()[:60],
       "token_sha": s.get("token_sha"), "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps(row), flush=True)
sys.exit(0 if row["found"] else 2)
