"""N11 (mp2): sampled decode cells at c = 1, 2, 4. The frozen ruler's prose prompt, streamed, max_tokens 200,
ignore_eos, thinking off as the ruler sends it, in waves of c streams (as dev/n10/nofields.py). Sampling fields: only
those given; with none, the server's defaults apply (temperature 1.0, top-k off, the recipe's --top-p 1.0). No seed:
the server seeds from the prompt, so every stream of a cell draws the same reply. Aggregates use the ruler's own
aggregate_rate / decode_rate. Each stream's token_sha comes from the stream's tensorfold block (None on vLLM).

  python3 sampled_conc.py --label nofields --concurrency 1 2 4 --runs 9 [--temperature T] [--top-p P]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks")
import bench_decode as bd  # noqa: E402  (the frozen ruler: its prompt and rate math only)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--label", required=True)
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--runs", type=int, default=9)
ap.add_argument("--max-tokens", type=int, default=200)
ap.add_argument("--temperature", type=float)
ap.add_argument("--top-p", type=float)
a = ap.parse_args()
fields = {k: v for k, v in (("temperature", a.temperature), ("top_p", a.top_p)) if v is not None}


def one() -> dict:
    body = {"model": a.model, "messages": [{"role": "user", "content": bd.PHASES["prose"]}],
            "max_tokens": a.max_tokens, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}, **fields}
    req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, first, usage, runtime = time.perf_counter(), None, {}, {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            runtime = ev.get("tensorfold") or runtime
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    first = first or time.perf_counter()
    t1 = time.perf_counter()
    n = int(usage.get("completion_tokens") or 0)
    return {"completion_tokens": n, "ttft_s": first - t0, "decode_s": t1 - first,
            "decode_tok_s": bd.decode_rate(n, t1 - first), "token_sha": runtime.get("token_sha")}


print(f"label={a.label} url={a.url} max_tokens={a.max_tokens} runs={a.runs} concurrency={a.concurrency} "
      f"sampling={json.dumps(fields)}", flush=True)
summary = []
for c in a.concurrency:
    aggs, rows = [], []
    for i in range(a.runs):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as pool:
            got = list(pool.map(lambda _: one(), range(c)))
        wall = time.perf_counter() - t0
        agg = bd.aggregate_rate([r["completion_tokens"] for r in got], wall, statistics.median(r["ttft_s"] for r in got))
        aggs.append(agg)
        rows += got
        print(f"phase={a.label} c={c} run={i + 1} wall={wall:.2f}s agg={agg:.2f} tok/s per_stream=["
              + ",".join(f"{r['decode_tok_s']:.2f}" for r in got) + "] shas=" + ",".join(str(r["token_sha"]) for r in got),
              flush=True)
    summary.append({"phase": a.label, "sampling": fields, "concurrency": c, "n": len(rows),
                    "median_agg_tok_s": statistics.median(aggs),
                    "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in rows),
                    "ttft_s_p50": statistics.median(r["ttft_s"] for r in rows),
                    "token_shas": sorted({str(r["token_sha"]) for r in rows})})
print("SUMMARY", json.dumps(summary))
