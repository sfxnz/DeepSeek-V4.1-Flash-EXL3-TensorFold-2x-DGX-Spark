"""s4: s2's sampled_cell.py, the same requests, plus each reply's token_sha from the stream's tensorfold block.

s2's docstring: a sampled decode cell beside the frozen ruler (which is greedy only). bench_decode.py's prose prompt, streamed,
max_tokens 200, ignore_eos, thinking off as the ruler sends it, --runs runs at c=1, timed with bench_decode's own
decode_rate. Sampling fields: only those given (--temperature, --top-p, --top-k); the rest are the server's defaults
(temperature 1.0, top_p 0.95, top_k off for this family). No seed: the server seeds from the prompt, so every run
draws the same reply.

  python3 sampled_cell.py --label default --runs 9 [--temperature T] [--top-p P] [--top-k K]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import bench_decode as bd  # noqa: E402  (the frozen ruler; only its prompt and decode_rate are used)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--label", required=True)
ap.add_argument("--runs", type=int, default=9)
ap.add_argument("--max-tokens", type=int, default=200)
ap.add_argument("--temperature", type=float)
ap.add_argument("--top-p", type=float)
ap.add_argument("--top-k", type=int)
a = ap.parse_args()
fields = {k: v for k, v in (("temperature", a.temperature), ("top_p", a.top_p), ("top_k", a.top_k)) if v is not None}
print(f"label={a.label} url={a.url} max_tokens={a.max_tokens} runs={a.runs} sampling={json.dumps(fields)}", flush=True)
rows = []
for i in range(a.runs):
    body = {"model": a.model, "messages": [{"role": "user", "content": bd.PHASES["prose"]}], "max_tokens": a.max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}, **fields}
    req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage, text, runtime = {}, [], {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            runtime = ev.get("tensorfold") or runtime
            for ch in ev.get("choices") or []:
                d = (ch.get("delta") or {}).get("content") or ""
                if d:
                    first = first or time.perf_counter()
                    text.append(d)
    t1 = time.perf_counter()
    n = int(usage.get("completion_tokens") or 0)
    r = {"run": i + 1, "completion_tokens": n, "ttft_s": round(first - t0, 3),
         "decode_tok_s": round(bd.decode_rate(n, t1 - first), 2), "text_head": "".join(text)[:60],
         "token_sha": runtime.get("token_sha")}
    rows.append(r)
    print(json.dumps(r), flush=True)
s = {"label": a.label, "sampling": fields, "runs": a.runs,
     "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in rows),
     "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
     "same_text_every_run": len({r["text_head"] for r in rows}) == 1,
     "token_shas": sorted({r["token_sha"] for r in rows}, key=str)}
print("SAMPLED", json.dumps(s))
