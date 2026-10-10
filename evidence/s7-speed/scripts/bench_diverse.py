#!/usr/bin/env python3
"""Dev only (M2): distinct-prompt decode cell against a live /v1/chat/completions.

Each stream of a wave gets a different prompt: stream s of wave w takes pool[(w * c + s) % len(pool)], so no two
streams of a wave share a prompt (c <= pool size). Two fixed pools: public-domain prose openings to continue, and
small code tasks. Request body as the frozen ruler (bench_decode.stream_one): greedy, ignore_eos, thinking off.
Per-stream decode tok/s and the wave aggregate use the ruler's own functions (decode_rate, aggregate_rate).
The engine's final-event stats ("tensorfold": rounds, accepted, token_sha) are kept per stream when present.
Exactness: greedy replies of one prompt must hash the same at every c; SHAS lists each prompt's distinct shas.
usage: bench_diverse.py --url URL [--phase prose|code|both] [--concurrency 1 2 4] [--runs 9] [--max-tokens 200]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks")
from bench_decode import aggregate_rate, decode_rate, inter_chunk_ms, pct_fields  # noqa: E402

CONTINUE = "Continue this passage in the same style for about two hundred words. No headings.\n\n"
PROSE = [  # opening lines of public-domain novels
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want "
    "of a wife.",
    "Call me Ishmael. Some years ago - never mind how long precisely - having little or no money in my purse, and "
    "nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of "
    "the world.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
    "it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of "
    "Darkness.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once "
    "or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it.",
    "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have "
    "regarded with such evil forebodings.",
    "My father's family name being Pirrip, and my Christian name Philip, my infant tongue could make of both names "
    "nothing longer or more explicit than Pip. So, I called myself Pip, and came to be called Pip.",
    "There was no possibility of taking a walk that day.",
    "The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us.",
]
CODE = [
    "Write a Python class LRUCache with get and put methods in O(1) time. Output only the code.",
    "Write a C function that reverses a singly linked list in place and returns the new head. Output only the code.",
    "Write a Python function that parses a CSV file and returns the sum of a named numeric column, skipping rows "
    "where the value is missing. Output only the code.",
    "Write Dijkstra's shortest path algorithm in Python using heapq, on an adjacency-list graph. Output only the code.",
    "Write a Rust function that counts word frequencies in a string and returns them sorted by count, highest first. "
    "Output only the code.",
    "Write a JavaScript function that debounces another function by a given number of milliseconds. Output only "
    "the code.",
    "Write a SQL query that returns the top three customers by total order value per country, from tables "
    "customers(id, name, country) and orders(id, customer_id, amount). Output only the code.",
    "Write a Go HTTP handler that accepts a JSON body with a list of integers and responds with their mean and "
    "median. Output only the code.",
]
POOLS = {"prose": [CONTINUE + p for p in PROSE], "code": CODE}


def stream_one(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed greedy reply with the ruler's request body; timings, usage, finish and the engine stats."""
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0, first, times, text, usage, stats, finish = time.perf_counter(), None, [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            usage = ev.get("usage") or usage
            stats = ev.get("tensorfold", stats)
            choices = ev.get("choices") or []
            if not choices:
                continue
            finish = choices[0].get("finish_reason") or finish
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                now = time.perf_counter()
                first = first or now
                times.append(now)
                text.append(delta)
    t1 = time.perf_counter()
    if first is None:
        raise RuntimeError("no streamed content tokens")
    completion = int(usage.get("completion_tokens") or 0)
    out = {"ttft_s": first - t0, "decode_s": t1 - first, "completion_tokens": completion,
           "prompt_tokens": int(usage.get("prompt_tokens") or 0), "decode_tok_s": decode_rate(completion, t1 - first),
           "inter_chunk_ms": inter_chunk_ms(times), "finish_reason": finish,
           "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:12]}
    if stats:
        out["token_sha"] = stats.get("token_sha")
        out["rounds"] = stats.get("rounds")
        out["accepted"] = stats.get("accepted")
        out["drafted"] = stats.get("drafted")
    return out


def wave(url: str, model: str, prompts: list[tuple[int, str]], max_tokens: int) -> tuple[list[dict], float, float]:
    """All prompts at once, one stream each; returns rows, wall and the ruler's aggregate tok/s."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futs = [pool.submit(stream_one, url, model, p, max_tokens) for _, p in prompts]
        rows = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    for (idx, _), r in zip(prompts, rows):
        r["prompt"] = idx
    if any(r["completion_tokens"] == 0 for r in rows):
        raise RuntimeError("a stream returned completion_tokens==0")
    agg = aggregate_rate([r["completion_tokens"] for r in rows], wall, statistics.median(r["ttft_s"] for r in rows))
    return rows, wall, agg


def ms_per_round(r: dict) -> float | None:
    """Engine rounds' mean wall in ms over the post-TTFT decode span."""
    return 1000.0 * r["decode_s"] / r["rounds"] if r.get("rounds") else None


def cell_summary(phase: str, c: int, rows: list[dict], aggs: list[float]) -> dict:
    """One (phase, c) SUMMARY row, field names as bench_decode's where they mean the same."""
    mspr = [x for x in map(ms_per_round, rows) if x is not None]
    tpr = [r["completion_tokens"] / r["rounds"] for r in rows if r.get("rounds")]
    return {
        "phase": phase, "concurrency": c, "n": len(rows),
        "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in rows),
        "median_agg_tok_s": statistics.median(aggs),
        "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
        "median_completion_tokens": statistics.median(r["completion_tokens"] for r in rows),
        "median_ms_per_round": statistics.median(mspr) if mspr else None,
        "median_tokens_per_round": statistics.median(tpr) if tpr else None,
        "prompts": sorted({r["prompt"] for r in rows}),
        **pct_fields([g for r in rows for g in r["inter_chunk_ms"]], "inter_chunk_ms"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=9)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--phase", choices=[*POOLS, "both"], default="both")
    args = p.parse_args()
    phases = list(POOLS) if args.phase == "both" else [args.phase]
    print(f"url={args.url} model={args.model} max_tokens={args.max_tokens} runs={args.runs} "
          f"concurrency={args.concurrency} phases={phases}", flush=True)
    summary, shas = [], {}
    for phase in phases:
        pool = POOLS[phase]
        for c in args.concurrency:
            if c > len(pool):
                raise SystemExit(f"c={c} exceeds the {phase} pool of {len(pool)}")
            rows, aggs = [], []
            for w in range(args.runs):
                picks = [(w * c + s) % len(pool) for s in range(c)]
                out, wall, agg = wave(args.url, args.model, [(i, pool[i]) for i in picks], args.max_tokens)
                rows.extend(out)
                aggs.append(agg)
                for r in out:
                    shas.setdefault(f"{phase}/{r['prompt']}", {}).setdefault(r.get("token_sha") or r["text_sha"], []).append(c)
                dec = ",".join(f"{r['decode_tok_s']:.2f}" for r in out)
                ttft = ",".join(f"{r['ttft_s']:.3f}" for r in out)
                rnd = ",".join(f"{x:.1f}" for x in map(ms_per_round, out) if x is not None)
                print(f"phase={phase} c={c} run={w + 1} prompts={picks} wall={wall:.2f}s agg={agg:.2f} tok/s "
                      f"per_stream=[{dec}] ttft=[{ttft}] ms_round=[{rnd}] "
                      f"sha=[{','.join(str(r.get('token_sha') or r['text_sha']) for r in out)}]", flush=True)
            summary.append(cell_summary(phase, c, rows, aggs))
    split = {k: v for k, v in shas.items() if len(v) > 1}
    print("SHAS", json.dumps({k: {s: sorted(set(cs)) for s, cs in v.items()} for k, v in sorted(shas.items())}))
    print(f"SHA_CHECK prompts={len(shas)} differing={len(split)} {'PASS' if not split else 'FAIL ' + ','.join(split)}")
    print("SUMMARY", json.dumps(summary, indent=2))
    return 0 if not split else 1


if __name__ == "__main__":
    sys.exit(main())
