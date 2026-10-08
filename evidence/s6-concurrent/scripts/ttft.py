"""N11 receipt 5 (and mp10): TTFT of a 16k prompt arriving while 3 lanes decode, the live lanes' rate and their
largest gap between tokens during the fill, every reply's token_sha; plus a background yield.

Requests (all greedy, ignore_eos, streamed, thinking off as the ruler sends it):
  LIVE    the frozen ruler's prose_long prompt, max_tokens 1000
  FILL(t) "Request n11-ttft-<t>." + the 16384-token prompt of receipts/final/prompts.json (its own first line
          dropped, as dev/n10/fillmix.py), max_tokens 64: a distinct prompt per tag, so nothing resumes
  BG      the ruler's prose prompt, max_tokens 400, "priority": "background"

  --mode alone   each request alone, one after another (the --parallel 1 reference): FILL(r0..r<reps-1>), FILL(yield),
                 LIVE, BG
  --mode cells   per rep r: 3 LIVE; once each has its first token, 1 s, then FILL(r<r>). Then the yield cell: 3 LIVE +
                 BG; once all 4 have a token, 1 s, then FILL(yield): the lanes are full, so the background stream
                 yields its lane and replays later

Live rate: a content delta is one decode round. Rounds of a LIVE stream inside [FILL sent, FILL first token],
(n - 1) / span, and tok/s = that times the stream's tokens per round. Max gap: the largest interval between consecutive
deltas of a LIVE stream that overlaps that window. tok_s of a whole stream: its tokens after the first delta over
first..last delta. One JSON line per
stream, a summary line per cell.

  python3 ttft.py --mode alone|cells --reps 3 --out FILE.jsonl [--base http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks")
import bench_decode as bd  # noqa: E402  (the frozen ruler: its prompts only)

PROMPTS = "/home/sfxnz/projects/data/tf-dsv41/receipts/final/prompts.json"
COLD = next(it for it in json.load(open(PROMPTS))["items"] if it["length"] == 16384)["messages"][0]["content"]
COLD = COLD.split("\n", 1)[1]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--mode", choices=["alone", "cells"], required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
OUT = open(a.out, "w")


def emit(row: dict) -> None:
    line = json.dumps(row)
    print(line, flush=True)
    OUT.write(line + "\n")
    OUT.flush()


def spec(kind: str, tag: str = "") -> dict:
    if kind == "LIVE":
        return {"content": bd.PHASES["prose_long"], "max_tokens": 1000}
    if kind == "BG":
        return {"content": bd.PHASES["prose"], "max_tokens": 400, "priority": "background"}
    return {"content": f"Request n11-ttft-{tag}.\n" + COLD, "max_tokens": 64}


def stream(kind: str, tag: str, out: dict, started: threading.Event | None = None) -> None:
    s = spec(kind, tag)
    body = {"model": a.model, "messages": [{"role": "user", "content": s["content"]}], "max_tokens": s["max_tokens"],
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True, "temperature": 0,
            "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}}
    if "priority" in s:
        body["priority"] = s["priority"]
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out.update(kind=kind, tag=tag, sent=time.perf_counter(), times=[], runtime={}, usage={})
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ev = json.loads(line[5:])
                out["runtime"] = ev.get("tensorfold") or out["runtime"]
                out["usage"] = ev.get("usage") or out["usage"]
                if any((c.get("delta") or {}).get("content") for c in ev.get("choices") or []):
                    out["times"].append(time.perf_counter())
                    if started is not None:
                        started.set()
    except Exception as exc:  # noqa: BLE001 - reported in the row
        out["error"] = f"{type(exc).__name__}: {exc}"
        if started is not None:
            started.set()
    out["done"] = time.perf_counter()


def window(times: list[float], lo: float, hi: float, per_delta: float) -> dict:
    """A delta is one decode round (several tokens with drafts): rounds/s inside the fill, and tok/s estimated at the
    stream's own tokens per round."""
    inside = [t for t in times if lo <= t <= hi]
    rate = (len(inside) - 1) / (inside[-1] - inside[0]) if len(inside) > 2 else None
    gaps = [b - x for x, b in zip(times, times[1:]) if b >= lo and x <= hi]
    return {"deltas_in_fill": len(inside), "rounds_per_s_in_fill": round(rate, 2) if rate else None,
            "tok_s_in_fill": round(rate * per_delta, 2) if rate else None,
            "max_gap_in_fill_s": round(max(gaps), 3) if gaps else None}


def row(o: dict, t0: float, fill: dict | None = None) -> dict:
    r = {"kind": o["kind"], "tag": o["tag"], "error": o.get("error"),
         "prompt_tokens": o["usage"].get("prompt_tokens"), "completion_tokens": o["usage"].get("completion_tokens"),
         "deltas": len(o["times"]), "token_sha": o["runtime"].get("token_sha"),
         "sent_s": round(o["sent"] - t0, 3),
         "ttft_s": round(o["times"][0] - o["sent"], 3) if o["times"] else None,
         "wall_s": round(o["done"] - o["sent"], 3),
         "rounds": o["runtime"].get("rounds"), "prefill_s": o["runtime"].get("prefill_s")}
    gaps = [b - x for x, b in zip(o["times"], o["times"][1:])]
    r["max_gap_s"] = round(max(gaps), 3) if gaps else None
    n = r["completion_tokens"] or 0
    if len(o["times"]) > 2:             # tokens after the first delta over first..last delta
        r["tok_s"] = round((n - n / len(o["times"])) / (o["times"][-1] - o["times"][0]), 2)
    if fill is not None and fill["times"] and o is not fill:
        r.update(window(o["times"], fill["sent"], fill["times"][0], n / max(1, len(o["times"]))))
    return r


def alone() -> None:
    for kind, tag in [*(("FILL", f"r{i}") for i in range(a.reps)), ("FILL", "yield"), ("LIVE", ""), ("BG", "")]:
        o = {}
        stream(kind, tag, o)
        emit({"mode": "alone", **row(o, o["sent"])})


def cell(tag: str, with_bg: bool) -> None:
    kinds = ["LIVE"] * 3 + (["BG"] if with_bg else [])
    outs = [{} for _ in kinds]
    ready = [threading.Event() for _ in kinds]
    t0 = time.perf_counter()
    threads = []
    for k, o, e in zip(kinds, outs, ready):
        threads.append(threading.Thread(target=stream, args=(k, "", o, e)))
        threads[-1].start()
        time.sleep(0.2)                     # BG arrives last: it is the newest stream
    for e in ready:
        e.wait(900)
    time.sleep(1.0)
    fill = {}
    stream("FILL", tag, fill)
    for t in threads:
        t.join()
    rows = [row(o, t0, fill) for o in outs] + [row(fill, t0)]
    for r in rows:
        emit({"mode": "cell", "cell": tag, **r})
    live = [r for r in rows if r["kind"] == "LIVE"]
    summary = {"mode": "cell", "cell": tag, "summary": True, "fill_ttft_s": rows[-1]["ttft_s"],
               "fill_prompt_tokens": rows[-1]["prompt_tokens"], "fill_sha": rows[-1]["token_sha"],
               "live_tok_s_in_fill": [r.get("tok_s_in_fill") for r in live],
               "live_max_gap_in_fill_s": [r.get("max_gap_in_fill_s") for r in live],
               "live_tok_s": [r.get("tok_s") for r in live], "live_shas": [r["token_sha"] for r in live],
               "errors": [r["error"] for r in rows if r["error"]]}
    if with_bg:
        bg = next(r for r in rows if r["kind"] == "BG")
        summary.update(bg_sha=bg["token_sha"], bg_max_gap_s=bg["max_gap_s"], bg_wall_s=bg["wall_s"],
                       bg_tok_s_in_fill=bg.get("tok_s_in_fill"))
    emit(summary)


if a.mode == "alone":
    alone()
else:
    for i in range(a.reps):
        cell(f"r{i}", False)
    cell("yield", True)
OUT.close()
