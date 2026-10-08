"""N11 receipt 6: the 1,039,528-token needle (evidence/s5-final/scripts/long_needle.py 1040000 0.5, unchanged) while
3 other lanes decode. Three threads each send the ruler's prose_long prompt (greedy, ignore_eos, max_tokens 1500,
streamed, thinking off as the ruler sends it) back to back from before the needle is sent until it answers; the
needle starts once all three have a token. One JSON line per live request (its token_sha, tok/s, whether it overlapped
the needle), the needle's own line(s) as long_needle.py prints them, and a summary.

  python3 needle_lanes.py OUT.jsonl [--base http://127.0.0.1:8000]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks")
import bench_decode as bd  # noqa: E402

NEEDLE = "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks/evidence/s5-final/scripts/long_needle.py"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
a = ap.parse_args()
OUT = open(a.out, "w")
lock = threading.Lock()


def emit(row: dict) -> None:
    with lock:
        line = json.dumps(row)
        print(line, flush=True)
        OUT.write(line + "\n")
        OUT.flush()


stop = threading.Event()
needle_window = {}


def live(i: int, first: threading.Event) -> None:
    n = 0
    while not stop.is_set():
        body = {"model": a.model, "messages": [{"role": "user", "content": bd.PHASES["prose_long"]}],
                "max_tokens": 1500, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
                "temperature": 0, "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}}
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        sent, times, runtime, usage, err = time.time(), [], {}, {}, None
        try:
            with urllib.request.urlopen(req, timeout=7200) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    ev = json.loads(line[5:])
                    runtime = ev.get("tensorfold") or runtime
                    usage = ev.get("usage") or usage
                    if any((c.get("delta") or {}).get("content") for c in ev.get("choices") or []):
                        times.append(time.time())
                        first.set()
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            first.set()
        gaps = [b - x for x, b in zip(times, times[1:])]
        lo, hi = needle_window.get("start"), needle_window.get("end")
        emit({"lane": i, "request": n, "error": err, "completion_tokens": usage.get("completion_tokens"),
              "token_sha": runtime.get("token_sha"), "sent": round(sent, 1), "done": round(time.time(), 1),
              "tok_s": round((usage.get("completion_tokens", 0) * (1 - 1 / len(times))) / (times[-1] - times[0]), 2)
              if len(times) > 2 else None,
              "max_gap_s": round(max(gaps), 3) if gaps else None,
              "overlaps_needle": lo is not None and sent < (hi or 1e18) and time.time() > lo})
        n += 1
        if err:
            time.sleep(5)


firsts = [threading.Event() for _ in range(3)]
threads = [threading.Thread(target=live, args=(i, e)) for i, e in enumerate(firsts)]
for t in threads:
    t.start()
    time.sleep(0.2)
for e in firsts:
    e.wait(900)
time.sleep(1.0)
needle_window["start"] = time.time()
p = subprocess.run([sys.executable, NEEDLE, "1040000", "0.5", "--url", a.base], capture_output=True, text=True)
needle_window["end"] = time.time()
stop.set()
for line in p.stdout.splitlines():
    try:
        emit({"needle": json.loads(line)})
    except json.JSONDecodeError:
        emit({"needle_text": line})
for t in threads:
    t.join()
emit({"summary": True, "needle_rc": p.returncode, "needle_stderr_tail": p.stderr[-800:],
      "needle_s": round(needle_window["end"] - needle_window["start"], 1)})
OUT.close()
sys.exit(p.returncode)
