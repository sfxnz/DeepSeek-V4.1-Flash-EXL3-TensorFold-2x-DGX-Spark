"""Exact replies under concurrent requests, per request, so two boots can be compared (after
evidence/s4-device-nucleus/scripts/pairs_nucleus.py).

Four distinct prompts of different lengths, the last over 16k tokens, start --stagger-ms apart: each later one
arrives while the earlier ones decode. Two are greedy (temperature 0), two keyed (temperature 1.0 with a seed). No
request sends top_k or top_p, so the server's defaults apply (top-k off, the recipe's --top-p 1.0). ignore_eos keeps
every reply decoding to --max-tokens. Each request asks for its reply's token ids ("return_token_ids"): one JSON line
a request with token_ids and token_sha, then a summary line.

  python3 tools/pairs_concurrent.py --url URL --model MODEL --output A.jsonl     # e.g. on a --parallel 1 boot
  python3 tools/pairs_concurrent.py --url URL --model MODEL --output B.jsonl     # and on a --parallel 4 boot
  python3 tools/pairs_concurrent.py --compare A.jsonl B.jsonl

Exits nonzero if a request fails, the long prompt is not over 16384 tokens, a token_sha does not hash its token_ids,
or (--compare) a case is missing or its token_sha differs. Standard library only.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request

LONG_MIN_TOKENS = 16384


def records(n: int) -> str:
    """n numbered records from a fixed LCG: the same text on every machine and Python version."""

    cities = ["Lisbon", "Osaka", "Quito", "Tromso", "Accra", "Hobart", "Tbilisi", "Fresno"]
    x, lines = 20261008, []
    for i in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        lines.append(f"record {i:05d}: city {cities[x % 8]}, code {x % 1000000:06d}, count {x // 8 % 997}")
    return "\n".join(lines)


def log_lines(n: int) -> str:
    return "\n".join(f"Day {d}: the sensor read {12 + d * 7 % 19} degrees and the pump ran {30 + d * 11 % 47} minutes."
                     for d in range(1, n + 1))


# (case, prompt, temperature, seed), in start order; the long prompt arrives last, while the others decode.
CASES = [
    ("short greedy", "List five prime numbers greater than 100 and explain how you checked each one.", 0.0, None),
    ("medium keyed", "Here is a pump log.\n" + log_lines(40) + "\nDescribe the trend in the readings and what it "
     "suggests about the pump, in two short paragraphs.", 1.0, 1234),
    ("short keyed", "Write a short Python function that computes the Fibonacci sequence and explain it.", 1.0, 7),
    ("long greedy", "Here is a table of records.\n" + records(1400) + "\nWhat are the city and code of record "
     "00917? Then say how many of the first twenty records name that city.", 0.0, None),
]


def token_sha(ids: list[int]) -> str:
    """The server's token_sha (tensorfold/cuda/server.py token_sha): sha256 of the comma-joined ids, 12 hex."""

    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()[:12]


def post(url: str, model: str, prompt: str, temperature: float, seed: int | None, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "ignore_eos": True, "return_token_ids": True}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.loads(resp.read())


def run(args: argparse.Namespace) -> int:
    rows: list[dict] = [{} for _ in CASES]
    t0 = time.perf_counter()

    def one(i: int) -> None:
        case, prompt, temperature, seed = CASES[i]
        time.sleep(max(0.0, t0 + i * args.stagger_ms / 1000 - time.perf_counter()))
        sent = time.perf_counter()
        row = {"case": case, "temperature": temperature, "seed": seed, "sent_s": round(sent - t0, 3),
               "prompt_chars": len(prompt)}
        try:
            out = post(args.url, args.model, prompt, temperature, seed, args.max_tokens)
            s = out.get("tensorfold") or {}
            ids = s.get("token_ids")
            row.update(prompt_tokens=out["usage"]["prompt_tokens"], completion=out["usage"]["completion_tokens"],
                       cached=(out["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                       finish=out["choices"][0].get("finish_reason"), token_sha=s.get("token_sha"),
                       sha_ok=ids is not None and token_sha(ids) == s.get("token_sha"), rounds=s.get("rounds"),
                       accepted=s.get("accepted"), drafted=s.get("drafted"), wall_s=round(time.perf_counter() - sent, 3),
                       token_ids=ids)
        except Exception as exc:  # noqa: BLE001 - a failed request is reported, then fails the run
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows[i] = row

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(CASES))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    long_tokens = rows[-1].get("prompt_tokens") or 0
    failures = [r["case"] for r in rows if r.get("error") or not r.get("sha_ok")]
    summary = {"section": "pairs_concurrent", "requests": len(rows), "failures": failures,
               "long_prompt_tokens": long_tokens, "long_over_16k": long_tokens > LONG_MIN_TOKENS,
               "stagger_ms": args.stagger_ms, "max_tokens": args.max_tokens}
    lines = [json.dumps(r) for r in rows] + [json.dumps(summary)]
    print("\n".join(lines), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 1 if failures or not summary["long_over_16k"] else 0


def compare(a_path: str, b_path: str) -> int:
    def shas(path: str) -> dict[str, str | None]:
        return {r["case"]: r.get("token_sha") for r in map(json.loads, open(path)) if "case" in r}

    a, b = shas(a_path), shas(b_path)
    bad = 0
    for case in [c for c, *_ in CASES]:
        equal = a.get(case) is not None and a.get(case) == b.get(case)
        bad += not equal
        print(json.dumps({"case": case, "a": a.get(case), "b": b.get(case), "equal": equal}), flush=True)
    print(json.dumps({"section": "pairs_concurrent compare", "a": a_path, "b": b_path, "unequal": bad}), flush=True)
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--stagger-ms", type=float, default=1500.0, help="start request i this many ms after request i - 1")
    p.add_argument("--output")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two runs' token_sha per case")
    args = p.parse_args()
    return compare(*args.compare) if args.compare else run(args)


if __name__ == "__main__":
    sys.exit(main())
