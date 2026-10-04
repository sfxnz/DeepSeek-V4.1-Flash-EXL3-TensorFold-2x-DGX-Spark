"""s3: drafted vs "draft": false token_sha for a request that sends no sampling field (no temperature, top_p, top_k
or min_p), on s2's three prompts, so the server's defaults decide the draw (TOP_P=1.0 -> --top-p 1.0, the engine's
temperature 1.0, the family's top-k off). seed 1234, max_tokens 160, ignore_eos, as s2's `top_p1` section. Each
reply's token_sha is also compared with s2's `top_p1` rows (temperature 1.0 and top_p 1.0 sent by the request):
equal hashes mean the default request draws exactly as the explicit top_p 1.0 one. One JSON line per request.

  python3 pairs_default.py URL MODEL S2_PAIRS_JSONL
"""
import json
import sys
import urllib.request

URL, MODEL, S2 = sys.argv[1], sys.argv[2], sys.argv[3]
ASKS = ["Write a short Python function that computes the Fibonacci sequence and explain it.",
        "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
        "List five prime numbers greater than 100 and explain how you checked each one."]
s2 = {}
for line in open(S2):
    if not line.startswith("{"):        # s2's file ends with the driver's `rc=0` line
        continue
    r = json.loads(line)
    if r.get("case", "").startswith("top_p1 ") and r["case"].endswith(" drafted"):
        s2[int(r["case"].split()[1].removeprefix("prompt"))] = r["token_sha"]


def post(messages, **kw):
    body = {"model": MODEL, "messages": messages, **kw}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        out = json.loads(resp.read())
    s = out.get("tensorfold") or {}
    row = {k: kw.get(k) for k in ("seed", "draft") if k in kw}
    row.update(prompt_tokens=out["usage"]["prompt_tokens"], completion=out["usage"]["completion_tokens"],
               cached=out["usage"]["prompt_tokens_details"]["cached_tokens"], token_sha=s.get("token_sha"),
               rounds=s.get("rounds"), accepted=s.get("accepted"), drafted=s.get("drafted"),
               prefill_s=round(s.get("prefill_s", 0), 3), decode_s=round(s.get("decode_s", 0), 3),
               tok_s=round(out["usage"]["completion_tokens"] / s["decode_s"], 2) if s.get("decode_s") else None,
               stages_ms=s.get("stages_ms"))
    return row


fails = 0


def check(label, ok):
    global fails
    fails += not ok
    print(json.dumps({"check": label, "equal": ok}), flush=True)


for i, ask in enumerate(ASKS):
    m = [{"role": "user", "content": ask}]
    kw = dict(max_tokens=160, seed=1234, ignore_eos=True)          # no sampling field: the server's defaults
    a = post(m, **kw)
    print(json.dumps({"case": f"nofields prompt{i} drafted", **a}), flush=True)
    b = post(m, draft=False, **kw)
    print(json.dumps({"case": f"nofields prompt{i} draft:false", **b}), flush=True)
    check(f"nofields prompt{i} drafted == draft:false", a["token_sha"] == b["token_sha"])
    check(f"nofields prompt{i} == s2 top_p1 prompt{i} ({s2[i]})", a["token_sha"] == s2[i])
print(json.dumps({"section": "pairs_default", "failures": fails}))
sys.exit(1 if fails else 0)
