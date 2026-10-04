"""s2: drafted vs "draft": false token_sha on three prompts, top_k 20 and top_k omitted (one JSON line per request).

Section `q2` repeats data/tf-dsv41/dev/q2/receipts.py `pairs` request for request (the s0 receipts' pairs.jsonl,
whose top_k-omitted drafted request decoded at 24.5 tok/s at 0d91389). Sections `default` (top_k omitted, the
server's top_p 0.95) and `top_p1` (top_k omitted, top_p 1.0: the rows the device keyed draw takes) add the other two
prompts. tok_s = completion / decode_s from the response's tensorfold block.

  python3 pairs.py URL MODEL
"""
import json
import sys
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
ASKS = ["Write a short Python function that computes the Fibonacci sequence and explain it.",
        "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
        "List five prime numbers greater than 100 and explain how you checked each one."]


def post(messages, **kw):
    body = {"model": MODEL, "messages": messages, **kw}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        out = json.loads(resp.read())
    s = out.get("tensorfold") or {}
    row = {k: kw.get(k) for k in ("temperature", "top_k", "top_p", "seed", "draft") if k in kw}
    row.update(prompt_tokens=out["usage"]["prompt_tokens"], completion=out["usage"]["completion_tokens"],
               cached=out["usage"]["prompt_tokens_details"]["cached_tokens"], token_sha=s.get("token_sha"),
               rounds=s.get("rounds"), accepted=s.get("accepted"), drafted=s.get("drafted"),
               prefill_s=round(s.get("prefill_s", 0), 3), decode_s=round(s.get("decode_s", 0), 3),
               tok_s=round(out["usage"]["completion_tokens"] / s["decode_s"], 2) if s.get("decode_s") else None,
               stages_ms=s.get("stages_ms"))
    return row


def show(tag, row):
    print(json.dumps({"case": tag, **row}), flush=True)
    return row


fails = 0


def check(label, ok):
    global fails
    fails += not ok
    print(json.dumps({"check": label, "equal": ok}), flush=True)


# q2: as the s0 receipts
for i, ask in enumerate(ASKS):
    m = [{"role": "user", "content": ask}]
    for t in (0.0, 1.0):
        kw = dict(max_tokens=160, temperature=t, top_k=20, seed=1234, ignore_eos=True)
        a = show(f"q2 prompt{i} t={t} top_k=20 drafted", post(m, **kw))
        b = show(f"q2 prompt{i} t={t} top_k=20 draft:false", post(m, draft=False, **kw))
        check(f"prompt{i} t={t} top_k=20 drafted == draft:false", a["token_sha"] == b["token_sha"])
m = [{"role": "user", "content": ASKS[0]}]
kw = dict(max_tokens=160, temperature=1.0, seed=99, ignore_eos=True)       # top_k omitted: the family default
a = show("q2 top_k omitted drafted", post(m, **kw))
b = show("q2 top_k omitted draft:false", post(m, draft=False, **kw))
c = show("q2 top_k omitted resend", post(m, **kw))
check("q2 top_k omitted: drafted == draft:false == resend, resend cached > 0",
      a["token_sha"] == b["token_sha"] == c["token_sha"] and c["cached"] > 0)
# default (top_p 0.95) and top_p 1.0, top_k omitted, three prompts
for sec, extra in (("default", {}), ("top_p1", {"top_p": 1.0})):
    for i, ask in enumerate(ASKS):
        m = [{"role": "user", "content": ask}]
        kw = dict(max_tokens=160, temperature=1.0, seed=1234, ignore_eos=True, **extra)
        a = show(f"{sec} prompt{i} t=1.0 top_k omitted drafted", post(m, **kw))
        b = show(f"{sec} prompt{i} t=1.0 top_k omitted draft:false", post(m, draft=False, **kw))
        check(f"{sec} prompt{i} t=1.0 top_k omitted drafted == draft:false", a["token_sha"] == b["token_sha"])
print(json.dumps({"section": "pairs", "failures": fails}))
sys.exit(1 if fails else 0)
