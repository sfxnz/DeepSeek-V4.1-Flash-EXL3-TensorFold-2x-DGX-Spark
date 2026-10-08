"""Dev only (outside the PR): Q2's HTTP receipts against a running server; one JSON line per request."""
import json
import sys
import urllib.request

URL, MODEL, PROMPTS = sys.argv[1], sys.argv[2], sys.argv[3]
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
    row = {k: kw.get(k) for k in ("temperature", "top_k", "seed", "draft") if k in kw}
    row.update(prompt_tokens=out["usage"]["prompt_tokens"], completion=out["usage"]["completion_tokens"],
               cached=out["usage"]["prompt_tokens_details"]["cached_tokens"], token_sha=s.get("token_sha"),
               rounds=s.get("rounds"), accepted=s.get("accepted"), drafted=s.get("drafted"),
               prefill_s=round(s.get("prefill_s", 0), 3), decode_s=round(s.get("decode_s", 0), 3),
               stages_ms=s.get("stages_ms"))
    return row


def show(tag, row):
    print(json.dumps({"case": tag, **row}), flush=True)
    return row


section = sys.argv[4]
fails = 0
if section == "pairs":
    for i, ask in enumerate(ASKS):
        m = [{"role": "user", "content": ask}]
        for t in (0.0, 1.0):
            kw = dict(max_tokens=160, temperature=t, top_k=20, seed=1234, ignore_eos=True)
            a = show(f"prompt{i} t={t} drafted", post(m, **kw))
            b = show(f"prompt{i} t={t} draft:false", post(m, draft=False, **kw))
            ok = a["token_sha"] == b["token_sha"]
            fails += not ok
            print(json.dumps({"check": f"prompt{i} t={t} top_k=20 drafted == draft:false", "equal": ok}), flush=True)
    m = [{"role": "user", "content": ASKS[0]}]
    kw = dict(max_tokens=160, temperature=1.0, seed=99, ignore_eos=True)       # top_k omitted: the family default
    a = show("top_k omitted drafted", post(m, **kw))
    b = show("top_k omitted draft:false", post(m, draft=False, **kw))
    c = show("top_k omitted resend", post(m, **kw))
    ok = a["token_sha"] == b["token_sha"] == c["token_sha"] and c["cached"] > 0
    fails += not ok
    print(json.dumps({"check": "top_k omitted: drafted == draft:false == resend, resend cached > 0", "equal": ok}))
elif section.startswith("long"):
    rep = int(section[4:] or 0)                     # long, long1, long2: prefill_cold's 64k items 0, 1, 2
    item = next(it for it in json.load(open(PROMPTS))["items"] if it["length"] == 65536 and it["rep"] == rep)
    for tag in ("cold 64k", "resume 1", "resume 2", "resume 3"):
        show(tag, post(item["messages"], max_tokens=2, temperature=0))
print(json.dumps({"section": section, "failures": fails}))
sys.exit(1 if fails else 0)
