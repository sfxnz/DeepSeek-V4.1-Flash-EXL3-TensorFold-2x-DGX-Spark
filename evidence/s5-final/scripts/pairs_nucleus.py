"""s4: drafted vs "draft": false token_sha at temperature 1.0 with top_k omitted (the family's top-k off), at top_p
0.95 and top_p 1.0, on s2's three prompts: seed 1234, max_tokens 160, ignore_eos, as s2's `default` and `top_p1`
sections (s2's `default` sent no top_p and got the server's 0.95 there; here top_p is sent). Each drafted hash is
also compared with s2's reply to the same request at ec28f35 (top_p 0.95: the host path then; top_p 1.0: the device
draw then). One JSON line per request.

  python3 pairs_nucleus.py URL MODEL S2_PAIRS_JSONL
"""
import json
import sys
import urllib.request

URL, MODEL, S2 = sys.argv[1], sys.argv[2], sys.argv[3]
ASKS = ["Write a short Python function that computes the Fibonacci sequence and explain it.",
        "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
        "List five prime numbers greater than 100 and explain how you checked each one."]
SECTIONS = {0.95: "default", 1.0: "top_p1"}      # s2's section for the same request
s2 = {}
for line in open(S2):
    if not line.startswith("{"):        # s2's file ends with the driver's `rc=0` line
        continue
    r = json.loads(line)
    case = r.get("case", "")
    if case.split(" ")[0] in SECTIONS.values() and case.endswith(" drafted"):
        s2[(case.split()[0], int(case.split()[1].removeprefix("prompt")))] = r["token_sha"]


def post(messages, **kw):
    body = {"model": MODEL, "messages": messages, **kw}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        out = json.loads(resp.read())
    s = out.get("tensorfold") or {}
    row = {k: kw.get(k) for k in ("temperature", "top_p", "seed", "draft") if k in kw}
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


for top_p, sec in SECTIONS.items():
    for i, ask in enumerate(ASKS):
        m = [{"role": "user", "content": ask}]
        kw = dict(max_tokens=160, temperature=1.0, top_p=top_p, seed=1234, ignore_eos=True)
        a = post(m, **kw)
        print(json.dumps({"case": f"top_p {top_p} prompt{i} drafted", **a}), flush=True)
        b = post(m, draft=False, **kw)
        print(json.dumps({"case": f"top_p {top_p} prompt{i} draft:false", **b}), flush=True)
        check(f"top_p {top_p} prompt{i} drafted == draft:false", a["token_sha"] == b["token_sha"])
        check(f"top_p {top_p} prompt{i} == s2 {sec} prompt{i} ({s2[(sec, i)]})", a["token_sha"] == s2[(sec, i)])
print(json.dumps({"section": "pairs_nucleus", "failures": fails}))
sys.exit(1 if fails else 0)
