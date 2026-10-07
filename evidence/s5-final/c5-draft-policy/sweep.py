"""Dev only (outside the PR, C5): decode cells against the two-rank server, the DSpark policy set through the server
wrapper's policy file before each policy's block; one JSON line per request.

usage: sweep.py URL MODEL POLICIES PASSES [CASES]
  POLICIES: comma list of d (fixed drafts) or cP@D (confidence P, at most D drafts), e.g. 1,2,3,4,5,c0.5@5
  PASSES: how many passes; even passes run the policies in reverse order and use the next seed
  CASES: comma list of case names (default: all)
"""
import json
import sys
import time
import urllib.request

URL, MODEL, POLICIES, PASSES = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
CTL = "/home/sfxnz/projects/data/tf-dsv41/receipts/c5/ctl/policy.json"
LAIL = ("Continue this essay in the same voice. Do not stop.\n\n"
        "Decode throughput and time-to-first-token feel different when a coding "
        "agent shares a long system prompt across tabs on a DGX Spark with unified "
        "memory. The KV cache is the product, not a leftover after util. ")
PROSE = ("Write a short paragraph about why sparse attention helps long-context "
         "language models. Keep it around eighty words. No bullet points.")
COUNT = "Count from 1 to 200. Output only the numbers, separated by commas, with no other text."
CODE = ("Write a Python module that implements an LRU cache with a fixed capacity, using a dictionary and a doubly "
        "linked list. Include type hints, docstrings and a short usage example at the end.")
SV = ("Fortsätt den här essän med samma röst. Sluta inte.\n\n"
      "När vintern kommer till Norrland blir dagarna korta och ljuset lågt över skogen. Människorna i byarna har lärt "
      "sig att planera arbetet efter mörkret, och de små stationerna längs järnvägen blir mötesplatser där man byter "
      "nyheter, väder och historier om älgar som korsat spåren. ")
DE = ("Setze diesen Essay in derselben Stimme fort. Höre nicht auf.\n\n"
      "Wenn im Herbst der Nebel über die Felder am Rhein zieht, beginnt in den Dörfern die Zeit der Weinlese. Die "
      "Familien arbeiten vom frühen Morgen bis in den Abend, und die alten Keller unter den Häusern füllen sich mit dem "
      "Geruch von frischem Most und feuchtem Holz. ")
ZH = ("请用同样的语气继续写这篇文章，不要停下来。\n\n"
      "每到春天，江南的小镇便被细雨笼罩。石板路上泛着微光，河边的柳树抽出新芽，茶农们清晨就上山采茶。老人们坐在桥头，"
      "一边喝茶一边讲述这座小镇几百年来的变迁，孩子们则在巷子里追逐嬉戏。")
# name: (prompt, temperature, max_tokens); sampled cases take the pass's seed
CASES = {"prose": (PROSE, 0.0, 200), "structured": (COUNT, 0.0, 200), "lail": (LAIL, 0.2, 512), "code": (CODE, 0.0, 400),
         "sv": (SV, 0.2, 512), "de": (DE, 0.2, 512), "zh": (ZH, 0.2, 512),
         "sv_greedy": (SV, 0.0, 300), "de_greedy": (DE, 0.0, 300), "zh_greedy": (ZH, 0.0, 300)}
names = sys.argv[5].split(",") if len(sys.argv) > 5 else list(CASES)


def policy(text):
    if text.startswith("c"):
        p, d = text[1:].split("@")
        return {"drafts": int(d), "confidence": float(p)}
    return {"drafts": int(text), "confidence": None}


def post(prompt, temperature, max_tokens, seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.loads(resp.read())
    wall = time.perf_counter() - t
    s = out.get("tensorfold") or {}
    n = out["usage"]["completion_tokens"]
    return {"completion": n, "prompt_tokens": out["usage"]["prompt_tokens"], "wall_s": round(wall, 4),
            "decode_s": s.get("decode_s"), "tok_s": round((n - 1) / s["decode_s"], 3) if s.get("decode_s") else None,
            "rounds": s.get("rounds"), "drafted": s.get("drafted"), "accepted": s.get("accepted"),
            "tokens_per_round": s.get("tokens_per_round"), "token_sha": s.get("token_sha"),
            "stages_ms": s.get("stages_ms"), "policy_reply": s.get("policy")}


i = 0
for pas in range(PASSES):
    order = POLICIES if pas % 2 == 0 else POLICIES[::-1]
    for pol in order:
        with open(CTL, "w") as f:
            json.dump(policy(pol), f)
        for name in names:
            prompt, temp, mx = CASES[name]
            seed = 1 + pas if temp > 0 else None
            row = post(prompt, temp, mx, seed)
            print(json.dumps({"i": i, "pass": pas, "policy": pol, "case": name, "temperature": temp, "seed": seed,
                              **row}), flush=True)
            i += 1
print(json.dumps({"done": i}), flush=True)
