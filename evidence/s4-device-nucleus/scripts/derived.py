"""s4: the README numbers that no evidence file holds literally, each printed with its inputs. Stdlib only.

  python3 scripts/derived.py > derived.txt        (run from evidence/s4-device-nucleus)
"""
import json
import re
from datetime import datetime

S2, S3 = "../s2-run-sh-ec28f35", "../s3-default-1m"
LABELS = ("nofields", "top_p1", "top_k20", "top_p095", "top_p09")
S3_LABELS = ("nofields", "top_p1", "top_k20", "top_p095")      # s3's cell order in boot.sh


def summary(path):
    text = open(path).read()
    return json.loads(text.split("SAMPLED ", 1)[1].splitlines()[0])


print("# Derived README numbers, s4-device-nucleus. Inputs are files in this directory, ../s3-default-1m and "
      "../s2-run-sh-ec28f35.")

print("\n## Engine and server sampling default: gate/serve-argv.txt, rank 0's `serving` line (gate/startup.txt)")
print(open("gate/serve-argv.txt").read().splitlines()[0].split(" ")[0])
line = next(l for l in open("gate/startup.txt") if "] serving " in l)
print(f"sampling: {re.search(r'sampling: ([^;]*);', line)[1]!r}; context: {re.search(r'context: (\d+)', line)[1]}")

print("\n## Ruler, one boot: gate/bench-*.out SUMMARY median_decode_tok_s, against s3's two-boot rows (decode.txt)")
s3 = {r["phase"]: r for r in json.loads(open(f"{S3}/decode.txt").read().split("SUMMARY ", 1)[1])}
for f in ("gate/bench-frozen.out", "gate/bench-prose-long.out"):
    for r in json.loads(open(f).read().split("SUMMARY ", 1)[1]):
        p = r["phase"]
        print(f"{p}: {r['median_decode_tok_s']:.2f} (ttft p50 {r['ttft_s_p50']:.3f}, n {r['n']}) vs s3 "
              f"{s3[p]['median_decode_tok_s']:.2f} (bootC {s3[p]['per_boot']['bootC']['median_decode_tok_s']:.2f}, "
              f"bootD {s3[p]['per_boot']['bootD']['median_decode_tok_s']:.2f}) -> "
              f"{r['median_decode_tok_s'] / s3[p]['median_decode_tok_s']:.3f}")

print("\n## Sampled cells: sampled-<label>.txt SAMPLED (one boot), against s3's boots C / D")
s3_sha = {}
for b in ("bootC", "bootD"):
    done = [l for l in open(f"{S3}/{b}/rank0-full.log") if "] done req-" in l and " prompt=29 " in l
            and " tokens=200 " in l]
    last = done[-9 * len(S3_LABELS):]             # the four cells' 36 requests, after the ruler's
    for k, lab in enumerate(S3_LABELS):
        s3_sha[(b, lab)] = sorted({re.search(r" sha=([0-9a-f]+)", l)[1] for l in last[9 * k:9 * k + 9]})
cells = {}
for lab in LABELS:
    s = summary(f"sampled-{lab}.txt")
    cells[lab] = s["median_decode_tok_s"]
    out = (f"{lab}: request sends {json.dumps(s['sampling'])}; median {s['median_decode_tok_s']} tok/s, ttft "
           f"{s['median_ttft_s']}, same text {s['same_text_every_run']}, token_sha {s['token_shas']}")
    if lab in S3_LABELS:
        old = [summary(f"{S3}/{b}/sampled-{lab}.txt")["median_decode_tok_s"] for b in ("bootC", "bootD")]
        heads = {json.loads(l)["text_head"] for b in ("bootC", "bootD") for l in open(f"{S3}/{b}/sampled-{lab}.txt")
                 if l.startswith('{"run"')}
        mine = {json.loads(l)["text_head"] for l in open(f"sampled-{lab}.txt") if l.startswith('{"run"')}
        out += (f"; s3 {old[0]} / {old[1]} tok/s, s3 token_sha (rank0-full.log) bootC {s3_sha[('bootC', lab)]} bootD "
                f"{s3_sha[('bootD', lab)]}; token_sha equal to s3: "
                f"{s['token_shas'] == s3_sha[('bootC', lab)] == s3_sha[('bootD', lab)]}; text head equal to s3: "
                f"{mine == heads}")
    print(out)
print(f"top_p095 / nofields = {cells['top_p095'] / cells['nofields']:.3f}; "
      f"top_p09 / nofields = {cells['top_p09'] / cells['nofields']:.3f}; "
      f"top_p095 / s3 top_p095 (median of boots) = "
      f"{cells['top_p095'] / ((summary(f'{S3}/bootC/sampled-top_p095.txt')['median_decode_tok_s'] + summary(f'{S3}/bootD/sampled-top_p095.txt')['median_decode_tok_s']) / 2):.2f}")

print("\n## Rounds of each sampled reply: rank0-full.log, the 45 `done` lines of the cells (after the ruler's)")
done = [l for l in open("rank0-full.log") if "] done req-" in l and " prompt=29 " in l and " tokens=200 " in l][-45:]
reply = {lab: sorted({(re.search(r" sha=(\S+)", l)[1], int(re.search(r" rounds=(\d+)", l)[1]),
                        re.search(r" accepted=(\S+)", l)[1]) for l in done[9 * k:9 * k + 9]})
         for k, lab in enumerate(LABELS)}
for lab in LABELS:
    print(f"{lab}: (sha, rounds, accepted) {reply[lab]}; median tok/s x rounds / top_p1's rounds = "
          f"{cells[lab] * reply[lab][0][1] / reply['top_p1'][0][1]:.2f}")

print("\n## Exactness: pairs.jsonl checks")
rows = [json.loads(l) for l in open("pairs.jsonl") if l.startswith("{")]
checks = [r for r in rows if "check" in r]
pair = [r for r in checks if "draft:false" in r["check"]]
old = [r for r in checks if " s2 " in r["check"]]
print(f"drafted == draft:false: {sum(r['equal'] for r in pair)} of {len(pair)}; equal to s2's reply at ec28f35: "
      f"{sum(r['equal'] for r in old)} of {len(old)}")
for r in rows:
    if "case" in r:
        print(f"{r['case']}: sha {r['token_sha']} {r['tok_s']} tok/s, accepted {r['accepted']}/{r['drafted']}")

print("\n## Before (s2 bootA/pairs.jsonl at ec28f35) and after (pairs.jsonl): drafted, the same request; tok/s, rounds, "
      "the sampling stage's ms a round")
s2rows = {r["case"].split(" t=")[0] + " " + r["case"].split()[-1]: r for r in (json.loads(l) for l in open(f"{S2}/bootA/pairs.jsonl")
                                                  if l.startswith('{"case'))}
for r in rows:
    if "case" in r and r["case"].endswith(" drafted"):
        top_p, i = r["case"].split()[1], r["case"].split()[2]
        o = s2rows[f"{'default' if top_p == '0.95' else 'top_p1'} {i} drafted"]
        print(f"top_p {top_p} {i}: before {o['tok_s']} tok/s, {o['rounds']} rounds, sampling {o['stages_ms']['sampling']} "
              f"ms; after {r['tok_s']} tok/s, {r['rounds']} rounds, sampling {r['stages_ms']['sampling']} ms; "
              f"after / before {r['tok_s'] / o['tok_s']:.2f}")

print("\n## Duration")
text = open("run.txt").read()
a, e = (datetime.strptime(re.search(rf"^{k} (\S+)$", text, re.M)[1], "%Y-%m-%dT%H:%M:%SZ") for k in ("start", "end"))
print(f"./run.sh to ready: run.txt start {a:%H:%M:%S} end {e:%H:%M:%S} = {(e - a).total_seconds():.0f} s")
img = open("image-only.txt").read().split()
a, e = (datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in (img[0], img[-1]))
print(f"IMAGE_ONLY=1 ./run.sh: image-only.txt {a:%H:%M:%S} to {e:%H:%M:%S} = {(e - a).total_seconds():.0f} s")

print("\n## Bench requests with thinking off: gate/docker-head.log")
done = [l for l in open("gate/docker-head.log") if "] done req-" in l]
print(f"done lines={len(done)} thinking=False={sum('thinking=False' in l for l in done)}")
