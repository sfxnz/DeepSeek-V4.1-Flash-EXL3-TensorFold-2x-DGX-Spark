"""s3: the README numbers that no evidence file holds literally (medians of per-boot medians, rounding, ratios,
counts, durations), each printed with its inputs. Stdlib only, deterministic.

  python3 scripts/derived.py > derived.txt        (run from evidence/s3-default-1m)
"""
import json
import re
import statistics as st
from datetime import datetime

BOOTS = ("bootC", "bootD")
S2 = "../s2-run-sh-ec28f35"


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def lower_pct(ours, theirs):
    return 100 * (1 - ours / theirs)


print("# Derived README numbers, s3-default-1m. Inputs are files in this directory or ../s2-run-sh-ec28f35.")

# --- Server sampling default: rank 0's serving line ------------------------------------------------------------
print("\n## Server sampling default: rank 0's `serving` line (gate/startup.txt)")
for b in BOOTS:
    line = next(l for l in open(f"{b}/gate/startup.txt") if "] serving " in l)
    print(f"{b}: sampling: {re.search(r'sampling: ([^;]*);', line)[1]!r}; "
          f"context: {re.search(r'context: (\d+)', line)[1]}")

# --- Sampled cells (README Sampled decode): sampled_cell.py SAMPLED lines -----------------------------------------
print("\n## Sampled cells: <boot>/sampled-<label>.txt SAMPLED median_decode_tok_s; row = median of the per-boot medians")
LABELS = ("nofields", "top_p1", "top_k20", "top_p095")
cells = {}
for label in LABELS:
    per = {}
    for b in BOOTS:
        text = open(f"{b}/sampled-{label}.txt").read()
        s = json.loads(text.split("SAMPLED ", 1)[1].splitlines()[0])
        per[b] = s
    meds = [per[b]["median_decode_tok_s"] for b in BOOTS]
    cells[label] = st.median(meds)
    print(f"{label}: request sends {json.dumps(per[BOOTS[0]]['sampling'])}; "
          + ", ".join(f"{b} {per[b]['median_decode_tok_s']} (ttft {per[b]['median_ttft_s']}, same text "
                      f"{per[b]['same_text_every_run']})" for b in BOOTS)
          + f" -> {cells[label]:.2f} = {cells[label]:.1f}")
for b in BOOTS:
    heads = {lab: {json.loads(l)["text_head"] for l in open(f"{b}/sampled-{lab}.txt") if l.startswith('{"run"')}
             for lab in ("nofields", "top_p1")}
    print(f"{b}: nofields reply head == top_p1 reply head: {heads['nofields'] == heads['top_p1']}")
print(f"nofields / top_p1 = {cells['nofields'] / cells['top_p1']:.3f}; "
      f"top_p095 / nofields = {cells['top_p095'] / cells['nofields']:.3f}")
print("s2 for comparison (bootA / bootB): " + "; ".join(
    f"{lab} " + " / ".join(str(json.loads(open(f"{S2}/{b}/sampled-{lab}.txt").read().split("SAMPLED ", 1)[1]
                                         .splitlines()[0])["median_decode_tok_s"]) for b in ("bootA", "bootB"))
    for lab in ("default", "top_p1", "top_k20")))

# --- Exactness (README Exactness): pairs.jsonl checks -----------------------------------------------------------
print("\n## Exactness: bootD/pairs.jsonl (bootC's run stopped on a client parse error before any request: "
      "bootC/pairs-client-error.txt)")
for b in ("bootD",):
    rows = [json.loads(l) for l in open(f"{b}/pairs.jsonl") if l.startswith("{")]
    checks = [r for r in rows if "check" in r]
    reqs = [r for r in rows if "case" in r]
    print(f"{b}: checks {sum(r['equal'] for r in checks)} of {len(checks)} equal; "
          + "; ".join(f"{r['case']} sha {r['token_sha']} {r['tok_s']} tok/s" for r in reqs))

# --- Gaps against the vLLM sibling (README "Against the vLLM sibling") ------------------------------------------
print("\n## Gaps against vLLM: 100 * (1 - this / vLLM)")
decode = {r["phase"]: r["median_decode_tok_s"] for r in json.loads(open("decode.txt").read().split("SUMMARY ", 1)[1])}
v = json.load(open(f"{S2}/vllm-sibling/round36-headline.json"))["cells"]["V"]
openai = {(r["prompt"], r["temperature"]): r["decode_tps_median"] for r in json.load(open(f"{S2}/bootA/bench_openai.json"))}
g7 = {(r["prompt"], r["temperature"]): r["decode_tps_median"]
      for f in ("G7_vllm-bench.json", "G7_vllm-bench_t0_chat.json") for r in json.load(open(f"{S2}/vllm-sibling/{f}"))}
pairs = [
    ("bench_decode prose", decode["prose"], "decode.txt", v["bench_prose_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.bench_prose_c1_tok_s"),
    ("bench_decode structured", decode["structured"], "decode.txt", v["structured_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.structured_c1_tok_s"),
    ("bench_decode prose_long", decode["prose_long"], "decode.txt", v["prose_long_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.prose_long_c1_tok_s"),
]
for key in (("fibonacci-raw", 1.0), ("gpu-chat-no-think", 1.0), ("gpu-chat-no-think", 0.0)):
    pairs.append((f"bench_openai {key[0]} t={key[1]:g}", openai[key], "s2 bootA/bench_openai.json", g7[key],
                  "vllm-sibling/G7_vllm-bench*.json"))
dec = []
for name, ours, src, theirs, vsrc in pairs:
    pct = lower_pct(ours, theirs)
    dec.append(pct)
    print(f"{name}: {ours:.4f} ({src}) vs {theirs:.4f} ({vsrc}) -> {pct:.1f}% lower")
print(f"decode range: {min(dec):.1f}-{max(dec):.1f}% lower")
ours_p = {r["length"]: r["tok_s"] for r in json.load(open("bootD/prefill_cold.json"))["summary"]}
s2_p = {r["length"]: r["tok_s"] for r in json.load(open(f"{S2}/bootA/prefill_cold.json"))["summary"]}
theirs_p = {r["length"]: r["tok_s"] for r in json.load(open(f"{S2}/vllm-sibling/G7_vllm-prefill.json"))["summary"]}
pre = []
for n in sorted(ours_p):
    pct = lower_pct(ours_p[n], theirs_p[n])
    pre.append(pct)
    print(f"prefill_cold {n}: {ours_p[n]} (bootD/prefill_cold.json; s2 bootA {s2_p[n]}) vs {theirs_p[n]} "
          f"(G7_vllm-prefill.json) -> {pct:.1f}% lower")
print(f"prompt tok/s range: {min(pre):.1f}-{max(pre):.1f}% lower")

# --- Quiet prefill (README prefill row): bootD/rank0-full.log done lines after the last pairs request ----------
print("\n## Quiet prefill_cold: bootD/rank0-full.log, the done lines after pairs_default's last request")
done = [line for line in open("bootD/rank0-full.log") if "] done req-" in line]
last_pair = max(i for i, line in enumerate(done) if "sha=65f2c13cd1dc" in line)
after = done[last_pair + 1:]
prompts = [int(re.search(r"prompt=(\d+)", line)[1]) for line in after]
cached = [int(re.search(r"cached=(\d+)", line)[1]) for line in after]
rows = [json.loads(l) for l in open("bootD/prefill_cold.log") if l.startswith('{"ttft_s"')]
print(f"done lines after pairs={len(after)}, prefill_cold requests={len(rows)}, prompt sizes equal: "
      f"{prompts == [r['prompt_tokens'] for r in rows]}, cached total={sum(cached)}")

# --- Durations ------------------------------------------------------------------------------------------------
print("\n## Durations")
for b in BOOTS:
    text = open(f"{b}/run.txt").read()
    a = re.search(r"^start (\S+)$", text, re.M)[1]
    e = re.search(r"^end (\S+)$", text, re.M)[1]
    sec = (ts(e) - ts(a)).total_seconds()
    print(f"{b} ./run.sh to ready: {b}/run.txt start {a} end {e} = {sec:.0f} s")

# --- Thinking off on every bench request ------------------------------------------------------------------------
print("\n## Bench requests with thinking off: <boot>/gate/docker-head.log")
for b in BOOTS:
    done = [line for line in open(f"{b}/gate/docker-head.log") if "] done req-" in line]
    off = [line for line in done if "thinking=False" in line]
    print(f"{b}: done lines={len(done)} thinking=False={len(off)}")
