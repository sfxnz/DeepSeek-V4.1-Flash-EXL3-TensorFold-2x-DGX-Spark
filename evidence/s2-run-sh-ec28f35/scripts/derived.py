"""s2: the README numbers that no evidence file holds literally (unit changes, rounding, durations, ratios, counts),
each printed with its inputs. Stdlib only, deterministic.

  python3 scripts/derived.py > derived.txt        (run from evidence/s2-run-sh-ec28f35)
"""
import json
import re
from datetime import datetime

MIB_PER_GIB = 1024


def gib(mib):
    return mib / MIB_PER_GIB


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def mmss(sec):
    return f"{int(sec) // 60} min {int(sec) % 60} s"


def lower_pct(ours, theirs):
    return 100 * (1 - ours / theirs)


print("# Derived README numbers, s2-run-sh-ec28f35. Inputs are files in this directory or ../s0-engine-receipts.")

# --- Window table (README "Window and long prompts"): needles.jsonl, MiB -> GiB, prompt tok/s recomputed ---------
print("\n## Window table: ctx1m/needles.jsonl (min_avail_mib_* / 1024; prompt tok/s = prompt_tokens / prefill_s)")
rows = [json.loads(line) for line in open("ctx1m/needles.jsonl")]
results = [r for r in rows if "prompt_tokens" in r]
mems = {r["cell"]: r for r in rows if "cell" in r}
for r in results:
    m = mems[f"{r['length']}@{r['depth']}"]
    tps = r["prompt_tokens"] / r["prefill_s"]
    print(f"prompt_tokens={r['prompt_tokens']} depth={r['depth']} prefill_s={r['prefill_s']:.4f} "
          f"file_prefill_tok_s={r['prefill_tok_s']} -> tok/s {tps:.3f} = {tps:.0f}; "
          f"min_avail head {m['min_avail_mib_head']} MiB = {gib(m['min_avail_mib_head']):.2f} GiB, "
          f"worker {m['min_avail_mib_worker']} MiB = {gib(m['min_avail_mib_worker']):.2f} GiB")
big = max(results, key=lambda r: r["prompt_tokens"])
print(f"longest prefill {big['prefill_s']:.1f} s = {big['prefill_s'] / 60:.1f} min")

# --- Whole-boot memory minimum (README, recipe.yaml "never below 22 GiB"): memwatch.tsv --------------------------
print("\n## Memory minimum over the CONTEXT=1048576 boot: ctx1m/memwatch.tsv")
lines = open("ctx1m/memwatch.tsv").read().splitlines()
cols = lines[0].split("\t")
samples = [dict(zip(cols, line.split("\t"))) for line in lines[1:]]
times = [ts(s["utc"]) for s in samples]
gap = max((b - a).total_seconds() for a, b in zip(times, times[1:]))
print(f"samples={len(samples)} from {samples[0]['utc']} to {samples[-1]['utc']} max_gap_s={gap:.0f}")
for node in ("head", "worker"):
    low = min(samples, key=lambda s: int(s[f"{node}_avail_mib"]))
    mib = int(low[f"{node}_avail_mib"])
    print(f"{node}: min MemAvailable {mib} MiB = {gib(mib):.2f} GiB at {low['utc']}")

# --- Gaps against the vLLM sibling (README "Against the vLLM sibling") --------------------------------------------
print("\n## Gaps against vLLM: 100 * (1 - this / vLLM)")
decode = {r["phase"]: r["median_decode_tok_s"] for r in json.loads(open("decode.txt").read().split("SUMMARY ", 1)[1])}
v = json.load(open("vllm-sibling/round36-headline.json"))["cells"]["V"]
openai = {(r["prompt"], r["temperature"]): r["decode_tps_median"] for r in json.load(open("bootA/bench_openai.json"))}
g7 = {(r["prompt"], r["temperature"]): r["decode_tps_median"]
      for f in ("G7_vllm-bench.json", "G7_vllm-bench_t0_chat.json") for r in json.load(open(f"vllm-sibling/{f}"))}
pairs = [
    ("bench_decode prose", decode["prose"], "decode.txt", v["bench_prose_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.bench_prose_c1_tok_s"),
    ("bench_decode structured", decode["structured"], "decode.txt", v["structured_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.structured_c1_tok_s"),
    ("bench_decode prose_long", decode["prose_long"], "decode.txt", v["prose_long_c1_tok_s"]["arm_median"],
     "round36-headline.json cells.V.prose_long_c1_tok_s"),
]
for key in (("fibonacci-raw", 1.0), ("gpu-chat-no-think", 1.0), ("gpu-chat-no-think", 0.0)):
    pairs.append((f"bench_openai {key[0]} t={key[1]:g}", openai[key], "bootA/bench_openai.json", g7[key],
                  "vllm-sibling/G7_vllm-bench*.json"))
dec = []
for name, ours, src, theirs, vsrc in pairs:
    pct = lower_pct(ours, theirs)
    dec.append(pct)
    print(f"{name}: {ours:.4f} ({src}) vs {theirs:.4f} ({vsrc}) -> {pct:.1f}% lower")
print(f"decode range: {min(dec):.1f}-{max(dec):.1f}% lower")
ours_p = {r["length"]: r["tok_s"] for r in json.load(open("bootA/prefill_cold.json"))["summary"]}
theirs_p = {r["length"]: r["tok_s"] for r in json.load(open("vllm-sibling/G7_vllm-prefill.json"))["summary"]}
pre = []
for n in sorted(ours_p):
    pct = lower_pct(ours_p[n], theirs_p[n])
    pre.append(pct)
    print(f"prefill_cold {n}: {ours_p[n]} (bootA/prefill_cold.json) vs {theirs_p[n]} (G7_vllm-prefill.json) "
          f"-> {pct:.1f}% lower")
print(f"prompt tok/s range: {min(pre):.1f}-{max(pre):.1f}% lower")

# --- Durations (README Image, Gotchas) -----------------------------------------------------------------------------
print("\n## Durations")
stamps = [line for line in open("image-only.txt").read().splitlines()
          if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", line)]
sec = (ts(stamps[-1]) - ts(stamps[0])).total_seconds()
print(f"IMAGE_ONLY=1 ./run.sh: image-only.txt {stamps[0]} to {stamps[-1]} = {sec:.0f} s = {mmss(sec)}")
for boot in ("bootA", "bootB", "ctx1m"):
    text = open(f"{boot}/run.txt").read()
    a = re.search(r"^start (\S+)$", text, re.M)[1]
    b = re.search(r"^end (\S+)$", text, re.M)[1]
    sec = (ts(b) - ts(a)).total_seconds()
    print(f"{boot} ./run.sh to ready: {boot}/run.txt start {a} end {b} = {sec:.0f} s = {mmss(sec)}")

# --- Admission reserve (README Memory): max(4 GiB, MemTotal / 10) from free -h ------------------------------------
print("\n## Admission reserve: ../s0-engine-receipts/free_before_s65538.txt (free -h total, rounded by free)")
for total in re.findall(r"^Mem:\s+(\S+)", open("../s0-engine-receipts/free_before_s65538.txt").read(), re.M):
    t = float(total.rstrip("Gi"))
    print(f"MemTotal {total} -> reserve max(4, {t:g} / 10) = {max(4.0, t / 10):.1f} GiB")
est = re.search(r"startup estimate ([0-9.]+) GiB", open("ctx1m/run.txt").read())[1]
print(f"gate floor: native-window estimate {est} GiB (ctx1m/run.txt) + {max(4.0, t / 10):.1f} = "
      f"{float(est) + max(4.0, t / 10):.2f} GiB")

# --- Thinking off on every bench request (README Measured) ---------------------------------------------------------
print("\n## Bench requests with thinking off: bootA/gate/docker-head.log")
done = [line for line in open("bootA/gate/docker-head.log") if "] done req-" in line]
off = [line for line in done if "thinking=False" in line]
print(f"done lines={len(done)} thinking=False={len(off)}")
