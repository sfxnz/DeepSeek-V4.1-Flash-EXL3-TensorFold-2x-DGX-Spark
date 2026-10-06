"""s5: the README numbers that no evidence file holds literally, each printed with its inputs. Stdlib only.

TF rows are the median of the two boots' medians (bootE, bootF; AGENTS.md "Final numbers come from two boots"); vLLM
rows are its one same-session boot (vllm/). Ratios are TF / vLLM.

  python3 scripts/derived.py > derived.txt        (run from evidence/s5-final)
"""
import json
import re
import statistics as st
from datetime import datetime

BOOTS = ("bootE", "bootF")
S2, S3, S4 = "../s2-run-sh-ec28f35", "../s3-default-1m", "../s4-device-nucleus"
LABELS = ("nofields", "top_p095", "top_k20")
GIB = 1024.0


def after(path, key):
    text = open(path).read()
    return json.loads(text.split(key, 1)[1].split("\nrc=")[0].strip().splitlines()[0] if key == "SAMPLED " else
                      text.split(key, 1)[1].split("\nrc=")[0])


def ruler(path):
    return {r["phase"]: r for r in json.loads(open(path).read().split("SUMMARY ", 1)[1])}


def ruler_all(d):
    return ruler(f"{d}/bench-frozen.out") | ruler(f"{d}/bench-prose-long.out")


def lail(path):
    return after(path, "SUMMARY ")


def jl(path):
    return [json.loads(line) for line in open(path) if line.startswith("{")]


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def two(vals):
    return st.median(vals)


print("# Derived README numbers, s5-final. Inputs: files in this directory and ../s2-run-sh-ec28f35, "
      "../s3-default-1m, ../s4-device-nucleus.")

print("\n## Serve: gate/serve-argv.txt and rank 0's lines (gate/startup.txt), each boot")
for b in BOOTS:
    argv = open(f"{b}/gate/serve-argv.txt").read().splitlines()
    flags = {f: re.search(rf"{f} (\S+)", argv[0])[1] for f in ("--mtp-drafts", "--mtp-confidence", "--top-p", "--context")}
    start = open(f"{b}/gate/startup.txt").read()
    est = re.findall(r"rank (\d) startup estimate ([0-9.]+) GiB within ([0-9.]+) GiB", start)
    load = re.search(r"loaded in ([0-9.]+)s", start)[1]
    print(f"{b}: image {argv[0].split()[0]}; rank 0 flags {flags}; rank 1 argv has the same --mtp-* flags: "
          f"{all(f'{k} {v}' in argv[1] for k, v in flags.items() if k.startswith('--mtp'))}; startup estimates "
          f"{est}; rank 0 loaded in {load} s")
reserve = 121.0 / 10  # max(4 GiB, MemTotal / 10); MemTotal 121 GiB (free -h) on both Sparks
est = 81.32
print(f"admission floor: startup estimate {est} GiB + reserve {reserve:.1f} GiB = {est + reserve:.2f} GiB -> "
      f"MIN_MEM_GATE_GIB 94 (was 92 = 79.09 + 12.1 at 41306d5)")

print("\n## Frozen ruler: gate/bench-*.out SUMMARY per boot; row = median of the two boots (decode.txt); vLLM: "
      "vllm/bench-*.out")
s3 = {r["phase"]: r for r in json.loads(open(f"{S3}/decode.txt").read().split("SUMMARY ", 1)[1])}
s5 = {r["phase"]: r for r in json.loads(open("decode.txt").read().split("SUMMARY ", 1)[1])}
v = ruler_all("vllm")
tf_cells = {}
for p in ("prose", "structured", "prose_long"):
    pb = [s5[p]["per_boot"][b]["median_decode_tok_s"] for b in BOOTS]
    row = s5[p]["median_decode_tok_s"]
    tf_cells[f"ruler {p}"] = row
    vv = v[p]["median_decode_tok_s"]
    print(f"{p}: E {pb[0]:.2f} / F {pb[1]:.2f} -> {row:.2f} (ttft p50 {s5[p]['ttft_s_p50']:.3f}); s3 (ec28f35, d=3) "
          f"{s3[p]['median_decode_tok_s']:.2f} -> x{row / s3[p]['median_decode_tok_s']:.3f}; vLLM {vv:.2f} (ttft p50 "
          f"{v[p]['ttft_s_p50']:.3f}, natural {v[p].get('natural_completion_tokens')}) -> TF/vLLM {row / vv:.3f}")

print("\n## L.A.I.L (the vLLM recipe's tools/measure_lail_prose.py, 512 tokens, t=0.2, 10 runs): lail.log SUMMARY")
lv = [lail(f"{b}/lail.log") for b in BOOTS]
vl = lail("vllm/lail.log")
row = two([x["median_lail_tok_s"] for x in lv])
tf_cells["L.A.I.L"] = row
print(f"TF E {lv[0]['median_lail_tok_s']:.2f} / F {lv[1]['median_lail_tok_s']:.2f} -> {row:.2f} (ttft "
      f"{two([x['median_ttft_s'] for x in lv]):.3f}); vLLM {vl['median_lail_tok_s']:.2f} (ttft {vl['median_ttft_s']:.3f}, "
      f"acceptance_len {vl.get('median_acceptance_len')}) -> TF/vLLM {row / vl['median_lail_tok_s']:.3f}")

print("\n## Sampled cells (scripts/sampled_cell.py, 9 runs): sampled-<label>.txt SAMPLED; token_sha against s4 "
      "(41306d5, 3 drafts a round)")
for lab in LABELS:
    sv = [after(f"{b}/sampled-{lab}.txt", "SAMPLED ") for b in BOOTS]
    vs = after(f"vllm/sampled-{lab}.txt", "SAMPLED ")
    s4f = f"{S4}/sampled-{lab}.txt"
    s4 = after(s4f, "SAMPLED ")
    row = two([x["median_decode_tok_s"] for x in sv])
    tf_cells[f"sampled {lab}"] = row
    shas = {b: x["token_shas"] for b, x in zip(BOOTS, sv)}
    print(f"{lab} {json.dumps(sv[0]['sampling'])}: TF E {sv[0]['median_decode_tok_s']} / F {sv[1]['median_decode_tok_s']} "
          f"-> {row:.2f}; token_sha {shas}, s4 {s4['token_shas']} ({s4['median_decode_tok_s']} tok/s), equal: "
          f"{all(x == s4['token_shas'] for x in shas.values())}, x{row / s4['median_decode_tok_s']:.3f} vs s4; vLLM "
          f"{vs['median_decode_tok_s']} (same text every run {vs['same_text_every_run']}) -> TF/vLLM "
          f"{row / vs['median_decode_tok_s']:.3f}")

print("\n## bench_openai (the engine's tools/bench_openai.py, 64 tokens, 5 reps): bench_openai.json; t=0 chat: "
      "bench_t0_chat.json")
for prompt, t in (("fibonacci-raw", 1.0), ("gpu-chat-no-think", 1.0), ("fibonacci-raw", 0.0), ("gpu-chat-no-think", 0.0)):
    tf = [next(r for r in json.load(open(f"{b}/bench_openai.json")) if r["prompt"] == prompt and r["temperature"] == t)
          for b in BOOTS]
    row = two([r["decode_tps_median"] for r in tf])
    tf_cells[f"bench_openai {prompt} t={t}"] = row
    vr = [r for r in json.load(open("vllm/bench_openai.json")) if r["prompt"] == prompt and r["temperature"] == t]
    vtxt = f"vLLM {vr[0]['decode_tps_median']:.2f} -> TF/vLLM {row / vr[0]['decode_tps_median']:.3f}" if vr else \
        "vLLM: the tool's t=0 pass not run (it fails on fibonacci-raw there); the chat prompt at t=0 is bench_t0_chat"
    print(f"{prompt} t={t}: TF E {tf[0]['decode_tps_median']:.2f} / F {tf[1]['decode_tps_median']:.2f} -> {row:.2f}; {vtxt}")
t0 = [json.load(open(f"{b}/bench_t0_chat.json"))[0] for b in BOOTS]
vt0 = json.load(open("vllm/bench_t0_chat.json"))[0]
row = two([r["decode_tps_median"] for r in t0])
tf_cells["bench_t0_chat gpu-chat-no-think t=0"] = row
print(f"bench_t0_chat gpu-chat-no-think t=0: TF E {t0[0]['decode_tps_median']:.2f} / F {t0[1]['decode_tps_median']:.2f} "
      f"-> {row:.2f} (texts identical {[r['texts_identical'] for r in t0]}); vLLM {vt0['decode_tps_median']:.2f} "
      f"(texts identical {vt0['texts_identical']}) -> TF/vLLM {row / vt0['decode_tps_median']:.3f}")

print("\n## prefill_cold (the engine's tools/prefill_cold.py, quiet; TF: s2's prompts.json, vLLM: the goldens' G7 copy, same 16 items): "
      "prefill_cold.json summary tok_s (median of 3)")
s3p = {r["length"]: r["tok_s"] for r in json.load(open(f"{S3}/bootD/prefill_cold.json"))["summary"]}
vp = {r["length"]: r for r in json.load(open("vllm/prefill_cold.json"))["summary"]}
for L in (2048, 8192, 16384, 32768, 65536):
    pb = [next(r for r in json.load(open(f"{b}/prefill_cold.json"))["summary"] if r["length"] == L) for b in BOOTS]
    row = two([r["tok_s"] for r in pb])
    tt = two([r["ttft_s"] for r in pb])
    tf_cells[f"prefill {L}"] = row
    print(f"{L}: TF E {pb[0]['tok_s']} / F {pb[1]['tok_s']} -> {row:.1f} tok/s (ttft {tt:.2f} s); s3 {s3p[L]} -> "
          f"x{row / s3p[L]:.2f}; vLLM {vp[L]['tok_s']} (ttft {vp[L]['ttft_s']:.2f} s) -> TF/vLLM {row / vp[L]['tok_s']:.3f}")
for b in BOOTS + ("vllm",):
    lines = open(f"{b}/prefill-clients.txt").read().splitlines()
    procs = sorted({re.search(r'users:\(\("(\w+)"', x)[1] for x in lines if "users:" in x})
    print(f"{b} prefill-clients.txt: connections held by {procs} (the lab app's model-list pollers)")

print("\n## Exactness: pairs.jsonl and bench_concurrent.json, each boot")
for b in BOOTS:
    rows = jl(f"{b}/pairs.jsonl")
    checks = [r for r in rows if "check" in r]
    pair = [r for r in checks if "draft:false" in r["check"]]
    old = [r for r in checks if " s2 " in r["check"]]
    tps = [r["tok_s"] for r in rows if r.get("case", "").endswith(" drafted")]
    print(f"{b}: drafted == draft:false {sum(r['equal'] for r in pair)} of {len(pair)}; drafted equal to s2's reply "
          f"(ec28f35, 3 drafts) {sum(r['equal'] for r in old)} of {len(old)}; drafted tok/s {min(tps)}-{max(tps)}")
    bc = json.load(open(f"{b}/bench_concurrent.json"))
    for r in bc["cells"]:
        if "alone" in r:
            print(f"  bench_concurrent {r['prompt']} t={r['temperature']} streams {r['streams']}: alone {r['alone']}, "
                  f"serial {r['serial']}, aggregate {r['aggregate_tps']} tok/s")

print("\n## TF / vLLM, every same-session cell")
vcell = {f"ruler {p}": v[p]["median_decode_tok_s"] for p in ("prose", "structured", "prose_long")}
vcell["L.A.I.L"] = vl["median_lail_tok_s"]
for lab in LABELS:
    vcell[f"sampled {lab}"] = after(f"vllm/sampled-{lab}.txt", "SAMPLED ")["median_decode_tok_s"]
for r in json.load(open("vllm/bench_openai.json")):
    vcell[f"bench_openai {r['prompt']} t={r['temperature']}"] = r["decode_tps_median"]
vcell["bench_t0_chat gpu-chat-no-think t=0"] = vt0["decode_tps_median"]
for L in (2048, 8192, 16384, 32768, 65536):
    vcell[f"prefill {L}"] = vp[L]["tok_s"]
for k, tv in tf_cells.items():
    if k in vcell:
        print(f"{k}: TF {tv:.2f}, vLLM {vcell[k]:.2f}, TF/vLLM {tv / vcell[k]:.3f} ({(tv / vcell[k] - 1) * 100:+.1f}%)")

print("\n## Needle (bootF/needle.jsonl, scripts/long_needle.py) and MemAvailable (bootF/memwatch.tsv, both nodes every 5 s)")
nd = [r for r in jl("bootF/needle.jsonl") if "found" in r][0]
start = [r for r in jl("bootF/needle.jsonl") if "start" in r][0]
mw = [line.split("\t") for line in open("bootF/memwatch.tsv").read().splitlines()[1:]]
win = [r for r in mw if ts(start["start"]) <= ts(r[0]) <= ts(nd["end"]) and r[3] != "NA"]
s2n = [r for r in jl(f"{S2}/ctx1m/needles.jsonl") if r.get("length") == 1040000 and "found" in r][0]
print(f"prompt {nd['prompt_tokens']} tokens (doc {start['doc_tokens']}), prefill {nd['prefill_s']:.1f} s = "
      f"{nd['prefill_s'] / 60:.1f} min, {nd['prefill_tok_s']} tok/s, found {nd['found']} ({nd['answer']!r}), wall "
      f"{nd['wall_s']} s, {start['start']} to {nd['end']}; s2 (ec28f35): {s2n['prompt_tokens']} tokens, prefill "
      f"{s2n['prefill_s']:.1f} s, {s2n['prefill_tok_s']} tok/s -> x{nd['prefill_tok_s'] / s2n['prefill_tok_s']:.2f}")
print(f"MemAvailable minimum over the needle ({len(win)} samples): head {min(int(r[1]) for r in win)} MiB = "
      f"{min(int(r[1]) for r in win) / GIB:.1f} GiB, worker {min(int(r[3]) for r in win)} MiB = "
      f"{min(int(r[3]) for r in win) / GIB:.1f} GiB; SwapFree minimum head {min(int(r[2]) for r in win)} MiB, worker "
      f"{min(int(r[4]) for r in win)} MiB")
allw = [r for r in mw if r[3] != "NA"]
print(f"over the whole boot F ({len(allw)} samples, {allw[0][0]} to {allw[-1][0]}): head min "
      f"{min(int(r[1]) for r in allw) / GIB:.1f} GiB, worker min {min(int(r[3]) for r in allw) / GIB:.1f} GiB")

print("\n## Quality: bootF/quality_full.json (qe_tf.py --full --only selfcons,gsm8k,gsm8k_think,mmlu,tools,needle, "
      "round-36 full baseline)")
q = json.load(open("bootF/quality_full.json"))
c = q["components"]
base = json.load(open("/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/results/"
                      "2026-09-27-viterbi-adopt/quality-baseline/full.json"))["components"]
for k in ("gsm8k", "gsm8k_think", "mmlu"):
    if "acc" in c.get(k, {}):
        print(f"{k}: {c[k]['acc']['k']}/{c[k]['acc']['n']} ({c[k]['acc']['rate']}); vLLM round-36 baseline "
              f"{base[k]['acc']['k']}/{base[k]['acc']['n']}")
    else:
        print(f"{k}: {c.get(k)}")
t = c["tools"]
print(f"tools: exact_args {t['exact_args']}, json_valid {t['json_valid']}, no_call {t['no_call']}")
print(f"needle: found {c['needle'].get('found')} of {len(c['needle'].get('cells', []))}; selfcons identical "
      f"{c['selfcons'].get('identical')}, aa hazard {c['selfcons'].get('aa', {}).get('hazard')}, golden "
      f"{c['selfcons'].get('golden')}")
print(f"gates: {[(g['gate'], g['pass'], g['value']) for g in q['gates']]}; pass {q['pass']}; elapsed {q['elapsed_s']} s")

print("\n## NLL: nll/nll_compare.txt")
print(open("nll/nll_compare.txt").read().strip())

print("\n## Durations")
for b in BOOTS:
    text = open(f"{b}/run.txt").read()
    a, e = (ts(re.search(rf"^{k} (\S+)$", text, re.M)[1]) for k in ("start", "end"))
    print(f"{b} ./run.sh to ready: {a:%H:%M:%S} to {e:%H:%M:%S} = {(e - a).total_seconds():.0f} s")
text = open("vllm/run.txt").read()
a, e = (ts(re.search(rf"^{k} (\S+)$", text, re.M)[1]) for k in ("start", "end"))
print(f"vLLM env AUDIT=strict ./run.sh to ready: {a:%H:%M:%S} to {e:%H:%M:%S} = {(e - a).total_seconds():.0f} s")
img = open("image-only.txt").read().split()
a, e = ts(img[1]), ts(img[-1])
print(f"IMAGE_ONLY=1 ./run.sh: {a:%H:%M:%S} to {e:%H:%M:%S} = {(e - a).total_seconds():.0f} s")

print("\n## Bench requests with thinking off: gate/docker-head.log")
for b in BOOTS:
    done = [line for line in open(f"{b}/gate/docker-head.log") if "] done req-" in line]
    print(f"{b}: done lines={len(done)} thinking=False={sum('thinking=False' in line for line in done)}")
