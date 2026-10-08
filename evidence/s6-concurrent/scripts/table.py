"""N11: abba_table.txt from the boot directories. Stdlib only. Every number names the file it comes from.

Arms: A = PARALLEL=1 (1A, 4A), B = PARALLEL=4 DECODE_SHARE=0.5 (2B, 3B); 5C / 6D = PARALLEL=4 at share 0.25 / 1;
7V = the vLLM sibling (MAX_NUM_SEQS=2). An arm's value is the median of its per-boot medians (two boots: their mean).

  python3 scripts/table.py > abba_table.txt          (from receipts/n11)
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone

A, B, S = ("1A", "4A"), ("2B", "3B"), ("5C", "6D")
V = "7V"
RECIPE = "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks"
S5 = f"{RECIPE}/evidence/s5-final"
C7 = "/home/sfxnz/projects/data/tf-dsv41/receipts/c7"
VLLM_R36 = {"prose": 81.02, "structured": 156.29}
problems: list[str] = []


def jl(path):
    return [json.loads(x) for x in open(path) if x.startswith("{")] if os.path.exists(path) else []


def ruler(path):
    """bench_decode.py's SUMMARY -> {(phase, c): row}."""
    if not os.path.exists(path):
        problems.append(f"missing {path}")
        return {}
    text = open(path).read()
    if "SUMMARY " not in text:
        problems.append(f"no SUMMARY in {path}")
        return {}
    rows = json.loads(text.split("SUMMARY ", 1)[1].split("\nrc=")[0])
    return {(r["phase"], r["concurrency"]): r for r in rows}


def sampled(path):
    if not os.path.exists(path):
        problems.append(f"missing {path}")
        return {}
    line = next((x for x in open(path) if x.startswith("SUMMARY ")), None)
    return {(r["phase"], r["concurrency"]): r for r in json.loads(line[8:])} if line else {}


def mid(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def f(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def pct(b, a):
    return None if a in (None, 0) or b is None else 100.0 * (b / a - 1)


def boot_cells(tag):
    """Every speed cell of a TF boot -> {(source, phase, c): (value, file)}."""
    out = {}
    for name in ("bench-frozen", "bench-prose-long", "bench-frozen-conc", "bench-prose-long-conc"):
        p = f"{tag}/gate/{name}.out"
        for (ph, c), r in ruler(p).items():
            out["ruler", ph, c] = (r["median_decode_tok_s"] if c == 1 else r["median_agg_tok_s"], p)
            out["ruler-per-stream", ph, c] = (r["median_decode_tok_s"], p)
            out["ruler-ttft", ph, c] = (r["ttft_s_p50"], p)
    for label in ("nofields", "top_p095"):
        p = f"{tag}/sampled-{label}.txt"
        for (ph, c), r in sampled(p).items():
            out["sampled", ph, c] = (r["median_decode_tok_s"] if c == 1 else r["median_agg_tok_s"], p)
            out["sampled-shas", ph, c] = (r["token_shas"], p)
    p = f"{tag}/prefill_cold.json"
    if os.path.exists(p):
        for r in json.load(open(p))["summary"]:
            out["prefill_cold", r["length"], 1] = (r["tok_s"], p)
    return out


def done_shas(tag):
    """rank 0's `done` lines -> {(prompt tokens, completion tokens): {sha: count}}."""
    p = f"{tag}/rank0-full.log"
    got = {}
    if os.path.exists(p):
        for m in re.finditer(r"done req-\S+ prompt=(\d+) cached=\d+ thinking=\S+ tokens=(\d+) sha=(\w+)", open(p).read()):
            k = (int(m[1]), int(m[2]))
            got.setdefault(k, {}).setdefault(m[3], 0)
            got[k][m[3]] += 1
    return got


def memwatch(tag):
    p = f"{tag}/memwatch.tsv"
    rows = []
    if os.path.exists(p):
        for line in open(p).read().splitlines()[1:]:
            t, h, _, w, _ = line.split("\t")
            rows.append((datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ"), int(h) if h != "NA" else None,
                         int(w) if w != "NA" else None))
    return rows, p


def steps(tag):
    p = f"{tag}/steps.txt"
    out = []
    if os.path.exists(p):
        for line in open(p):
            t, _, what = line.rstrip("\n").split(" ", 2)
            out.append((datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ"), what))
    return out


def minima(rows, lo=None, hi=None):
    sel = [r for r in rows if (lo is None or r[0] >= lo) and (hi is None or r[0] <= hi)]
    h = [r[1] for r in sel if r[1] is not None]
    w = [r[2] for r in sel if r[2] is not None]
    return (min(h) / 1024 if h else None, min(w) / 1024 if w else None)


def probe(tag):
    p = f"{tag}/startup.txt"
    rows = []
    if os.path.exists(p):
        for line in open(p):
            if "[n11probe] {" in line:
                rows.append((line.split()[0], json.loads(line.split("[n11probe] ", 1)[1])))
    return rows, p


cells = {b: boot_cells(b) if os.path.isdir(b) and b not in S else {} for b in (*A, *B, *S)}   # S boots: TTFT only
vl = {}
for name in ("bench-frozen", "bench-prose-long"):
    for (ph, c), r in ruler(f"{V}/{name}.out").items():
        vl["ruler", ph, c] = (r["median_decode_tok_s"] if c == 1 else r["median_agg_tok_s"], f"{V}/{name}.out")
        vl["ruler-ttft", ph, c] = (r["ttft_s_p50"], f"{V}/{name}.out")
for label in ("nofields", "top_p095"):
    for (ph, c), r in sampled(f"{V}/sampled-{label}.out").items():
        vl["sampled", ph, c] = (r["median_decode_tok_s"] if c == 1 else r["median_agg_tok_s"], f"{V}/sampled-{label}.out")


def arm(tags, key):
    vals = [cells[t][key][0] for t in tags if t in cells and key in cells[t]]
    return mid(vals), vals


print("# N11 ABBA table (receipts/n11). A = ./run.sh PARALLEL=1, B = ./run.sh (PARALLEL=4 DECODE_SHARE=0.5), both at "
      "CONTEXT=1048576 and the recipe defaults, image tf-dsv41-flash:0.6.4-b86514a (TF b86514a), recipe b55aac8.")
print("# Boot order 1A 2B 3B 4A, then 5C (share 0.25), 6D (share 1), 7V (vLLM sibling, MAX_NUM_SEQS=2), after two "
      "warm-up boots W1 (PARALLEL=4) and W2 (PARALLEL=1) that filled the new Triton cache (not counted).")
print("# Arm value = median of per-boot medians. c=1 ruler cells: median per-stream decode tok/s (as the recipe rows); "
      "c>=2: median wave aggregate tok/s (bench_decode.py's median_agg_tok_s).")

# ---- 1. B against A at c = 1 ----------------------------------------------------------------------------------
print("\n## 1. c = 1: B (PARALLEL=4, a lone lane) against A (PARALLEL=1, legacy). Gate: B within 3% of A.")
print(f"{'cell':34} {'A 1A':>8} {'A 4A':>8} {'A':>8} {'B 2B':>8} {'B 3B':>8} {'B':>8} {'B/A-1':>7}  gate")
c1_keys = [("ruler", "prose", 1), ("ruler", "structured", 1), ("ruler", "prose_long", 1), ("sampled", "nofields", 1),
           ("sampled", "top_p095", 1)] + [("prefill_cold", n, 1) for n in (2048, 8192, 16384, 32768, 65536)]
c1_miss = []
for key in c1_keys:
    a, av = arm(A, key)
    b, bv = arm(B, key)
    d = pct(b, a)
    ok = d is not None and d >= -3.0
    gated = key[0] in ("ruler", "prefill_cold")
    if gated and not ok:
        c1_miss.append((key, d))
    name = f"{key[0]} {key[1]}"
    row = [cells[t].get(key, (None,))[0] for t in (*A, *B)]
    print(f"{name:34} {f(row[0]):>8} {f(row[1]):>8} {f(a):>8} {f(row[2]):>8} {f(row[3]):>8} {f(b):>8} "
          f"{f(d):>6}%  {('PASS' if ok else 'MISS') if gated else '(reported)'}")
print("files: <boot>/gate/bench-frozen.out, <boot>/gate/bench-prose-long.out, <boot>/sampled-<label>.txt, "
      "<boot>/prefill_cold.json (prefill_cold tok/s per length, median of 3 reps)")
print("c=1 gate (ruler and prefill_cold cells): " + ("PASS" if not c1_miss else f"MISS {c1_miss}"))
print("B against s5 (55.8 / 97.3 / 40.7, evidence/s5-final/decode.txt): " + " / ".join(
    f(arm(B, ("ruler", ph, 1))[0]) for ph in ("prose", "structured", "prose_long")))

# ---- 2. c = 2 and c = 4 against vLLM and the predictions --------------------------------------------------------
print("\n## 2. Concurrent aggregates (tok/s) against vLLM and the restated predictions (conc-design R15.mp1)")
print(f"{'cell':26} {'c':>2} {'B 2B':>8} {'B 3B':>8} {'B':>8} {'A (queued)':>10} {'vLLM 7V':>8} {'B/vLLM':>7} {'R36':>7}  prediction")
pred = {("prose", 2): "0.92-0.97x vLLM", ("structured", 2): "about 0.85x vLLM", ("prose", 4): "93-104 tok/s"}
for src, ph in (("ruler", "prose"), ("ruler", "structured"), ("ruler", "prose_long"), ("sampled", "nofields"),
                ("sampled", "top_p095")):
    for c in (2, 4):
        key = (src, ph, c)
        b, bv = arm(B, key)
        a, _ = arm(A, key)
        v = vl.get(key, (None,))[0]
        row = [cells[t].get(key, (None,))[0] for t in B]
        r36 = VLLM_R36.get(ph) if src == "ruler" and c == 2 else None
        ratio = b / v if b and v else None
        extra = ""
        if (ph, c) in pred and src == "ruler":
            extra = pred[ph, c]
            if c == 2 and r36:
                extra += f"; B/R36 {f(b / r36 if b else None, 3)}"
        print(f"{src + ' ' + ph:26} {c:>2} {f(row[0]):>8} {f(row[1]):>8} {f(b):>8} {f(a):>10} {f(v):>8} "
              f"{f(ratio, 3):>7} {f(r36):>7}  {extra}")
print("files: <boot>/gate/bench-frozen-conc.out, <boot>/gate/bench-prose-long-conc.out, <boot>/sampled-<label>.txt; "
      "7V/bench-frozen.out, 7V/bench-prose-long.out, 7V/sampled-<label>.out. vLLM c=4 queues behind MAX_NUM_SEQS=2. "
      "A (PARALLEL=1) queues every request: its c>=2 'aggregate' is serial decoding.")
print("vLLM c=1 (7V; s5-final same-session vLLM was 47.88 / 82.57 / 43.32, evidence/s5-final/vllm): " + ", ".join(f"{k[1]} {f(vl.get(k, (None,))[0])}" for k in
                                   (("ruler", "prose", 1), ("ruler", "structured", 1), ("ruler", "prose_long", 1))))
print("per-stream decode tok/s, B: " + "; ".join(
    f"{ph} c={c} {f(arm(B, ('ruler-per-stream', ph, c))[0])}" for ph in ("prose", "structured", "prose_long") for c in (1, 2, 4)))
print("TTFT p50 s, B: " + "; ".join(
    f"{ph} c={c} {f(arm(B, ('ruler-ttft', ph, c))[0], 3)}" for ph in ("prose", "structured") for c in (1, 2, 4))
      + " | vLLM: " + "; ".join(f"{ph} c={c} {f(vl.get(('ruler-ttft', ph, c), (None,))[0], 3)}"
                                for ph in ("prose", "structured") for c in (1, 2, 4)))

# ---- 3. Exactness -----------------------------------------------------------------------------------------------
print("\n## 3. Exactness")
for t in (*A, *B):
    for name in ("conc-check",):
        p = f"{t}/gate/{name}.out"
        ex = open(f"{t}/gate/{name}.exit").read().strip() if os.path.exists(f"{t}/gate/{name}.exit") else "missing"
        txt = open(p).read().strip().splitlines()[-1] if os.path.exists(p) else "missing"
        print(f"{t} bench_concurrent plain + --mixed --stagger-ms 2000: exit {ex}: {txt}  ({p})")
    tot = {"equal": 0, "unequal": 0, "failed": 0}
    for fn in ("bench_concurrent.json", "bench_concurrent_mixed.json"):
        p = f"{t}/gate/{fn}"
        if os.path.exists(p):
            for cell in json.load(open(p))["cells"]:
                for k in tot:
                    tot[k] += (cell.get("alone") or {}).get(k, 0)
    print(f"   alone checks summed over both files: {tot}")
# pairs_concurrent: every B-side run against both A runs
pc = {t: f"{t}/gate/pairs_concurrent.jsonl" for t in (*A, *B)}
pc.update({t: f"{t}/pairs_concurrent.jsonl" for t in S})
for b in (*B, *S):
    for a in A:
        if os.path.exists(pc[b]) and os.path.exists(pc[a]):
            r = subprocess.run([sys.executable, f"{RECIPE}/tools/pairs_concurrent.py", "--compare", pc[a], pc[b]],
                               capture_output=True, text=True)
            last = json.loads(r.stdout.strip().splitlines()[-1])
            print(f"pairs_concurrent {b} vs {a}: unequal {last['unequal']} of 4 (rc {r.returncode})  ({pc[a]}, {pc[b]})")
            if r.returncode:
                problems.append(f"pairs_concurrent {b} vs {a}: {r.stdout}")
for t in (*A, *B, *S):
    rows = jl(pc[t])
    if rows:
        print(f"   {t}: " + ", ".join(f"{r['case']} {r.get('token_sha')} ({r.get('prompt_tokens')} prompt)" for r in rows if "case" in r))
# TTFT cells: fill / live / bg shas against A alone
ref = {}
for a in A:
    for r in jl(f"{a}/ttft_alone.jsonl"):
        ref.setdefault((r["kind"], r["tag"]), set()).add(r["token_sha"])
print("A references (ttft_alone.jsonl): " + "; ".join(f"{k[0]}{('(' + k[1] + ')') if k[1] else ''} {sorted(v)}"
                                                     for k, v in sorted(ref.items())))
for t in (*B, *S):
    bad, n = [], 0
    for r in jl(f"{t}/ttft_cells.jsonl"):
        if r.get("summary"):
            continue
        n += 1
        want = ref.get((r["kind"], r["tag"] if r["kind"] == "FILL" else ""), set())
        if len(want) != 1 or r["token_sha"] not in want:
            bad.append(f"{r['cell']} {r['kind']} {r['token_sha']} want {sorted(want)}")
    print(f"ttft cells {t}: {n} streams, {len(bad)} unequal to A{': ' + '; '.join(bad) if bad else ''}  ({t}/ttft_cells.jsonl)")
# A vs published s5 and c7
def cases(path):
    return {r["case"]: r["token_sha"] for r in jl(path) if "case" in r}
s5p = cases(f"{S5}/bootF/pairs.jsonl")
c7p = {}
for p in sorted(glob.glob(f"{C7}/*/pairs.jsonl")):
    for k, v in cases(p).items():
        c7p.setdefault(k, set()).add(v)
for t in (*A, *B):
    mine = cases(f"{t}/pairs_nucleus.jsonl")
    eq = [k for k in s5p if mine.get(k) == s5p[k]]
    ne = [k for k in s5p if mine.get(k) != s5p[k]]
    print(f"{t} pairs_nucleus vs s5 bootF/pairs.jsonl: {len(eq)} equal, {len(ne)} differ{' ' + str(ne) if ne else ''}")
    mine = cases(f"{t}/pairs_q2.jsonl")
    shared = [k for k in c7p if "t=0.0" in k]       # greedy: the server's top_p does not enter
    eq = [k for k in shared if len(c7p[k]) == 1 and mine.get(k) in c7p[k]]
    ne = [k for k in shared if not (len(c7p[k]) == 1 and mine.get(k) in c7p[k])]
    other = [(k, mine.get(k), sorted(c7p[k])) for k in c7p if k not in shared]
    print(f"{t} q2 pairs vs receipts/c7/*/pairs.jsonl (4 boots), shared (greedy) cells: {len(eq)} equal, {len(ne)} differ"
          f"{' ' + str([(k, mine.get(k), sorted(c7p[k])) for k in ne]) if ne else ''}; not shared (c7 served without "
          f"--top-p, so its t=1.0 requests drew at the family's top_p 0.95; the recipe serves --top-p 1.0): {other}")
s5s = {}
for label in ("nofields", "top_p095"):
    line = next((x for x in open(f"{S5}/bootF/sampled-{label}.txt") if x.startswith("SAMPLED ")), None)
    s5s[label] = json.loads(line[8:])["token_shas"] if line else None
for t in (*A, *B):
    got = {k: v[0] for k, v in cells[t].items() if k[0] == "sampled-shas"}
    print(f"{t} sampled shas: " + "; ".join(f"{k[1]} c={k[2]} {v}" for k, v in sorted(got.items()))
          + f"  (s5 bootF: nofields {s5s['nofields']}, top_p095 {s5s['top_p095']})")
ds = {t: done_shas(t) for t in (*A, *B)}
keys = sorted(set().union(*[set(d) for d in ds.values()]))
print("rank 0 done lines, ruler prompts (prompt tokens, completion tokens): sha per boot (count)")
for k in keys:
    if k[0] in (26, 29, 42) and k[1] == 200:
        print(f"   {k}: " + "; ".join(f"{t} {ds[t].get(k)}" for t in (*A, *B)))

# ---- 4. TTFT of a 16k prompt while 3 lanes decode ----------------------------------------------------------------
print("\n## 4. TTFT of a 16k prompt arriving while 3 lanes decode (ttft.py), by --decode-share")
alone_fill = [r["ttft_s"] for a in A for r in jl(f"{a}/ttft_alone.jsonl") if r["kind"] == "FILL"]
alone_live = [r.get("tok_s") for a in A for r in jl(f"{a}/ttft_alone.jsonl") if r["kind"] == "LIVE"]
print(f"A alone: 16k FILL TTFT median {f(mid(alone_fill), 2)} s over {len(alone_fill)}; LIVE alone tok/s {alone_live}")
for t, share in (("5C", 0.25), ("2B", 0.5), ("3B", 0.5), ("6D", 1.0)):
    rows = [r for r in jl(f"{t}/ttft_cells.jsonl") if r.get("summary")]
    for r in rows:
        print(f"{t} share {share} {r['cell']:>5}: fill TTFT {f(r['fill_ttft_s'], 2)} s ({r['fill_prompt_tokens']} tok); "
              f"live tok/s in fill (rounds/s x tokens/round) {r['live_tok_s_in_fill']}, max gap {r['live_max_gap_in_fill_s']} s; "
              f"live tok/s whole {r['live_tok_s']}"
              + (f"; BG sha {r.get('bg_sha')} max gap {r.get('bg_max_gap_s')} s wall {r.get('bg_wall_s')} s" if 'bg_sha' in r else "")
              + (f"; errors {r['errors']}" if r["errors"] else ""))
    reps = [r["fill_ttft_s"] for r in rows if r["cell"] != "yield"]
    print(f"{t} share {share}: median fill TTFT of the no-yield reps {f(mid(reps), 2)} s  ({t}/ttft_cells.jsonl)")

# ---- 5. Memory ---------------------------------------------------------------------------------------------------
print("\n## 5. Memory. memwatch.tsv every 2 s on both nodes; GiB = MiB / 1024. Predictions (conc-memory-perf §4, 4 x 1M): "
      "head 14.8 serving, 11.7 worst case; worker 21.9. Gate: head >= 5 GiB.")
for t in (*A, *B, *S):
    rows, p = memwatch(t)
    sp = steps(t)
    if not rows or not sp:
        print(f"{t}: missing memwatch/steps")
        continue
    t_warm = next((x for x, w in sp if w.startswith("warm pass")), None)
    t_stop = next((x for x, w in sp if w.startswith("logs and stop")), None)
    t_needle = next((x for x, w in sp if w.startswith("needle")), None)
    boot = minima(rows, None, t_warm)
    serving = minima(rows, t_warm, t_stop)
    line = (f"{t}: boot/load min head {f(boot[0])} worker {f(boot[1])}; serving min head {f(serving[0])} "
            f"worker {f(serving[1])}")
    if t_needle:
        nd = minima(rows, t_needle, t_stop)
        line += f"; 1M needle with 3 lanes decoding: head {f(nd[0])} worker {f(nd[1])}"
    print(line + f"  ({p})")
    for k in ("free-serving.txt",):
        pass
print("Startup estimate and probe (n11probe around graph warm; GiB; seconds include Triton compiles only in W1/W2):")
for t in (*A, *B, *S):
    rows, p = probe(t)
    est = re.findall(r"rank (\d) startup estimate ([0-9.]+) GiB within ([0-9.]+) GiB; native \d+, allocated prompt/reply "
                     r"window (\d+)", open(p).read()) if os.path.exists(p) else []
    print(f"{t}: startup estimate {[(r, e, w) for r, e, w, _ in est]} (rank, estimate, budget), window {[x[3] for x in est]}  ({p})")
    for rank, r in rows:
        if r["event"] == "after" and r["what"] in ("Graphs.warm", "LaneGraphs.warm"):
            print(f"   {rank} {r['what']}: {r.get('returned')} graphs in {r['seconds']} s; memory_reserved {r['reserved']} "
                  f"(d {r['d_reserved']}), allocated d {r['d_allocated']}, mem_get_info free {r['dev_free']} "
                  f"(d {r['d_dev_free']}), MemAvailable {r['memavail']} (d {r['d_memavail']})")
        if r["event"] == "after" and r["what"] in ("DeepSeekV41Engine._warm", "DeepSeekV41Engine._warm_lanes",
                                                   "DeepSeekV41Engine._lanes"):
            print(f"   {rank} {r['what']}: {r['seconds']} s; reserved d {r['d_reserved']}, mem_get_info free d "
                  f"{r['d_dev_free']}, MemAvailable d {r['d_memavail']} -> {r['memavail']}")
for t in (*A, *B):
    p = f"{t}/needle_lanes.jsonl"
    if os.path.exists(p):
        rows = jl(p)
        nd = [r["needle"] for r in rows if "needle" in r and "prompt_tokens" in r["needle"]]
        live = [r for r in rows if "lane" in r]
        summ = next((r for r in rows if r.get("summary")), {})
        for n in nd:
            print(f"{t} needle: prompt {n['prompt_tokens']} tokens, found {n['found']}, token_sha {n['token_sha']} "
                  f"(s5 bootF alone: 4dfbacf6d86b), answer wall {n['wall_s']} s = "
                  f"{n['prompt_tokens'] / n['wall_s']:.1f} prompt tok/s (s5 bootF alone: wall 756.6 s, prefill 754.3 s, "
                  f"1378.1 tok/s); its tensorfold prefill_s field reads {f(n['prefill_s'], 1)} s under lanes  ({p})")
        ov = [r for r in live if r["overlaps_needle"]]
        print(f"   live requests {len(live)} ({len(ov)} overlapping the needle), errors {[r['error'] for r in live if r['error']]}, "
              f"shas {sorted({r['token_sha'] for r in live if r['completion_tokens'] == 1500})}, tok/s of overlapping "
              f"{[r['tok_s'] for r in ov]}, max gap {max([r['max_gap_s'] or 0 for r in ov] or [0])} s; needle rc "
              f"{summ.get('needle_rc')}, {summ.get('needle_s')} s")

# ---- 6. Gate constants for N12b --------------------------------------------------------------------------------
print("\n## 6. MIN_MEM_GATE_GIB for N12b (run.sh mp5 formula: 75.99 + max(5.33, 3.71 + 0.95 (P-1) CONTEXT/1048576) + 12.1, "
      "rounded up)")
for P in (1, 2, 3, 4):
    est = 75.99 + max(5.33, 3.71 + 0.95 * (P - 1))
    print(f"   P={P} at 1M: formula estimate {est:.2f} GiB, floor {int(-(-(est + 12.1) // 1))}")
for t in (*A, *B, *S):
    p = f"{t}/startup.txt"
    if os.path.exists(p):
        e = re.findall(r"startup estimate ([0-9.]+) GiB", open(p).read())
        print(f"   {t} measured estimate {e} -> floor {[int(-(-(float(x) + 12.1) // 1)) for x in e]}  ({p})")

if problems:
    print("\n## Problems")
    for p in problems:
        print("   " + p)
