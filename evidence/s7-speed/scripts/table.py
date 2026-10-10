"""s7: abba_table.txt from the boot directories. Stdlib only. Every number names the file it comes from.

Arms: A = TensorFold b86514a (the recipe at origin/main 38a0607: 1A, 4A), B = TensorFold 19f5478 (this branch: 2B, 3B),
both through their own run.sh at the shipped defaults. 5V = the vLLM sibling (MAX_NUM_SEQS=2). W1 = B's warm-up boot.
An arm's value is the median of its per-boot medians (two boots: their mean). delta = B/A - 1. noise = |1A - 4A| / A.
Cell value per boot: c=1 the median per-stream decode tok/s (the recipe rows), c>=2 the median wave aggregate tok/s.

  python3 scripts/table.py > abba_table.txt          (cwd evidence/s7-speed)
"""
import json
import os
import re
import statistics as st
import subprocess
from collections import Counter
from datetime import datetime

A, B, V, W = ("1A", "4A"), ("2B", "3B"), "5V", "W1"
BOOTS = ("1A", "2B", "3B", "4A")  # column order = boot order
RECIPE = "/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks"
S6 = f"{RECIPE}/evidence/s6-concurrent"
problems: list[str] = []


def summary_after(path, marker, multiline):
    if not os.path.exists(path):
        problems.append(f"missing {path}")
        return None
    text = open(path).read()
    if marker not in text:
        problems.append(f"no {marker.strip()} in {path}")
        return None
    body = text.split(marker, 1)[1]
    body = body.split("\nrc=")[0] if multiline else body.splitlines()[0]
    return json.loads(body)


def rows_by(path, marker="SUMMARY ", multiline=True):
    rows = summary_after(path, marker, multiline) or []
    return {(r["phase"], r["concurrency"]): r for r in rows}


def cell_value(r, c):
    return r["median_decode_tok_s"] if c == 1 else r["median_agg_tok_s"]


def boot_cells(tag):
    """{(group, phase, c): (value, file)} for one boot."""
    out = {}
    if tag == V:
        ruler_files = [f"{V}/bench-frozen.out", f"{V}/bench-prose-long.out"]
        div, samp = f"{V}/diverse.out", {lb: f"{V}/sampled-{lb}.out" for lb in ("nofields", "top_p095")}
    else:
        ruler_files = [f"{tag}/gate/{n}.out" for n in ("bench-frozen", "bench-prose-long", "bench-frozen-conc",
                                                        "bench-prose-long-conc")]
        div, samp = f"{tag}/diverse.txt", {lb: f"{tag}/sampled-{lb}.txt" for lb in ("nofields", "top_p095")}
    for p in ruler_files:
        for (ph, c), r in rows_by(p).items():
            out["ruler", ph, c] = (cell_value(r, c), p)
            out["ruler per-stream", ph, c] = (r["median_decode_tok_s"], p)
    for (ph, c), r in rows_by(div).items():
        out["diverse", ph, c] = (cell_value(r, c), div)
        out["diverse per-stream", ph, c] = (r["median_decode_tok_s"], div)
        if r.get("median_ms_per_round") is not None:
            out["diverse ms/round", ph, c] = (r["median_ms_per_round"], div)
    for lb, p in samp.items():
        for (ph, c), r in rows_by(p, multiline=False).items():
            out["sampled", ph, c] = (cell_value(r, c), p)
    return out


def f(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def pct(b, a):
    return None if a in (None, 0) or b is None else 100.0 * (b / a - 1)


def arm(cells, tags, key):
    vals = [cells[t].get(key, (None,))[0] for t in tags]
    return st.median(vals) if all(v is not None for v in vals) else None


cells = {t: boot_cells(t) for t in (*BOOTS, V)}
keys = sorted({k for t in BOOTS for k in cells[t]}, key=lambda k: (k[0], k[1], k[2]))

print("# s7 ABBA table. A = ./run.sh at origin/main 38a0607 (TensorFold b86514a, image tf-dsv41-flash:0.6.4-b86514a);")
print("# B = ./run.sh at agent/tensorfold-dsv41-speed 3bfdd72 (TensorFold 19f5478, image tf-dsv41-flash:0.6.6-19f5478).")
print("# Both at the shipped defaults (CONTEXT 1048576, PARALLEL 4, DECODE_SHARE 0.5, drafts 5, confidence 0.15, top_p 1.0),")
print("# no knob or debug env. Order W1 (B, empty kernel cache, not counted) 1A 2B 3B 4A, then 5V (vLLM sibling, MAX_NUM_SEQS=2).")
print("# Cell value per boot: c=1 median per-stream decode tok/s; c>=2 median wave aggregate tok/s. Arm = median of per-boot")
print("# medians. delta = B/A - 1. noiseA = |1A-4A|/A, noiseB = |2B-3B|/B. vLLM ratio = B / 5V.")
print()
print("## 1. Decode cells (tok/s)")
hdr = f"{'group':<20}{'phase':<12}{'c':>2}{'1A':>9}{'2B':>9}{'3B':>9}{'4A':>9}{'A':>9}{'B':>9}{'delta%':>8}{'noiseA%':>9}{'noiseB%':>9}{'vLLM':>9}{'B/vLLM':>8}"
for group in ("ruler", "diverse", "sampled", "ruler per-stream", "diverse per-stream", "diverse ms/round"):
    print(hdr)
    for k in [k for k in keys if k[0] == group]:
        per = [cells[t].get(k, (None,))[0] for t in BOOTS]
        a, b = arm(cells, A, k), arm(cells, B, k)
        na = None if a is None or None in per else 100 * abs(per[0] - per[3]) / a
        nb = None if b is None or None in per else 100 * abs(per[1] - per[2]) / b
        v = cells[V].get(k, (None,))[0]
        print(f"{k[0]:<20}{k[1]:<12}{k[2]:>2}" + "".join(f"{f(x):>9}" for x in (per[0], per[1], per[2], per[3], a, b))
              + f"{f(pct(b, a)):>8}{f(na):>9}{f(nb):>9}{f(v):>9}{f(None if not v or b is None else b / v, 3):>8}")
    print()

# The diverse cell's waves cycle through fixed prompt sets (9 waves: at c=4 five of prompts 0-3, four of 4-7), so a
# boot's median wave can land in either set. Matched view (as receipts/sp-P6b/abba/rep2/diverse_matched.txt): per boot
# the mean over prompt sets of the per-set median.
wave_rx = re.compile(r"^phase=(\w+) c=(\d+) run=\d+ prompts=\[([\d, ]+)\] wall=\S+ agg=([\d.]+) tok/s .*ms_round=\[([\d.,]*)\]", re.M)


def matched(tag):
    p = f"{tag}/diverse.out" if tag == V else f"{tag}/diverse.txt"
    sets = {}
    if os.path.exists(p):
        for m in wave_rx.finditer(open(p).read()):
            d = sets.setdefault((m[1], int(m[2])), {}).setdefault(m[3], {"agg": [], "ms": []})
            d["agg"].append(float(m[4]))
            d["ms"] += [float(x) for x in m[5].split(",") if x]
    out = {}
    for k, by in sets.items():
        out[k, "agg tok/s"] = st.mean(st.median(d["agg"]) for d in by.values())
        if all(d["ms"] for d in by.values()):
            out[k, "ms/round"] = st.mean(st.median(d["ms"]) for d in by.values())
    return out, p


mt = {t: matched(t) for t in (*BOOTS, V)}
print("## 1b. Diverse cell on matched prompt sets (per boot: mean over prompt sets of the per-set median wave value)")
print(f"{'phase':<8}{'c':>2} {'metric':<10}" + "".join(f"{t:>9}" for t in BOOTS) + f"{'A':>9}{'B':>9}{'delta%':>8}{'noiseA%':>9}{'noiseB%':>9}{'vLLM':>9}{'B/vLLM':>8}{'B/4A%':>8}")
for key in sorted({k for t in BOOTS for k in mt[t][0]}):
    per = [mt[t][0].get(key) for t in BOOTS]
    if None in per:
        continue
    a, b = st.median([per[0], per[3]]), st.median([per[1], per[2]])
    v = mt[V][0].get(key)
    print(f"{key[0][0]:<8}{key[0][1]:>2} {key[1]:<10}" + "".join(f"{x:>9.2f}" for x in per) + f"{a:>9.2f}{b:>9.2f}{f(pct(b, a)):>8}"
          f"{100 * abs(per[0] - per[3]) / a:>9.2f}{100 * abs(per[1] - per[2]) / b:>9.2f}{f(v):>9}{f(None if not v else b / v, 3):>8}"
          f"{f(pct(b, per[3])):>8}")
print("B/4A%: B against 4A alone (1A's c=1 diverse rounds ran longer than 4A's on 17 of 18 waves; see noiseA)")
print("files: <boot>/diverse.txt, 5V/diverse.out (the per-wave lines)")
print()
files = sorted({(k[0], cells[t][k][1].split("/", 1)[1]) for t in (*BOOTS, V) for k in cells[t]})
print("files (under each boot dir <tag>/): " + "; ".join(sorted({f"{g}: {p}" for g, p in files})))
print()

# --- exactness -----------------------------------------------------------------------------------------------
print("## 2. Exactness")
for t in (W, *BOOTS):
    p = f"{t}/gate/gate.txt"
    gate = [x for x in open(p).read().splitlines() if x.startswith("GATE=")] if os.path.exists(p) else ["missing"]
    cc = f"{t}/gate/conc-check.out"
    print(f"{t} gate: {gate[0]}  ({p}); conc-check: {open(cc).read().strip() if os.path.exists(cc) else 'missing'}  ({cc})")


def done_lines(tag, before=None):
    """rank 0's done entries -> Counter of (prompt, tokens, sha) and of (prompt, tokens, sha, rounds, accepted).
    Read from rank0-full.log: two lanes finishing together can print two entries on one log line."""
    p = f"{tag}/rank0-full.log"
    if not os.path.exists(p):
        problems.append(f"missing {p}")
        return Counter(), Counter(), p
    rx = re.compile(r"done req-\S+ prompt=(\d+) cached=\d+ thinking=\S+ tokens=(\d+) sha=(\w+) .*rounds=(\d+) accepted=(\d+)/(\d+)")
    k3, k5 = Counter(), Counter()
    for m in rx.finditer(open(p).read().replace("[tensorfold] done", "\n[tensorfold] done")):
        k3[m[1], m[2], m[3]] += 1
        k5[m[1], m[2], m[3], m[4], m[5] + "/" + m[6]] += 1
    return k3, k5, p


def s6_done(tag):
    rx = re.compile(r"done req-\S+ prompt=(\d+) cached=\d+ thinking=\S+ tokens=(\d+) sha=(\w+)")
    got = Counter()
    for m in rx.finditer(open(f"{S6}/{tag}/rank0-full.log").read()):
        got[m[1], m[2], m[3]] += 1
    return got


d3 = {t: done_lines(t) for t in BOOTS}
print()
print("rank 0 done lines (every reply of the boot), multiset of (prompt tokens, completion tokens, sha), read from")
print("rank0-full.log (two lanes can print their entries on one log line; <boot>/done-rank0.txt lists one entry a line):")
for t in BOOTS:
    print(f"  {t}: {sum(d3[t][0].values())} replies, {len(d3[t][0])} distinct (prompt, tokens, sha)  ({d3[t][2]})")
# 2B also ran the needle and its 3 lanes after the cells: compare the shared workload only (keys of 1A).
ref3, ref5 = d3["1A"][0], d3["1A"][1]
for t in ("4A", "3B", "2B"):
    c3, c5 = d3[t][0], d3[t][1]
    if t == "2B":
        extra = {k: v for k, v in c3.items() if k not in ref3}
        c3 = Counter({k: v for k, v in c3.items() if k in ref3})
        c5 = Counter({k: v for k, v in c5.items() if k[:3] in ref3})
        print(f"  2B keys absent from 1A (the needle and its lanes): {dict(extra)}")
    diff3 = (ref3 - c3) + (c3 - ref3)
    diff5 = (ref5 - c5) + (c5 - ref5)
    print(f"  {t} vs 1A: (prompt, tokens, sha) multiset {'EQUAL' if not diff3 else 'DIFFERS ' + str(dict(diff3))}; "
          f"with (rounds, accepted) {'EQUAL' if not diff5 else 'DIFFERS in ' + str(len(diff5)) + ' entries'}")
# Against s6 (b86514a, 2B/3B, a different workload): every s7 (prompt, tokens) key s6 also served must carry s6's shas.
s6 = s6_done("2B") + s6_done("3B")
s6k = {}
for (pt, tk, sha) in s6:
    s6k.setdefault((pt, tk), set()).add(sha)
for t in BOOTS:
    mine = {}
    for (pt, tk, sha) in d3[t][0]:
        mine.setdefault((pt, tk), set()).add(sha)
    shared = [k for k in mine if k in s6k]
    bad = {k: (sorted(mine[k]), sorted(s6k[k])) for k in shared if not mine[k] <= s6k[k]}
    print(f"  {t} vs s6 2B+3B (evidence/s6-concurrent/{{2B,3B}}/rank0-full.log): {len(shared)} shared (prompt, tokens) keys, "
          f"{'every sha found in s6' if not bad else 'NOT IN s6: ' + str(bad)}")

print()
print("pairs_concurrent (tools/pairs_concurrent.py --compare):")
for a_ in (*A, f"{S6}/2B"):
    for b_ in B:
        pa = f"{a_}/gate/pairs_concurrent.jsonl"
        pb = f"{b_}/gate/pairs_concurrent.jsonl"
        r = subprocess.run(["python3", f"{RECIPE}/tools/pairs_concurrent.py", "--compare", pa, pb], capture_output=True, text=True)
        last = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
        print(f"  {b_} vs {os.path.relpath(a_, RECIPE + '/evidence') if a_.startswith('/') else a_}: unequal {last.get('unequal')} of 4 (rc {r.returncode})")
for t in BOOTS:
    p = f"{t}/gate/pairs_concurrent.jsonl"
    if os.path.exists(p):
        rs = [json.loads(x) for x in open(p) if '"case"' in x]
        print(f"  {t}: " + ", ".join(f"{r['case']} {r['token_sha']}" for r in rs))

print()
print("sampled cells, token_shas per (label, c):")
for t in (*BOOTS, "s6/2B", "s6/3B"):
    base = f"{S6}/{t[3:]}" if t.startswith("s6/") else t
    parts = []
    for lb in ("nofields", "top_p095"):
        p = f"{base}/sampled-{lb}.txt"
        for (ph, c), r in sorted(rows_by(p, multiline=False).items()):
            parts.append(f"{ph} c={c} {','.join(r['token_shas'])}")
    print(f"  {t}: " + "; ".join(parts))

print()
print("diverse cell SHA_CHECK and per-prompt shas:")
div = {}
for t in BOOTS:
    p = f"{t}/diverse.txt"
    chk = [x for x in open(p).read().splitlines() if x.startswith("SHA_CHECK")] if os.path.exists(p) else ["missing"]
    div[t] = summary_after(p, "SHAS ", False) or {}
    print(f"  {t}: {chk[0]}  ({p})")
p = f"{V}/diverse.out"
chk = [x for x in open(p).read().splitlines() if x.startswith("SHA_CHECK")] if os.path.exists(p) else ["missing"]
print(f"  5V (vLLM): {chk[0]}  ({p}; vLLM greedy replies change with the batch; the tok/s are still read)")
ref = {k: sorted(v) for k, v in div["1A"].items()}
for t in ("4A", "2B", "3B"):
    got = {k: sorted(v) for k, v in div[t].items()}
    print(f"  {t} vs 1A per-prompt shas: {'EQUAL' if got == ref else 'DIFFER ' + str({k: (ref.get(k), got.get(k)) for k in set(ref) | set(got) if ref.get(k) != got.get(k)})}")
print("  1A: " + ", ".join(f"{k} {'/'.join(v)}" for k, v in sorted(ref.items())))

# --- needle, prefill ------------------------------------------------------------------------------------------
print()
print("## 3. 1,039,528-token needle beside 3 decoding lanes (2B) against s6")
for name, p in (("s7 2B", "2B/needle_lanes.jsonl"), ("s6 2B", f"{S6}/2B/needle_lanes.jsonl")):
    if not os.path.exists(p):
        print(f"  {name}: missing {p}")
        continue
    rows = [json.loads(x) for x in open(p)]
    nd = [r["needle"] for r in rows if "needle" in r and "found" in r["needle"]]
    lanes = [r for r in rows if "lane" in r]
    summ = [r for r in rows if r.get("summary")]
    ov = [r["tok_s"] for r in lanes if r["overlaps_needle"] and r["tok_s"]]
    print(f"  {name}: {json.dumps(nd[0]) if nd else 'no needle result'}")
    print(f"     lanes: {len(lanes)} replies, errors {sum(1 for r in lanes if r['error'])}, shas {sorted({r['token_sha'] for r in lanes})}, "
          f"tok/s overlapping the needle median {f(st.median(ov)) if ov else '-'}; needle_s {summ[0]['needle_s'] if summ else '-'}  ({p})")

print()
print("## 4. prefill_cold tok/s (median of 3 prompts a length), s7 boots and s6 2B/3B")
lengths = (2048, 8192, 16384, 32768, 65536)
pre = {}
for t in (*BOOTS, "s6/1A", "s6/2B", "s6/3B", "s6/4A"):
    p = f"{S6}/{t[3:]}/prefill_cold.json" if t.startswith("s6/") else f"{t}/prefill_cold.json"
    pre[t] = {r["length"]: r["tok_s"] for r in json.load(open(p))["summary"]} if os.path.exists(p) else {}
print(f"{'length':>8}" + "".join(f"{t:>9}" for t in pre) + f"{'A':>9}{'B':>9}{'delta%':>8}{'B/s6B%':>8}")
for L in lengths:
    v = {t: pre[t].get(L) for t in pre}
    a = st.median([v["1A"], v["4A"]]) if v["1A"] and v["4A"] else None
    b = st.median([v["2B"], v["3B"]]) if v["2B"] and v["3B"] else None
    s6b = st.median([v["s6/2B"], v["s6/3B"]]) if v["s6/2B"] and v["s6/3B"] else None
    print(f"{L:>8}" + "".join(f"{f(v[t], 1):>9}" for t in pre) + f"{f(a, 1):>9}{f(b, 1):>9}{f(pct(b, a)):>8}{f(pct(b, s6b)):>8}")
print("files: <boot>/prefill_cold.json; s6: evidence/s6-concurrent/<boot>/prefill_cold.json")

# --- memory, startup ------------------------------------------------------------------------------------------
print()
print("## 5. MemAvailable minima (GiB) from memwatch.tsv (every 2 s, whole boot: run.sh start to after stop)")


def memwatch(tag):
    p = f"{tag}/memwatch.tsv"
    rows = []
    if os.path.exists(p):
        for line in open(p).read().splitlines()[1:]:
            t, h, _, w, _ = (line.split("\t") + ["NA"] * 5)[:5]
            rows.append((t, int(h) if h.isdigit() else None, int(w) if w.isdigit() else None))
    return rows, p


for t in (W, *BOOTS, V):
    rows, p = memwatch(t)
    h = [(r[1], r[0]) for r in rows if r[1] is not None]
    w = [(r[2], r[0]) for r in rows if r[2] is not None]
    if not h or not w:
        print(f"  {t}: no samples ({p})")
        continue
    mh, mw = min(h), min(w)
    print(f"  {t}: head {mh[0] / 1024:.2f} at {mh[1]}, worker {mw[0] / 1024:.2f} at {mw[1]}, {len(rows)} samples  ({p})")
    # Serving window: from run.sh's end (run.txt) to the "logs and stop" mark (steps.txt).
    rt, sp = f"{t}/run.txt", f"{t}/steps.txt"
    lo = next((x.split()[1] for x in open(rt) if x.startswith("end ")), None) if os.path.exists(rt) else None
    hi = next((x.split()[0] for x in open(sp) if "logs and stop" in x), None) if os.path.exists(sp) else None
    if lo and hi:
        hs = [r[1] for r in rows if lo <= r[0] <= hi and r[1] is not None]
        ws = [r[2] for r in rows if lo <= r[0] <= hi and r[2] is not None]
        print(f"      serving {lo}..{hi}: head {min(hs) / 1024:.2f}, worker {min(ws) / 1024:.2f}  ({rt}, {sp})")

print()
print("## 6. Startup estimate (rank 0 and rank 1 lines)")
for t in (W, *BOOTS):
    p = f"{t}/startup.txt"
    lines = [x for x in open(p).read().splitlines() if "startup estimate" in x] if os.path.exists(p) else ["missing"]
    for x in lines:
        print(f"  {t}: {x}  ({p})")

if problems:
    print()
    print("PROBLEMS: " + "; ".join(problems))
