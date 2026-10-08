"""s6 (N12b): the README decode rows from the two B boots (2B, 3B: ./run.sh at the defaults, PARALLEL=4).

Per boot and cell: the file's own SUMMARY medians (bench_decode.py for the ruler cells, sampled_conc.py for the sampled
cells). Row value: the median of the per-boot medians (AGENTS.md "Final numbers come from two boots"). Sampled phases
are renamed for the table: nofields -> "sampled (no field)", top_p095 -> "sampled (top_p 0.95)". Stdlib only,
deterministic; reads the receipts unchanged.

  python3 scripts/decode_compute.py > decode.txt        (run from evidence/s6-concurrent)
"""
import json
import re
import statistics as st

BOOTS = ("2B", "3B")
FILES = ("gate/bench-frozen.out", "gate/bench-prose-long.out", "gate/bench-frozen-conc.out",
         "gate/bench-prose-long-conc.out", "sampled-nofields.txt", "sampled-top_p095.txt")
PHASE = {"nofields": "sampled (no field)", "top_p095": "sampled (top_p 0.95)"}
ORDER = ("prose", "structured", "prose_long", "sampled (no field)", "sampled (top_p 0.95)")
KEEP = ("median_decode_tok_s", "median_agg_tok_s", "ttft_s_p50", "n", "inter_chunk_ms_p50", "inter_chunk_ms_p90",
        "token_shas")

cells, settings = {}, set()
for b in BOOTS:
    for f in FILES:
        text = open(f"{b}/{f}").read()
        head = next(line for line in text.splitlines() if "max_tokens=" in line)
        settings.add(re.search(r"max_tokens=\d+ runs=\d+", head).group(0))
        for row in json.loads(text.split("SUMMARY ", 1)[1].splitlines()[0] if f.startswith("sampled")
                              else text.split("SUMMARY ", 1)[1]):
            phase = PHASE.get(row["phase"], row["phase"])
            c = cells.setdefault((phase, row["concurrency"]), {})
            c[b] = {k: row[k] for k in KEEP if k in row} | {"file": f"{b}/{f}"}
assert len(settings) == 1, settings
out = []
for (phase, conc) in sorted(cells, key=lambda k: (ORDER.index(k[0]), k[1])):
    pb = [cells[(phase, conc)][b] for b in BOOTS]
    out.append({"phase": phase, "concurrency": conc,
                "median_decode_tok_s": st.median(p["median_decode_tok_s"] for p in pb),
                "median_agg_tok_s": st.median(p["median_agg_tok_s"] for p in pb),
                "ttft_s_p50": st.median(p["ttft_s_p50"] for p in pb),
                "per_boot": dict(zip(BOOTS, pb))})
print(f"two boots {', '.join(BOOTS)} (./run.sh, PARALLEL=4 DECODE_SHARE=0.5), each: {settings.pop()} per cell and "
      "concurrency (bench_decode.py ruler cells; sampled_conc.py sampled cells); row = median of the per-boot medians")
print("SUMMARY", json.dumps(out, indent=2))
