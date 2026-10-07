"""s2: the README decode rows from the two boots' frozen-ruler runs (tools/session_gate.sh output, unchanged).

Per boot and cell: bench_decode.py's own SUMMARY medians. Row value: the median of the per-boot medians (AGENTS.md
"Final numbers come from two boots"). Also the pooled median over both boots' per-run values (the vLLM sibling's
round-36 method), for comparison. Stdlib only, deterministic.

  python3 scripts/decode_compute.py > decode.txt        (run from evidence/s2-run-sh-ec28f35)
"""
import json
import re
import statistics as st

BOOTS = ("bootA", "bootB")
FILES = ("gate/bench-frozen.out", "gate/bench-prose-long.out")
RUN = re.compile(r"^phase=(\w+) c=(\d+) run=\d+ .* agg=([0-9.]+) tok/s per_stream=\[([0-9.,]+)\] ttft=\[([0-9.,]+)\]")

cells, headers = {}, set()
for b in BOOTS:
    for f in FILES:
        text = open(f"{b}/{f}").read()
        h = text.splitlines()[0]
        headers.add(re.sub(r" phases=.*", "", h))
        for row in json.loads(text.split("SUMMARY ", 1)[1]):
            c = cells.setdefault((row["phase"], row["concurrency"]), {"per_boot": {}, "runs": {}})
            c["per_boot"][b] = {k: row[k] for k in ("median_decode_tok_s", "median_agg_tok_s", "ttft_s_p50", "n",
                                                    "inter_chunk_ms_p50", "natural_completion_tokens",
                                                    "post_eos_fraction")} | {"file": f"{b}/{f}"}
        for line in text.splitlines():
            m = RUN.match(line)
            if m:
                c = cells[(m[1], int(m[2]))]["runs"].setdefault(b, [])
                c.append((float(m[3]), [float(x) for x in m[4].split(",")], [float(x) for x in m[5].split(",")]))
assert len(headers) == 1, headers
out = []
for (phase, conc), c in cells.items():
    pb = [c["per_boot"][b] for b in BOOTS]
    runs = [r for b in BOOTS for r in c["runs"][b]]
    out.append({"phase": phase, "concurrency": conc,
                "median_decode_tok_s": st.median(p["median_decode_tok_s"] for p in pb),
                "median_agg_tok_s": st.median(p["median_agg_tok_s"] for p in pb),
                "ttft_s_p50": st.median(p["ttft_s_p50"] for p in pb),
                "per_boot": c["per_boot"],
                "pooled": {"n": len(runs), "median_decode_tok_s": st.median(x for r in runs for x in r[1]),
                           "median_agg_tok_s": st.median(r[0] for r in runs),
                           "ttft_s_p50": st.median(x for r in runs for x in r[2])}})
print(f"two boots {', '.join(BOOTS)}, each: {headers.pop()} (per boot and cell); "
      "row = median of the per-boot medians, pooled = median over both boots' runs")
print("SUMMARY", json.dumps(out, indent=2))
