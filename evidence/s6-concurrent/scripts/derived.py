"""s6 (N12b): the README's concurrency comparison, TensorFold (B boots, decode.txt) against the vLLM sibling booted the
same session (7V, MAX_NUM_SEQS=2), cell by cell at c = 1, 2, 4: per-stream decode, aggregate, TTFT p50 and the
ruler's inter-chunk p50 / p90 (the gap between streamed chunks). Each number is read from a file here; ratios are
TF / vLLM. The other N11 numbers the README cites (the ABBA gate, the A arm, exactness, the TTFT cells, the needle,
memory, the startup estimate) are in abba_table.txt (scripts/table.py, N11's).

  python3 scripts/derived.py > derived.txt        (run from evidence/s6-concurrent)
"""
import json
import statistics as st

PHASE = {"nofields": "sampled (no field)", "top_p095": "sampled (top_p 0.95)"}


def summary(path):
    text = open(path).read().split("SUMMARY ", 1)[1]
    return json.loads(text.splitlines()[0] if text.lstrip().startswith("[{") else text)


tf = {(r["phase"], r["concurrency"]): r for r in summary("decode.txt")}
vl = {}
for f in ("7V/bench-frozen.out", "7V/bench-prose-long.out", "7V/sampled-nofields.out", "7V/sampled-top_p095.out"):
    for r in summary(f):
        vl[(PHASE.get(r["phase"], r["phase"]), r["concurrency"])] = r | {"file": f}


def ic(r, k):
    vals = [p[k] for p in r.get("per_boot", {}).values() if p.get(k) is not None] if "per_boot" in r else \
        ([r[k]] if r.get(k) is not None else [])
    return f"{st.median(vals):.0f}" if vals else "-"


print("# TF = median of the B boots' per-boot medians (decode.txt); vLLM = 7V, one boot. tok/s; TTFT s; inter-chunk ms.")
print(f"{'cell':22} {'c':>2} | {'TF dec':>7} {'vLLM dec':>8} {'ratio':>6} | {'TF agg':>7} {'vLLM agg':>8} {'ratio':>6} | "
      f"{'TF ttft':>7} {'vLLM ttft':>9} | {'TF ic p50/p90':>13} {'vLLM ic p50/p90':>15}")
for key in tf:
    t, v = tf[key], vl.get(key)
    if v is None:
        continue
    print(f"{key[0]:22} {key[1]:>2} | {t['median_decode_tok_s']:7.2f} {v['median_decode_tok_s']:8.2f} "
          f"{t['median_decode_tok_s'] / v['median_decode_tok_s']:6.3f} | {t['median_agg_tok_s']:7.2f} "
          f"{v['median_agg_tok_s']:8.2f} {t['median_agg_tok_s'] / v['median_agg_tok_s']:6.3f} | "
          f"{t['ttft_s_p50']:7.3f} {v['ttft_s_p50']:9.3f} | {ic(t, 'inter_chunk_ms_p50') + '/' + ic(t, 'inter_chunk_ms_p90'):>13} "
          f"{ic(v, 'inter_chunk_ms_p50') + '/' + ic(v, 'inter_chunk_ms_p90'):>15}")
print("files: decode.txt (2B, 3B gate/bench-*.out, sampled-*.txt); 7V/bench-frozen.out, 7V/bench-prose-long.out, "
      "7V/sampled-*.out. vLLM serves 2 sequences (MAX_NUM_SEQS=2): its c=4 waves queue 2 of 4 streams.")
print("aggregate gain over c=1, TF: " + "; ".join(
    f"{p} c={c} x{tf[(p, c)]['median_agg_tok_s'] / tf[(p, 1)]['median_agg_tok_s']:.2f}"
    for p in ("prose", "structured", "prose_long") for c in (2, 4)))
