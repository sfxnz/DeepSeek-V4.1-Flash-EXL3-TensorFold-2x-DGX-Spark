#!/usr/bin/env python3
"""TF tools/bench_openai.py at temperature 0 on its chat prompt only. Its raw 'fibonacci-raw' completion
prompt makes this model emit EOS (id 1) greedily; with ignore_eos every token is id 1, no text streams and
the tool crashes (first is None). Same stream() and settings otherwise; seeds 1234+i; 1 warm-up."""
import json, statistics, sys
sys.path.insert(0, "/home/sfxnz/projects/code/TensorFold/tools")
import bench_openai as b
item = [p for p in b.PROMPTS if p["name"] == "gpu-chat-no-think"][0]
base, model, out = "http://127.0.0.1:8000", "deepseek-ai/DeepSeek-V4.1-Flash", sys.argv[1]
b.stream(base, model, item, 64, 0.0, 1234)
runs = [b.stream(base, model, item, 64, 0.0, 1234 + i) for i in range(5)]
tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
row = {"label": "", "prompt": item["name"], "temperature": 0.0, "tokens": 64,
       "decode_tps_median": statistics.median(tps), "decode_tps_all": [round(x, 2) for x in tps],
       "ttft_s_median": statistics.median(r["ttft_s"] for r in runs), "sample": runs[0]["text"][:160],
       "texts_identical": len({r["text"] for r in runs}) == 1}
print(json.dumps(row)); json.dump([row], open(out, "w"), indent=1)
