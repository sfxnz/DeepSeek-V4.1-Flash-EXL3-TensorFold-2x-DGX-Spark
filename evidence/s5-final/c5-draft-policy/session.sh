#!/usr/bin/env bash
# Dev only (outside the PR, C5): exactness, decode and prompt receipts against the two-rank server on 127.0.0.1:8080,
# then (POLICIES set) the d=3 vs default policy sweep, ABBA within the boot (sweep passes run the policies forward then
# reversed). Outputs under receipts/c5/TAG.
# usage: [POLICIES=3,c0.15@5] session.sh TAG WT
set -uo pipefail
TAG=$1 WT=$2
U=http://127.0.0.1:8080 MODEL=deepseek-ai/DeepSeek-V4.1-Flash
D=/home/sfxnz/projects/data/tf-dsv41/dev R=/home/sfxnz/projects/data/tf-dsv41/receipts
REC=/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
OUT=$R/c5/$TAG PROMPTS=$R/final/prompts.json
mkdir -p "$OUT"
PY=/home/sfxnz/projects/code/.venv-tf/bin/python
cd "$WT"
git -C "$WT" rev-parse HEAD > "$OUT/commit.txt"
date -u +%FT%TZ > "$OUT/started.txt"
step() { echo "== $TAG $1 $(date -u +%T)"; }
rm -f $R/c5/ctl/policy.json
step bench_concurrent
$PY tools/bench_concurrent.py $U $MODEL --alone --serial --label "$TAG" --output "$OUT/bench_concurrent.json" \
  > "$OUT/bench_concurrent.log" 2>&1; echo "rc=$?" >> "$OUT/bench_concurrent.log"
step pairs
$PY $D/q2/receipts.py $U $MODEL $PROMPTS pairs > "$OUT/pairs.jsonl" 2>&1
step bench_openai
$PY tools/bench_openai.py $U $MODEL --label "$TAG" --output "$OUT/bench_openai.json" > "$OUT/bench_openai.log" 2>&1
echo "rc=$?" >> "$OUT/bench_openai.log"
step bench_decode
$PY $REC/bench_decode.py --url $U/v1/chat/completions --model $MODEL --concurrency 1 --phase all \
  > "$OUT/bench_decode.log" 2>&1; echo "rc=$?" >> "$OUT/bench_decode.log"
step lail
$PY $REC/tools/measure_lail_prose.py --url $U/v1/chat/completions --model $MODEL --runs 3 > "$OUT/lail.log" 2>&1
echo "rc=$?" >> "$OUT/lail.log"
step prefill_cold; $D/c4/evict_both.sh > "$OUT/evict_prefill.jsonl" 2>&1
$PY tools/prefill_cold.py run $U $MODEL $PROMPTS "$OUT/prefill_cold.json" > "$OUT/prefill_cold.log" 2>&1
echo "rc=$?" >> "$OUT/prefill_cold.log"
if [ -n "${POLICIES:-}" ]; then
  step sweep
  $PY $D/c5/sweep.py $U $MODEL "$POLICIES" 2 > "$OUT/sweep.jsonl" 2> "$OUT/sweep.err"
  rm -f $R/c5/ctl/policy.json
fi
step done
date -u +%FT%TZ > "$OUT/finished.txt"
