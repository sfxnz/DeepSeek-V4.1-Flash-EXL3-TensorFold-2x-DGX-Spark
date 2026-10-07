#!/usr/bin/env bash
# s5: one boot at the shipped defaults (no env override) on TF_SHA 903a1e8, its cells, then a graceful stop.
#
#   evidence/s5-final/scripts/boot.sh BOOT [long]      (from the recipe root; BOOT is bootE or bootF)
#
# Order: free -h both nodes -> (long: scripts/memwatch.sh, both nodes every 5 s, until the stop) -> ./run.sh ->
# tools/session_gate.sh BOOT/gate (RUNS=9 RUNS_LONG=9: the frozen ruler's prose, structured and prose_long at c=1) ->
# L.A.I.L (the vLLM recipe's tools/measure_lail_prose.py, unchanged: 512 tokens, temperature 0.2, --runs 10 as its
# ARMS.md fresh-boot order) -> sampled cells (scripts/sampled_cell.py, s4's, 9 runs each: no sampling field;
# temperature 1.0 + top_p 0.95; temperature 1.0 + top_k 20) -> the engine's tools/bench_openai.py (64 tokens, 5 reps,
# t=1.0 and 0) and scripts/bench_t0_chat.py (the goldens' t=0 chat-only twin, also run on vLLM) -> pairs
# (scripts/pairs_nucleus.py, s4's) -> the engine's tools/bench_concurrent.py --alone --serial -> quiet
# tools/prefill_cold.py (s2's prompts) -> long only: the ~1,040,000-token needle (scripts/long_needle.py, s2's, the
# same length and depth as s2's), then the vLLM recipe's quality_eval.py --full through the lab's qe_tf.py (selfcons,
# gsm8k, gsm8k_think, mmlu, tools, needle; round-36 full baseline) -> both ranks' full logs -> free -h -> ./stop.sh.
# The engine's tools run from the host checkout at TF_SHA in the lab's host venv (the image does not ship tools/).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 2
B="$1"
S5="evidence/s5-final"
SC="$S5/scripts"
EV="$S5/$B"
S2="evidence/s2-run-sh-ec28f35"
URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
TF_SHA=903a1e8af62c8f46eceee6b95481706ada30ae49
TF=/home/sfxnz/projects/code/TensorFold
PY=/home/sfxnz/projects/code/.venv-tf/bin/python
VLLM=/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
QE=/home/sfxnz/projects/data/tf-dsv41/dev/q3/qe_tf.py
PROMPTS=/home/sfxnz/projects/data/tf-dsv41/receipts/final/prompts.json
[[ "$(git -C "$TF" rev-parse HEAD)" == "$TF_SHA" && -z "$(git -C "$TF" status --porcelain)" ]] ||
  { echo "host checkout $TF is not a clean $TF_SHA" >&2; exit 2; }
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { stamp; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }
mark() { echo "$(stamp) $*" | tee -a "$EV/steps.txt"; }

free_both free-preboot.txt
MW=""
if [[ "${2:-}" == long ]]; then
  bash "$SC/memwatch.sh" "$EV/memwatch.tsv" spark2 &
  MW=$!
fi
mark "run.sh"
{ echo "start $(stamp)"; echo '$ ./run.sh   (defaults, no env overrides)'; ./run.sh; echo "rc=$?"; echo "end $(stamp)"; } \
  >"$EV/run.txt" 2>&1
if ! grep -q '^rc=0$' "$EV/run.txt"; then
  echo "run.sh failed, see $EV/run.txt" >&2; ./stop.sh >"$EV/stop.txt" 2>&1; [[ -n "$MW" ]] && kill "$MW"; exit 1
fi

mark "gate"
RUNS=9 RUNS_LONG=9 tools/session_gate.sh "$EV/gate" >"$EV/gate-stdout.txt" 2>&1

mark "lail"
{ python3 "$VLLM/tools/measure_lail_prose.py" --url "$URL/v1/chat/completions" --model "$MODEL" --runs 10; echo "rc=$?"; } \
  >"$EV/lail.log" 2>&1

cell() {
  local label="$1"; shift
  { python3 "$SC/sampled_cell.py" --label "$label" --runs 9 "$@"; echo "rc=$?"; } >"$EV/sampled-$label.txt" 2>&1
}
mark "sampled cells"
cell nofields
cell top_p095 --temperature 1.0 --top-p 0.95
cell top_k20 --temperature 1.0 --top-k 20

mark "bench_openai"
{
  echo "# tools/bench_openai.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, $(stamp)"
  (cd "$TF" && "$PY" tools/bench_openai.py "$URL" "$MODEL" --tokens 64 --reps 5 --temperatures 1.0,0 --label "s5-$B" \
     --output "$ROOT/$EV/bench_openai.json")
  echo "rc=$?"
} >"$EV/bench_openai.log" 2>&1
{ "$PY" "$SC/bench_t0_chat.py" "$EV/bench_t0_chat.json"; echo "rc=$?"; } >"$EV/bench_t0_chat.log" 2>&1

mark "pairs"
{ python3 "$SC/pairs_nucleus.py" "$URL" "$MODEL" "$S2/bootA/pairs.jsonl"; echo "rc=$?"; } >"$EV/pairs.jsonl" 2>&1

mark "bench_concurrent"
{
  echo "# tools/bench_concurrent.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, $(stamp)"
  (cd "$TF" && "$PY" tools/bench_concurrent.py "$URL" "$MODEL" --alone --serial --label "s5-$B" \
     --output "$ROOT/$EV/bench_concurrent.json")
  echo "rc=$?"
} >"$EV/bench_concurrent.log" 2>&1

mark "prefill_cold"
# Who holds a connection to the API, and the GPU processes on the head, just before the quiet run.
{ stamp; ss -tnp 2>/dev/null | grep ':8000 '; nvidia-smi --query-compute-apps=pid,name --format=csv; } \
  >"$EV/prefill-clients.txt" 2>&1
{
  echo "# tools/prefill_cold.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, prompts $PROMPTS sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1), $(stamp)"
  (cd "$TF" && "$PY" tools/prefill_cold.py run "$URL" "$MODEL" "$PROMPTS" "$ROOT/$EV/prefill_cold.json")
  echo "rc=$?"
  stamp
} >"$EV/prefill_cold.log" 2>&1

if [[ "${2:-}" == long ]]; then
  mark "needle 1040000@0.5"
  free_both free-before-needle.txt
  { python3 "$SC/long_needle.py" 1040000 0.5; echo "rc=$?"; } >"$EV/needle.jsonl" 2>"$EV/needle.err"
  free_both free-after-needle.txt
  mark "quality full"
  { python3 "$QE" --full --only selfcons,gsm8k,gsm8k_think,mmlu,tools,needle \
      --baseline "$VLLM/results/2026-09-27-viterbi-adopt/quality-baseline/full.json" --out "$EV/quality_full.json"
    echo "rc=$?"; } >"$EV/quality_full.txt" 2>&1
fi

mark "logs and stop"
docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
free_both free-end.txt
{ echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
[[ -n "$MW" ]] && kill "$MW"
mark "$B done"
