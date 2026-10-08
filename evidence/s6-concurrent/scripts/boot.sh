#!/usr/bin/env bash
# N11: one boot through the recipe's ./run.sh (image tf-dsv41-flash:0.6.4-b86514a, recipe b55aac8), its cells, then
# ./stop.sh. Receipts under receipts/n11/TAG.
#
#   boot.sh TAG PARALLEL DECODE_SHARE PLAN
#
# Every boot: free -h both nodes -> memwatch.sh (both nodes, every 2 s, until the stop) -> env PARALLEL DECODE_SHARE
# EXTRA_ENV=PYTHONPATH=/cache/tf/n11probe ./run.sh (the probe prints memory around graph warm, scripts/n11probe) ->
# startup lines -> a discarded warm pass (the ruler's prose at c=1, 2 runs) -> PLAN's cells -> both ranks' logs ->
# free -h -> ./stop.sh -> free -h.
# PLAN:
#   A        tools/session_gate.sh TAG/gate (RUNS=9 RUNS_LONG=9) -> sampled cells c=1 (nofields; t=1.0 top_p 0.95)
#            -> pairs_nucleus.py (s5's) and q2_receipts.py pairs (c7's) -> ttft.py --mode alone -> quiet prefill_cold
#   B        the same, sampled cells at c=1 2 4, ttft.py --mode cells instead of alone
#   Bneedle  B, then needle_lanes.py (the 1M needle while 3 lanes decode)
#   S        ttft.py --mode cells -> tools/pairs_concurrent.py (the decode-share boots)
#   W        warm-up and dry run, not a receipt: gate RUNS=1, sampled c=1 2 4 one run, both pairs, ttft alone
#            (PARALLEL=1) or cells (otherwise) with one rep, prefill_cold
set -uo pipefail
N11=/home/sfxnz/projects/data/tf-dsv41/receipts/n11
SC=$N11/scripts
RECIPE=/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks
TAG=$1 P=$2 SHARE=$3 PLAN=$4
EV=$N11/$TAG
BASE=http://127.0.0.1:8000
URL=$BASE/v1/chat/completions
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
TF=/home/sfxnz/projects/code/TensorFold
PY=/home/sfxnz/projects/code/.venv-tf/bin/python
TF_SHA=b86514a5ac8700b32e7b24f1095349df4ce2b922
RECIPE_SHA=b55aac8
PROMPTS=/home/sfxnz/projects/data/tf-dsv41/receipts/final/prompts.json
S2PAIRS=$RECIPE/evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl
cd "$RECIPE" || exit 2
[[ "$(git -C "$TF" rev-parse HEAD)" == "$TF_SHA" && -z "$(git -C "$TF" status --porcelain)" ]] ||
  { echo "host checkout $TF is not a clean $TF_SHA" >&2; exit 2; }
[[ "$(git rev-parse --short=7 HEAD)" == "$RECIPE_SHA" && -z "$(git status --porcelain)" ]] ||
  { echo "recipe is not a clean $RECIPE_SHA" >&2; exit 2; }
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { stamp; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }
mark() { echo "$(stamp) $TAG $*" | tee -a "$EV/steps.txt" >>"$N11/session.log"; }
run() { local out="$1"; shift; { echo "# $(stamp) $*"; "$@"; echo "rc=$?"; } >"$EV/$out" 2>&1; }

{ echo "TAG=$TAG PARALLEL=$P DECODE_SHARE=$SHARE PLAN=$PLAN"; git rev-parse HEAD; git -C "$TF" rev-parse HEAD
  sha256sum "$SC"/*.py "$SC"/*.sh "$SC"/n11probe/usercustomize.py; } >"$EV/boot-info.txt" 2>&1
free_both free-preboot.txt
bash "$SC/memwatch.sh" "$EV/memwatch.tsv" spark2 2 &
MW=$!
trap 'kill $MW 2>/dev/null' EXIT
mark "run.sh PARALLEL=$P DECODE_SHARE=$SHARE"
{ echo "start $(stamp)"
  echo "\$ env PARALLEL=$P DECODE_SHARE=$SHARE EXTRA_ENV=PYTHONPATH=/cache/tf/n11probe ./run.sh"
  env PARALLEL="$P" DECODE_SHARE="$SHARE" EXTRA_ENV=PYTHONPATH=/cache/tf/n11probe ./run.sh
  echo "rc=$?"; echo "end $(stamp)"; } >"$EV/run.txt" 2>&1
startup() {
  docker logs tf-dsv41-flash 2>&1 | grep -E '^\[n11probe\]|^\[tensorfold\] (CUDA rank|rank [01] of 2|serving|other conv)' \
    | sed 's/^/rank0 /' >"$EV/startup.txt"
  ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash 2>&1 | grep -E '^\[n11probe\]|^\[tensorfold\] (CUDA rank|rank [01] of 2|rank 1)' \
    | sed 's/^/rank1 /' >>"$EV/startup.txt"
}
if ! grep -q '^rc=0$' "$EV/run.txt"; then
  mark "run.sh FAILED"
  startup
  { echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
  exit 1
fi
startup
free_both free-serving.txt
docker image inspect -f '{{.Id}} {{index .Config.Labels "tensorfold.sha"}}' tf-dsv41-flash:0.6.4-b86514a >"$EV/image.txt" 2>&1
docker inspect -f '{{.Config.Image}} {{join .Args " "}}' tf-dsv41-flash >"$EV/serve-argv.txt" 2>&1
ssh -o BatchMode=yes spark2 "docker inspect -f '{{.Config.Image}} {{join .Args \" \"}}' tf-dsv41-flash" >>"$EV/serve-argv.txt" 2>&1

mark "warm pass (discarded)"
run warm-pass.txt python3 bench_decode.py --url "$URL" --model "$MODEL" --phase prose --concurrency 1 --max-tokens 200 --runs 2

gate() { mark "gate RUNS=$1"; RUNS=$1 RUNS_LONG=$1 tools/session_gate.sh "$EV/gate" >"$EV/gate-stdout.txt" 2>&1; }
sampled() {
  mark "sampled cells c=$*"
  run sampled-nofields.txt python3 "$SC/sampled_conc.py" --url "$URL" --label nofields --runs "$RUNS_S" --concurrency "$@"
  run sampled-top_p095.txt python3 "$SC/sampled_conc.py" --url "$URL" --label top_p095 --runs "$RUNS_S" --temperature 1.0 \
    --top-p 0.95 --concurrency "$@"
}
pairs() {
  mark "pairs (s5 pairs_nucleus, c7 q2 receipts)"
  run pairs_nucleus.jsonl python3 "$SC/pairs_nucleus.py" "$BASE" "$MODEL" "$S2PAIRS"
  run pairs_q2.jsonl python3 "$SC/q2_receipts.py" "$BASE" "$MODEL" "$PROMPTS" pairs
}
ttft() { mark "ttft $1 reps=$2"; run "ttft_$1.log" python3 "$SC/ttft.py" --mode "$1" --reps "$2" --out "$EV/ttft_$1.jsonl"; }
prefill_cold() {
  mark "prefill_cold"
  { stamp; ss -tnp 2>/dev/null | grep ':8000 '; nvidia-smi --query-compute-apps=pid,name --format=csv; } \
    >"$EV/prefill-clients.txt" 2>&1
  { echo "# tools/prefill_cold.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, prompts $PROMPTS sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1), $(stamp)"
    (cd "$TF" && "$PY" tools/prefill_cold.py run "$BASE" "$MODEL" "$PROMPTS" "$EV/prefill_cold.json")
    echo "rc=$?"; stamp; } >"$EV/prefill_cold.log" 2>&1
}

RUNS_S=9
case "$PLAN" in
  A) gate 9; sampled 1; pairs; ttft alone 3; prefill_cold ;;
  B|Bneedle)
    gate 9; sampled 1 2 4; pairs; ttft cells 3; prefill_cold
    if [[ "$PLAN" == Bneedle ]]; then
      mark "needle 1040000@0.5 with 3 lanes decoding"
      free_both free-before-needle.txt
      run needle_lanes.log python3 "$SC/needle_lanes.py" "$EV/needle_lanes.jsonl"
      free_both free-after-needle.txt
    fi ;;
  S) ttft cells 3
     mark "pairs_concurrent"
     run pairs_concurrent.log python3 tools/pairs_concurrent.py --url "$URL" --model "$MODEL" --output "$EV/pairs_concurrent.jsonl" ;;
  W) RUNS_S=1; gate 1; sampled 1 2 4; pairs
     if [[ "$P" == 1 ]]; then ttft alone 1; else ttft cells 1; fi
     prefill_cold ;;
  *) echo "unknown PLAN $PLAN" >&2 ;;
esac

mark "logs and stop"
docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
free_both free-end.txt
{ echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
sleep 10
free_both free-stopped.txt
mark "done"
