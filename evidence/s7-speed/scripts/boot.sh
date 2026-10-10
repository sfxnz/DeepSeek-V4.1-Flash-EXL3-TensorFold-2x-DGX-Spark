#!/usr/bin/env bash
# s7: one boot through an arm's own ./run.sh at the shipped defaults (no env knobs, no debug env), its cells, then
# that arm's ./stop.sh. Receipts under evidence/s7-speed/TAG.
#
#   boot.sh TAG ARM PLAN
#
# ARM:
#   A  old: TensorFold b86514a, the recipe at origin/main 38a0607 (worktree /home/sfxnz/projects/data/tf-dsv41/s7-old-recipe)
#   B  new: TensorFold 19f5478, this branch (agent/tensorfold-dsv41-speed, 3bfdd72)
# PLAN:
#   W        warm-up, not counted: gate RUNS=1, diverse 1 wave, sampled 1 run at c=1 2 4, prefill_cold
#   C        gate RUNS=9 RUNS_LONG=9 (that arm's tools/session_gate.sh) -> diverse c=1 2 4, 9 waves -> sampled cells
#            (no field; temperature 1.0 top_p 0.95) at c=1 2 4, 9 runs -> prefill_cold (2k..64k, 3 prompts a length)
#   Cneedle  C, then needle_lanes.py (the 1,039,528-token needle while 3 lanes decode)
# Every boot: free -h both nodes -> memwatch.sh (both nodes, every 2 s, until the stop) -> ./run.sh -> startup lines
# -> a discarded warm pass (the ruler's prose at c=1, 2 runs) -> PLAN's cells -> both ranks' logs, rank 0's done lines
# -> free -h -> ./stop.sh -> free -h.
set -uo pipefail
NEW=/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks
OLD=/home/sfxnz/projects/data/tf-dsv41/s7-old-recipe
S7=$NEW/evidence/s7-speed
SC=$S7/scripts
TAG=$1 ARM=$2 PLAN=$3
EV=$S7/$TAG
BASE=http://127.0.0.1:8000
URL=$BASE/v1/chat/completions
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
PROMPTS=/home/sfxnz/projects/data/tf-dsv41/receipts/final/prompts.json
case "$ARM" in
  A) RECIPE=$OLD RECIPE_SHA=38a0607 IMG=tf-dsv41-flash:0.6.4-b86514a TF_SHA=b86514a5ac8700b32e7b24f1095349df4ce2b922 ;;
  B) RECIPE=$NEW RECIPE_SHA=3bfdd72 IMG=tf-dsv41-flash:0.6.6-19f5478 TF_SHA=19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb ;;
  *) echo "ARM must be A or B" >&2; exit 2 ;;
esac
cd "$RECIPE" || exit 2
# Tracked files only: this boot's receipts are untracked files under evidence/.
[[ "$(git rev-parse --short=7 HEAD)" == "$RECIPE_SHA" ]] && git diff --quiet HEAD ||
  { echo "$RECIPE is not a clean $RECIPE_SHA" >&2; exit 2; }
grep -q "^IMAGE=\"\${IMAGE:-$IMG}\"" run.sh && grep -q "^TF_SHA=\"\${TF_SHA:-$TF_SHA}\"" run.sh ||
  { echo "$RECIPE/run.sh does not pin $IMG / $TF_SHA" >&2; exit 2; }
# No knob or debug env reaches run.sh: the shipped defaults only.
knobs="$(env | grep -E '^(TF_|TORCH|TRITON|CUDA_|NCCL|EXTRA_ENV|PARALLEL|CONTEXT|DECODE_SHARE|MTP_|TOP_P|IMAGE|MEM_GATE|FORCE_|PYTHONPATH)' || true)"
[[ -z "$knobs" ]] || { echo "refusing: knob env set: $knobs" >&2; exit 2; }
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { stamp; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }
mark() { echo "$(stamp) $TAG $*" | tee -a "$EV/steps.txt" >>"$S7/session.log"; }
run() { local out="$1"; shift; { echo "# $(stamp) $*"; "$@"; echo "rc=$?"; } >"$EV/$out" 2>&1; }

{ echo "TAG=$TAG ARM=$ARM PLAN=$PLAN RECIPE=$RECIPE IMAGE=$IMG TF_SHA=$TF_SHA"; git rev-parse HEAD
  echo "env knobs: none"; sha256sum "$SC"/*.py "$SC"/*.sh; } >"$EV/boot-info.txt" 2>&1
free_both free-preboot.txt
bash "$SC/memwatch.sh" "$EV/memwatch.tsv" spark2 2 &
MW=$!
trap 'kill $MW 2>/dev/null' EXIT
mark "run.sh arm $ARM ($RECIPE)"
{ echo "start $(stamp)"; echo "\$ ./run.sh   (cwd $RECIPE)"; ./run.sh; echo "rc=$?"; echo "end $(stamp)"; } >"$EV/run.txt" 2>&1
startup() {
  docker logs tf-dsv41-flash 2>&1 | grep -E '^\[tensorfold\] (CUDA rank|rank [01] of 2|serving|other conv)' \
    | sed 's/^/rank0 /' >"$EV/startup.txt"
  ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash 2>&1 | grep -E '^\[tensorfold\] (CUDA rank|rank [01] of 2|rank 1)' \
    | sed 's/^/rank1 /' >>"$EV/startup.txt"
}
if ! grep -q '^rc=0$' "$EV/run.txt"; then
  mark "run.sh FAILED"
  startup
  docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
  ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
  { echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
  exit 1
fi
startup
free_both free-serving.txt
{ docker image inspect -f '{{.Id}} {{index .Config.Labels "tensorfold.sha"}}' "$IMG"
  ssh -o BatchMode=yes spark2 "docker image inspect -f '{{.Id}} {{index .Config.Labels \"tensorfold.sha\"}}' $IMG"
  docker inspect -f '{{.Image}} {{.Config.Image}}' tf-dsv41-flash
  ssh -o BatchMode=yes spark2 "docker inspect -f '{{.Image}} {{.Config.Image}}' tf-dsv41-flash"; } >"$EV/image.txt" 2>&1
docker inspect -f '{{.Config.Image}} {{join .Args " "}}' tf-dsv41-flash >"$EV/serve-argv.txt" 2>&1
ssh -o BatchMode=yes spark2 "docker inspect -f '{{.Config.Image}} {{join .Args \" \"}}' tf-dsv41-flash" >>"$EV/serve-argv.txt" 2>&1

mark "warm pass (discarded)"
run warm-pass.txt python3 bench_decode.py --url "$URL" --model "$MODEL" --phase prose --concurrency 1 --max-tokens 200 --runs 2

gate() { mark "gate RUNS=$1"; RUNS=$1 RUNS_LONG=$1 tools/session_gate.sh "$EV/gate" >"$EV/gate-stdout.txt" 2>&1; echo "gate rc=$?" >>"$EV/gate-stdout.txt"; }
diverse() { mark "diverse c=1 2 4 runs=$1"; run diverse.txt python3 "$SC/bench_diverse.py" --url "$URL" --model "$MODEL" --phase both --concurrency 1 2 4 --runs "$1" --max-tokens 200; }
sampled() {
  mark "sampled cells c=1 2 4 runs=$1"
  run sampled-nofields.txt python3 "$SC/sampled_conc.py" --url "$URL" --label nofields --runs "$1" --concurrency 1 2 4
  run sampled-top_p095.txt python3 "$SC/sampled_conc.py" --url "$URL" --label top_p095 --runs "$1" --temperature 1.0 \
    --top-p 0.95 --concurrency 1 2 4
}
prefill_cold() {
  mark "prefill_cold"
  { stamp; ss -tnp 2>/dev/null | grep ':8000 '; } >"$EV/prefill-clients.txt" 2>&1
  { echo "# scripts/prefill_cold.py (TensorFold tools/prefill_cold.py, same bytes at b86514a and 19f5478), prompts $PROMPTS sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1), $(stamp)"
    python3 "$SC/prefill_cold.py" run "$BASE" "$MODEL" "$PROMPTS" "$EV/prefill_cold.json"
    echo "rc=$?"; stamp; } >"$EV/prefill_cold.log" 2>&1
}

case "$PLAN" in
  W) gate 1; diverse 1; sampled 1; prefill_cold ;;
  C|Cneedle)
    gate 9; diverse 9; sampled 9; prefill_cold
    if [[ "$PLAN" == Cneedle ]]; then
      mark "needle 1040000@0.5 with 3 lanes decoding"
      free_both free-before-needle.txt
      run needle_lanes.log python3 "$SC/needle_lanes.py" "$EV/needle_lanes.jsonl"
      free_both free-after-needle.txt
    fi ;;
  *) echo "unknown PLAN $PLAN" >&2 ;;
esac

mark "logs and stop"
docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
grep -E '^\[tensorfold\] done req-' "$EV/rank0-full.log" >"$EV/done-rank0.txt"
free_both free-end.txt
{ echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
sleep 10
free_both free-stopped.txt
mark "done"
