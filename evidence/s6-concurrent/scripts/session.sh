#!/usr/bin/env bash
# N11 session after the warm-up boots W1 (PARALLEL=4) and W2 (PARALLEL=1): ABBA (A = PARALLEL=1, B = PARALLEL=4,
# CONTEXT 1048576 and the recipe defaults), then the decode-share boots, then the vLLM sibling at MAX_NUM_SEQS=2.
#   1A  PARALLEL=1                 plan A
#   2B  PARALLEL=4 share 0.5       plan Bneedle (B + the 1M needle while 3 lanes decode)
#   3B  PARALLEL=4 share 0.5       plan B
#   4A  PARALLEL=1                 plan A
#   5C  PARALLEL=4 share 0.25      plan S (TTFT cells, pairs_concurrent)
#   6D  PARALLEL=4 share 1         plan S
#   7V  vLLM (scripts/vllm.sh)
# A failed boot is recorded and the session goes on.
set -uo pipefail
SC=/home/sfxnz/projects/data/tf-dsv41/receipts/n11/scripts
for spec in "1A 1 0.5 A" "2B 4 0.5 Bneedle" "3B 4 0.5 B" "4A 1 0.5 A" "5C 4 0.25 S" "6D 4 1 S"; do
  read -r tag p share plan <<<"$spec"
  bash "$SC/boot.sh" "$tag" "$p" "$share" "$plan"
  echo "$(date -u +%FT%TZ) boot.sh $tag rc=$?"
done
bash "$SC/vllm.sh" 7V
echo "$(date -u +%FT%TZ) vllm.sh 7V rc=$?"
echo "$(date -u +%FT%TZ) session done"
