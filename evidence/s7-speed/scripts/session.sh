#!/usr/bin/env bash
# s7 session: the new engine (B, TensorFold 19f5478) against the old (A, b86514a), both through their own run.sh at
# the shipped defaults. W1 (B, empty kernel cache, not counted), then ABBA 1A 2B 3B 4A, then 5V (the vLLM sibling,
# canonical `env AUDIT=strict ./run.sh`). A failed boot is recorded and the session goes on. Ends with end_state.txt.
set -uo pipefail
S7=/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks/evidence/s7-speed
SC=$S7/scripts
for spec in "W1 B W" "1A A C" "2B B Cneedle" "3B B C" "4A A C"; do
  read -r tag arm plan <<<"$spec"
  bash "$SC/boot.sh" "$tag" "$arm" "$plan"
  echo "$(date -u +%FT%TZ) boot.sh $tag rc=$?"
done
bash "$SC/vllm.sh" 5V
echo "$(date -u +%FT%TZ) vllm.sh 5V rc=$?"
{ date -u +%FT%TZ
  echo "== spark1 docker ps"; docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
  echo "== spark2 docker ps"; ssh -o BatchMode=yes spark2 "docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  echo "== spark1 GPU processes"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
  echo "== spark2 GPU processes"; ssh -o BatchMode=yes spark2 nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
  echo "== free"; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$S7/end_state.txt" 2>&1
echo "$(date -u +%FT%TZ) session done"
