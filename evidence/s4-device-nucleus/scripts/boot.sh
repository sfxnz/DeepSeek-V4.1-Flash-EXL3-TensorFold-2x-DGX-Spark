#!/usr/bin/env bash
# s4: one boot at the shipped defaults (no env override) on TF_SHA 41306d5, its gate and cells, then a graceful stop.
#
#   evidence/s4-device-nucleus/scripts/boot.sh      (from the recipe root)
#
# Order: free -h both nodes -> ./run.sh -> tools/session_gate.sh gate (RUNS=9 RUNS_LONG=9, as s3) -> scripts/sampled_cell.py
# (s2's requests, plus token_sha) on the ruler's prose prompt, 9 runs each: no sampling field, temperature 1.0 + top_p
# 1.0, temperature 1.0 + top_k 20, temperature 1.0 + top_p 0.95, temperature 1.0 + top_p 0.9 -> scripts/pairs_nucleus.py
# -> both ranks' full logs -> free -h -> ./stop.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 2
EV="evidence/s4-device-nucleus"
S2="evidence/s2-run-sh-ec28f35"
URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
stamp() { date -u +%FT%TZ; }
free_both() { { free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }

free_both free-preboot.txt
{ echo "start $(stamp)"; echo '$ ./run.sh   (defaults, no env overrides)'; ./run.sh; echo "rc=$?"; echo "end $(stamp)"; } \
  >"$EV/run.txt" 2>&1
grep -q '^rc=0$' "$EV/run.txt" || { echo "run.sh failed, see $EV/run.txt" >&2; ./stop.sh >"$EV/stop.txt" 2>&1; exit 1; }

RUNS=9 RUNS_LONG=9 tools/session_gate.sh "$EV/gate" >"$EV/gate-stdout.txt" 2>&1

cell() {
  local label="$1"; shift
  { python3 "$EV/scripts/sampled_cell.py" --label "$label" --runs 9 "$@"; echo "rc=$?"; } >"$EV/sampled-$label.txt" 2>&1
}
cell nofields
cell top_p1 --temperature 1.0 --top-p 1.0
cell top_k20 --temperature 1.0 --top-k 20
cell top_p095 --temperature 1.0 --top-p 0.95
cell top_p09 --temperature 1.0 --top-p 0.9

{ python3 "$EV/scripts/pairs_nucleus.py" "$URL" "$MODEL" "$S2/bootA/pairs.jsonl"; echo "rc=$?"; } >"$EV/pairs.jsonl" 2>&1

docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
free_both free-end.txt
{ echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
echo "s4 done $(stamp)"
