#!/usr/bin/env bash
# s3: one boot at the shipped defaults (no env override), its gate and cells, then a graceful stop.
#
#   evidence/s3-default-1m/scripts/boot.sh BOOT [prefill]      (from the recipe root; BOOT is bootC or bootD)
#
# Order: free -h both nodes -> ./run.sh -> tools/session_gate.sh BOOT/gate (RUNS=9 RUNS_LONG=9, as s2) -> s2's
# sampled_cell.py on the ruler's prose prompt, 9 runs each: no sampling field, temperature 1.0 + top_p 1.0,
# temperature 1.0 + top_k 20, temperature 1.0 + top_p 0.95 -> pairs_default.py -> with `prefill`: the engine's
# tools/prefill_cold.py at TF_SHA from the host checkout, s2's command and prompts -> both ranks' full logs ->
# free -h -> ./stop.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 2
B="$1"
S3="evidence/s3-default-1m"
S2="evidence/s2-run-sh-ec28f35"
EV="$S3/$B"
URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
ENGINE_WT=/home/sfxnz/projects/code/wt/recipe-engine
VENV_PY=/home/sfxnz/projects/code/.venv-tf/bin/python
PROMPTS=/home/sfxnz/projects/data/tf-dsv41/receipts/final/prompts.json
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }

free_both free-preboot.txt
{ echo "start $(stamp)"; echo '$ ./run.sh   (defaults, no env overrides)'; ./run.sh; echo "rc=$?"; echo "end $(stamp)"; } \
  >"$EV/run.txt" 2>&1
grep -q '^rc=0$' "$EV/run.txt" || { echo "run.sh failed, see $EV/run.txt" >&2; exit 1; }

RUNS=9 RUNS_LONG=9 tools/session_gate.sh "$EV/gate" >"$EV/gate-stdout.txt" 2>&1

cell() {
  local label="$1"; shift
  { python3 "$S2/scripts/sampled_cell.py" --label "$label" --runs 9 "$@"; echo "rc=$?"; } >"$EV/sampled-$label.txt" 2>&1
}
cell nofields
cell top_p1 --temperature 1.0 --top-p 1.0
cell top_k20 --temperature 1.0 --top-k 20
cell top_p095 --temperature 1.0 --top-p 0.95

{ python3 "$S3/scripts/pairs_default.py" "$URL" "$MODEL" "$S2/bootA/pairs.jsonl"; echo "rc=$?"; } >"$EV/pairs.jsonl" 2>&1

if [[ "${2:-}" == prefill ]]; then
  # Who holds a connection to the API, and the GPU processes on the head, just before the quiet run.
  { stamp; ss -tnp 2>/dev/null | grep ':8000 '; nvidia-smi --query-compute-apps=pid,name --format=csv; } \
    >"$EV/prefill-clients.txt" 2>&1
  {
    echo "# tools/prefill_cold.py at $(git -C "$ENGINE_WT" rev-parse HEAD) (worktree $ENGINE_WT), host venv, prompts $PROMPTS sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1), $(stamp)"
    (cd "$ENGINE_WT" && "$VENV_PY" tools/prefill_cold.py run "$URL" "$MODEL" "$PROMPTS" "$ROOT/$EV/prefill_cold.json")
    echo "rc=$?"
    stamp
  } >"$EV/prefill_cold.log" 2>&1
fi

docker logs tf-dsv41-flash >"$EV/rank0-full.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs tf-dsv41-flash >"$EV/rank1-full.log" 2>&1
free_both free-end.txt
{ echo "$(stamp) ./stop.sh"; ./stop.sh; echo "rc=$?"; } >"$EV/stop.txt" 2>&1
echo "$B done $(stamp)"
