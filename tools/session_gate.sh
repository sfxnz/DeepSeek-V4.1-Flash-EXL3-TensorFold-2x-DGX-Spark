#!/usr/bin/env bash
# Gate pack for one session on a live serve of this recipe.
#
#   tools/session_gate.sh EVDIR
#
# Order: receipts (sha256 of the clients, run.sh, stop.sh and recipe.yaml, git head, both ranks' argv and container env, free -h on both nodes)
# -> smokes (smoke_chat.py, smoke_count.py) -> the frozen bench_decode.py (the vLLM sibling's ruler,
# byte-identical): prose + structured at c=1, 9 runs; prose_long at c=1, 5 runs; the same at c=2 and c=4
# -> tools/bench_concurrent.py (TensorFold's, verbatim): every request alone and "draft": false, then together at
# 1, 2 and 4, and mixed prompts started 2 s apart at 2 and 4; every reply's token_sha must equal its alone run's
# -> tools/pairs_concurrent.py: distinct staggered prompts, one over 16k tokens, token_ids and token_sha a request
# -> receipts (free -h, both ranks' logs, startup lines, error needles) -> gate.txt. Exits nonzero if any step fails.
#
# Env:
#   RUN_FROZEN=0     skip the frozen bench_decode.py runs.
#   RUNS=9           runs per prose / structured cell; RUNS_LONG=5 for prose_long.
#   URL, MODEL, WORKER_HOST, CONTAINER_NAME, PORT as in run.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
EV="$(mkdir -p "$1" && cd "$1" && pwd)"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
URL="${URL:-${BASE}/v1/chat/completions}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
WORKER_HOST="${WORKER_HOST:-spark2}"
CONTAINER_NAME="${CONTAINER_NAME:-tf-dsv41-flash}"
RUN_FROZEN="${RUN_FROZEN:-1}"
RUNS="${RUNS:-9}"
RUNS_LONG="${RUNS_LONG:-5}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
cd "$ROOT" || exit 2

declare -a RESULTS=()
FAILED=0
stamp() { date -u +%FT%T.%3NZ; }
log() { echo "$(stamp) session_gate: $*" | tee -a "$EV/gate.log"; }

# step NAME CMD... : run with stdout/stderr/exit captured under EVDIR.
step() {
  local name="$1"; shift
  log "start $name"
  "$@" >"$EV/$name.out" 2>"$EV/$name.err"
  local rc=$?
  echo "$rc" >"$EV/$name.exit"
  RESULTS+=("$name=$rc")
  [[ $rc -eq 0 ]] || FAILED=1
  log "end $name rc=$rc"
  return 0
}

free_both() {
  free -h >"$EV/free-$1.txt" 2>&1
  "${SSH[@]}" "$WORKER_HOST" 'free -h' >"$EV/free-$1-${WORKER_HOST}.txt" 2>&1 || echo "ssh failed" >>"$EV/free-$1-${WORKER_HOST}.txt"
}

logs_both() {
  docker logs "$CONTAINER_NAME" 2>&1 | grep -v 'GET /v1/models\|GET /health\|GET /metrics' >"$EV/docker-head.log" || true
  "${SSH[@]}" "$WORKER_HOST" "docker logs '$CONTAINER_NAME' 2>&1" >"$EV/docker-${WORKER_HOST}.log" 2>&1 || true
  grep -E '^\[tensorfold\] (CUDA rank [01] startup estimate|rank [01] of 2:|serving|rank 1 ready)' "$EV/docker-head.log" "$EV/docker-${WORKER_HOST}.log" >"$EV/startup.txt" || true
  grep -E 'Traceback|Error|error:|NCCL WARN.*(timeout|abort)' "$EV/docker-head.log" "$EV/docker-${WORKER_HOST}.log" >"$EV/engine-needles.txt" || true
}

# --- preflight ---------------------------------------------------------------
log "evidence dir $EV"
code="$(curl -s -o "$EV/health-before.json" -w '%{http_code}' --max-time 5 "$BASE/health" || true)"
echo "$code" >"$EV/health.code"
if [[ "$code" != 200 ]]; then
  log "GET /health = ${code:-none}; refusing to run the gate on a server that is not up"
  exit 1
fi
curl -sS --max-time 5 "$BASE/v1/models" >"$EV/models.json" || true
if ! grep -q "\"$MODEL\"" "$EV/models.json"; then
  log "/v1/models does not list $MODEL"
  exit 1
fi
sha256sum bench_decode.py smoke_chat.py smoke_count.py tools/bench_concurrent.py tools/pairs_concurrent.py tools/session_gate.sh run.sh stop.sh recipe.yaml >"$EV/harness.sha256"
git rev-parse HEAD >"$EV/git-head.txt" 2>/dev/null || true
docker inspect -f '{{.Config.Image}} {{join .Args " "}}' "$CONTAINER_NAME" >"$EV/serve-argv.txt" 2>&1 || true
"${SSH[@]}" "$WORKER_HOST" "docker inspect -f '{{.Config.Image}} {{join .Args \" \"}}' '$CONTAINER_NAME'" >>"$EV/serve-argv.txt" 2>&1 || true
# The container env goes into committed evidence: blank any value whose name looks like a secret (EXTRA_ENV can carry one).
redact() { sed -E 's/^([A-Za-z0-9_]*(TOKEN|KEY|SECRET|PASS)[A-Za-z0-9_]*)=.*/\1=<redacted>/'; }
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME" 2>&1 | redact >"$EV/env-rank0.txt" || true
"${SSH[@]}" "$WORKER_HOST" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' '$CONTAINER_NAME'" 2>&1 | redact >"$EV/env-rank1.txt" || true
free_both before

# --- smokes ------------------------------------------------------------------
step smoke-chat python3 smoke_chat.py --url "$URL" --model "$MODEL"
step smoke-count python3 smoke_count.py --url "$URL" --model "$MODEL"

# --- frozen ruler ------------------------------------------------------------
if [[ "$RUN_FROZEN" == 1 ]]; then
  step bench-frozen python3 bench_decode.py --url "$URL" --model "$MODEL" --phase both --concurrency 1 --max-tokens 200 --runs "$RUNS"
  step bench-prose-long python3 bench_decode.py --url "$URL" --model "$MODEL" --phase prose_long --concurrency 1 --max-tokens 200 --runs "$RUNS_LONG"
fi

# --- concurrent requests -----------------------------------------------------
# bench_concurrent.py exits 0 whatever its checks found: conc-check fails on any unequal or failed request.
conc_check() {
  python3 - "$@" <<'PY'
import json, sys
bad = []
for path in sys.argv[1:]:
    for cell in json.load(open(path))["cells"]:
        for key in ("alone", "serial"):
            got = cell.get(key) or {}
            if got.get("unequal") or got.get("failed"):
                bad.append(f"{path} {cell['prompt']} t={cell['temperature']} n={cell['streams']} {key} {got}")
        if cell.get("failed") or "alone" not in cell:
            bad.append(f"{path} {cell['prompt']} t={cell['temperature']} n={cell['streams']} failed={cell.get('failed')} checked={'alone' in cell}")
print("\n".join(bad) or "every request equal to its alone run")
sys.exit(1 if bad else 0)
PY
}
if [[ "$RUN_FROZEN" == 1 ]]; then
  step bench-frozen-conc python3 bench_decode.py --url "$URL" --model "$MODEL" --phase both --concurrency 2 4 --max-tokens 200 --runs "$RUNS"
  step bench-prose-long-conc python3 bench_decode.py --url "$URL" --model "$MODEL" --phase prose_long --concurrency 2 4 --max-tokens 200 --runs "$RUNS_LONG"
fi
step bench-concurrent python3 tools/bench_concurrent.py "$BASE" "$MODEL" --levels 1,2,4 --alone --serial --output "$EV/bench_concurrent.json"
step bench-concurrent-mixed python3 tools/bench_concurrent.py "$BASE" "$MODEL" --mixed --stagger-ms 2000 --levels 2,4 --alone --output "$EV/bench_concurrent_mixed.json"
step conc-check conc_check "$EV/bench_concurrent.json" "$EV/bench_concurrent_mixed.json"
step pairs-concurrent python3 tools/pairs_concurrent.py --url "$URL" --model "$MODEL" --output "$EV/pairs_concurrent.jsonl"

# --- receipts ----------------------------------------------------------------
curl -sS --max-time 5 "$BASE/health" >"$EV/health-after.json" || true
free_both after
logs_both
{
  echo "finished_at=$(stamp)"
  printf '%s\n' "${RESULTS[@]}"
  if [[ $FAILED -eq 0 ]]; then echo "GATE=PASS"; else echo "GATE=FAIL"; fi
} >"$EV/gate.txt"
tee -a "$EV/gate.log" <"$EV/gate.txt"
exit "$FAILED"
