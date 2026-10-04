#!/usr/bin/env bash
# Graceful stop of both TensorFold ranks, this recipe's containers only (name and label). Rank 0 (HTTP) first: SIGTERM
# closes its server and its rendezvous store; the script then waits up to STOP_TIMEOUT seconds for rank 1 to exit by
# itself before it stops it. A rank stuck in a collective (its peer died mid-decode) is killed after STOP_TIMEOUT
# seconds.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-tf-dsv41-flash}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$CONTAINER_NAME" =~ ^tf-dsv41-[a-z0-9-]+$ ]] || { echo "CONTAINER_NAME=$CONTAINER_NAME must match tf-dsv41-[a-z0-9-]+." >&2; exit 1; }
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "STOP_TIMEOUT=$STOP_TIMEOUT is not a positive integer." >&2; exit 1; }

if [[ -z "${WORKER_HOST:-}" && -f "$SCRIPT_DIR/.run-state/worker_host" ]]; then
  WORKER_HOST="$(tr -d '[:space:]' <"$SCRIPT_DIR/.run-state/worker_host")"
fi
WORKER_HOST="${WORKER_HOST:-spark2}"

# stop_ours NAME TIMEOUT WAIT: stop and remove NAME only when it carries run.sh's label. WAIT=1 first gives it
# TIMEOUT seconds to exit by itself. Runs here and, sent over ssh, on the worker.
stop_ours() {
  local name="$1" t="$2" wait="$3"
  # A here-string, not a pipe: under pipefail a `grep -q` that exits first can turn a match into "No container".
  if ! grep -Fqx "$name" <<<"$(docker ps -a --filter label=ai-lab.recipe=dsv41-tensorfold --format '{{.Names}}')"; then
    echo "No container $name from this recipe on $(hostname -s)"
    return 0
  fi
  if [[ "$wait" == 1 ]]; then
    for _ in $(seq 1 "$t"); do
      [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" == true ]] || break
      sleep 1
    done
  fi
  echo "Stopping $name on $(hostname -s)"
  docker stop -t "$t" "$name" >/dev/null 2>&1 || true
  docker rm -f "$name" >/dev/null
  echo "Stopped $name on $(hostname -s)"
}

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

stop_ours "$CONTAINER_NAME" "$STOP_TIMEOUT" 0

if [[ "$ORCHESTRATE" == "0" ]]; then
  exit 0
fi

if [[ "$ORCHESTRATE" == "auto" ]]; then
  case "$(host_short)" in
    spark2*) ;;
    *)
      if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
        ssh "$WORKER_HOST" bash -s <<<"$(declare -f stop_ours)
stop_ours $(printf %q "$CONTAINER_NAME") $STOP_TIMEOUT 1"
      else
        echo "Cannot SSH to $WORKER_HOST. Remote $CONTAINER_NAME may still be running. Set WORKER_HOST to the rank this head started." >&2
        exit 1
      fi
      ;;
  esac
fi
