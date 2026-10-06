#!/usr/bin/env bash
# s5: the same-session vLLM comparison. Boots the vLLM sibling recipe with its canonical command (`env AUDIT=strict
# ./run.sh` in its main checkout, git 3de9146, image dsv41-flash-exl3-sm121:canonical-e14), runs the cells the TF
# boots ran with the same clients, then stops it gracefully (its stop.sh uses `docker rm -f`, so `docker stop -t 60`
# and `docker rm` on both nodes instead).
#
#   evidence/s5-final/scripts/vllm.sh      (from this recipe's root, with no TF serve up)
#
# Cells: the vLLM recipe's smoke_chat.py -> the frozen ruler (this repo's bench_decode.py, byte-identical to the vLLM
# recipe's) with the gate's exact commands: --phase both --concurrency 1 --max-tokens 200 --runs 9, then --phase
# prose_long --concurrency 1 --max-tokens 200 --runs 9 -> L.A.I.L (tools/measure_lail_prose.py --runs 10) -> sampled
# cells (scripts/sampled_cell.py, the same three requests) -> TF tools/bench_openai.py at t=1.0 (its t=0 fibonacci-raw
# prompt crashes the tool on vLLM: greedy emits only EOS, goldens MANIFEST) and scripts/bench_t0_chat.py -> quiet
# TF tools/prefill_cold.py on the goldens' G7 prompt set (TF's corpus and builder, counted by vLLM's /tokenize).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EV="$ROOT/evidence/s5-final/vllm"
SC="$ROOT/evidence/s5-final/scripts"
VLLM=/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
TF=/home/sfxnz/projects/code/TensorFold
PY=/home/sfxnz/projects/code/.venv-tf/bin/python
URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
PROMPTS=/home/sfxnz/projects/data/tf-dsv41/goldens/G7_prefill_prompts.json
C=dsv41-flash-exl3
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { stamp; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }
mark() { echo "$(stamp) $*" | tee -a "$EV/steps.txt"; }
step() {  # step NAME CMD...: stdout/stderr/exit under EV, as tools/session_gate.sh writes them
  local name="$1"; shift
  "$@" >"$EV/$name.out" 2>"$EV/$name.err"; echo "$?" >"$EV/$name.exit"
}

{ git -C "$VLLM" rev-parse HEAD; git -C "$VLLM" status --porcelain; } >"$EV/vllm-git.txt" 2>&1
sha256sum "$ROOT/bench_decode.py" "$VLLM/bench_decode.py" "$VLLM/tools/measure_lail_prose.py" "$PROMPTS" >"$EV/harness.sha256"
free_both free-preboot.txt
mark "run.sh (env AUDIT=strict ./run.sh)"
{ echo "start $(stamp)"; echo '$ env AUDIT=strict ./run.sh'; (cd "$VLLM" && env AUDIT=strict ./run.sh); echo "rc=$?"; echo "end $(stamp)"; } \
  >"$EV/run.txt" 2>&1
if ! grep -q '^rc=0$' "$EV/run.txt"; then echo "vLLM run.sh failed, see $EV/run.txt" >&2; fi
curl -s -o "$EV/health.json" -w '%{http_code}\n' --max-time 5 "$URL/health" >"$EV/health.code"
curl -sS --max-time 5 "$URL/v1/models" >"$EV/models.json"
for h in local spark2; do
  if [[ $h == local ]]; then docker inspect -f '{{.Config.Image}} {{.Image}} {{json .Config.Cmd}}' "$C"
  else ssh -o BatchMode=yes spark2 "docker inspect -f '{{.Config.Image}} {{.Image}} {{json .Config.Cmd}}' $C"; fi
done >"$EV/serve-argv.txt" 2>&1

if [[ "$(cat "$EV/health.code")" == 200 ]]; then
  mark "smoke"
  step smoke-chat python3 "$VLLM/smoke_chat.py"
  mark "ruler"
  step bench-frozen python3 "$ROOT/bench_decode.py" --url "$URL/v1/chat/completions" --model "$MODEL" --phase both \
    --concurrency 1 --max-tokens 200 --runs 9
  step bench-prose-long python3 "$ROOT/bench_decode.py" --url "$URL/v1/chat/completions" --model "$MODEL" \
    --phase prose_long --concurrency 1 --max-tokens 200 --runs 9
  mark "lail"
  { python3 "$VLLM/tools/measure_lail_prose.py" --url "$URL/v1/chat/completions" --model "$MODEL" --runs 10; echo "rc=$?"; } \
    >"$EV/lail.log" 2>&1
  mark "sampled cells"
  for spec in "nofields" "top_p095 --temperature 1.0 --top-p 0.95" "top_k20 --temperature 1.0 --top-k 20"; do
    read -r label args <<<"$spec"
    # shellcheck disable=SC2086
    { python3 "$SC/sampled_cell.py" --label "$label" --runs 9 $args; echo "rc=$?"; } >"$EV/sampled-$label.txt" 2>&1
  done
  mark "bench_openai"
  {
    echo "# tools/bench_openai.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, $(stamp)"
    (cd "$TF" && "$PY" tools/bench_openai.py "$URL" "$MODEL" --tokens 64 --reps 5 --temperatures 1.0 --label s5-vllm \
       --output "$EV/bench_openai.json")
    echo "rc=$?"
  } >"$EV/bench_openai.log" 2>&1
  { "$PY" "$SC/bench_t0_chat.py" "$EV/bench_t0_chat.json"; echo "rc=$?"; } >"$EV/bench_t0_chat.log" 2>&1
  mark "prefill_cold"
  { stamp; ss -tnp 2>/dev/null | grep ':8000 '; nvidia-smi --query-compute-apps=pid,name --format=csv; } \
    >"$EV/prefill-clients.txt" 2>&1
  {
    echo "# tools/prefill_cold.py at $(git -C "$TF" rev-parse HEAD) ($TF), host venv, prompts $PROMPTS sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1), $(stamp)"
    (cd "$TF" && "$PY" tools/prefill_cold.py run "$URL" "$MODEL" "$PROMPTS" "$EV/prefill_cold.json")
    echo "rc=$?"
    stamp
  } >"$EV/prefill_cold.log" 2>&1
  curl -sS --max-time 5 "$URL/metrics" 2>/dev/null | grep -E '^vllm:(spec_decode|prompt_tokens|generation_tokens)' >"$EV/metrics-end.txt"
fi

mark "logs and stop"
docker logs "$C" >"$EV/head-docker.log" 2>&1
ssh -o BatchMode=yes spark2 docker logs "$C" >"$EV/worker-docker.log" 2>&1
free_both free-end.txt
{
  echo "stop begin $(stamp)"
  docker stop -t 60 "$C" && docker rm "$C"; echo "head stop rc=$? $(stamp)"
  ssh -o BatchMode=yes spark2 "docker stop -t 60 $C && docker rm $C"; echo "worker stop rc=$? $(stamp)"
  echo "stop end $(stamp)"
} >"$EV/stop.txt" 2>&1
mark "vllm done"
