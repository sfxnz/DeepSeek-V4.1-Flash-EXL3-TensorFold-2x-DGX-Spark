#!/usr/bin/env bash
# s7 (from s6's scripts/vllm.sh): the same-session vLLM measurement with the same clients. Boots the vLLM sibling recipe with its canonical
# command (`env AUDIT=strict ./run.sh` in its main checkout; MAX_NUM_SEQS=2, its default), runs the cells, then stops
# it gracefully (its stop.sh uses `docker rm -f`, so `docker stop -t 60` and `docker rm` on both nodes instead), as
# evidence/s5-final/scripts/vllm.sh did.
#
#   vllm.sh TAG      (no TF serve up)
#
# Cells: the vLLM recipe's smoke_chat.py -> the frozen ruler (bench_decode.py, byte-identical in both recipes):
# --phase both --concurrency 1 2 4 --max-tokens 200 --runs 9, then --phase prose_long, the same -> sampled_conc.py
# (nofields; temperature 1.0 + top_p 0.95) at c = 1 2 4, 9 runs -> bench_diverse.py --phase both at c = 1 2 4,
# 9 waves. c = 4 queues behind MAX_NUM_SEQS=2.
set -uo pipefail
RECIPE=/home/sfxnz/projects/ai-lab/recipes/Deepseek-v4.1-Flash-EXL3-TensorFold-2x-DGX-Sparks
S7=$RECIPE/evidence/s7-speed
SC=$S7/scripts
VLLM=/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
TAG=$1
EV=$S7/$TAG
URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
C=dsv41-flash-exl3
mkdir -p "$EV"
stamp() { date -u +%FT%TZ; }
free_both() { { stamp; free -h; ssh -o BatchMode=yes spark2 free -h; } >"$EV/$1" 2>&1; }
mark() { echo "$(stamp) $TAG $*" | tee -a "$EV/steps.txt" >>"$S7/session.log"; }
step() { local name="$1"; shift; "$@" >"$EV/$name.out" 2>"$EV/$name.err"; echo "$?" >"$EV/$name.exit"; }

{ git -C "$VLLM" rev-parse HEAD; git -C "$VLLM" status --porcelain; } >"$EV/vllm-git.txt" 2>&1
sha256sum "$RECIPE/bench_decode.py" "$VLLM/bench_decode.py" "$SC/sampled_conc.py" "$SC/bench_diverse.py" >"$EV/harness.sha256"
free_both free-preboot.txt
bash "$SC/memwatch.sh" "$EV/memwatch.tsv" spark2 2 &
MW=$!
trap 'kill $MW 2>/dev/null' EXIT
mark "run.sh (env AUDIT=strict ./run.sh)"
{ echo "start $(stamp)"; echo '$ env AUDIT=strict ./run.sh'; (cd "$VLLM" && env AUDIT=strict ./run.sh); echo "rc=$?"; echo "end $(stamp)"; } \
  >"$EV/run.txt" 2>&1
curl -s -o "$EV/health.json" -w '%{http_code}\n' --max-time 5 "$URL/health" >"$EV/health.code"
curl -sS --max-time 5 "$URL/v1/models" >"$EV/models.json"
for h in local spark2; do
  if [[ $h == local ]]; then docker inspect -f '{{.Config.Image}} {{.Image}} {{json .Config.Cmd}}' "$C"
  else ssh -o BatchMode=yes spark2 "docker inspect -f '{{.Config.Image}} {{.Image}} {{json .Config.Cmd}}' $C"; fi
done >"$EV/serve-argv.txt" 2>&1

if [[ "$(cat "$EV/health.code")" == 200 ]]; then
  free_both free-serving.txt
  mark "smoke"
  step smoke-chat python3 "$VLLM/smoke_chat.py"
  mark "warm pass (discarded)"
  step warm-pass python3 "$RECIPE/bench_decode.py" --url "$URL/v1/chat/completions" --model "$MODEL" --phase prose \
    --concurrency 1 --max-tokens 200 --runs 2
  mark "ruler c=1 2 4"
  step bench-frozen python3 "$RECIPE/bench_decode.py" --url "$URL/v1/chat/completions" --model "$MODEL" --phase both \
    --concurrency 1 2 4 --max-tokens 200 --runs 9
  step bench-prose-long python3 "$RECIPE/bench_decode.py" --url "$URL/v1/chat/completions" --model "$MODEL" \
    --phase prose_long --concurrency 1 2 4 --max-tokens 200 --runs 9
  mark "sampled cells"
  step sampled-nofields python3 "$SC/sampled_conc.py" --url "$URL/v1/chat/completions" --label nofields --runs 9 \
    --concurrency 1 2 4
  step sampled-top_p095 python3 "$SC/sampled_conc.py" --url "$URL/v1/chat/completions" --label top_p095 --runs 9 \
    --temperature 1.0 --top-p 0.95 --concurrency 1 2 4
  mark "diverse cell"
  step diverse python3 "$SC/bench_diverse.py" --url "$URL/v1/chat/completions" --model "$MODEL" --phase both \
    --concurrency 1 2 4 --runs 9 --max-tokens 200
  curl -sS --max-time 5 "$URL/metrics" 2>/dev/null | grep -E '^vllm:(spec_decode|prompt_tokens|generation_tokens|num_requests)' >"$EV/metrics-end.txt"
else
  mark "vLLM not healthy, see run.txt"
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
sleep 10
free_both free-stopped.txt
mark "done"
