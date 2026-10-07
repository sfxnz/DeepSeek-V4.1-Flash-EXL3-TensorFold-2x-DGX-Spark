#!/usr/bin/env bash
# s2: teacher-forced NLL of the 40 G1 passages (+ their repeat) in-engine on both ranks, with the recipe's image
# (TensorFold ec28f35 as pip installed it) and the 982b704 snapshot run.sh serves. The scorer is the lab's
# data/tf-dsv41/dev/final/nll_tf.py (unchanged; it builds DeepSeekV41Engine directly, no server, no DSpark) and
# tests/cuda/dsv41_score.py from a checkout of ec28f35 (byte-identical to 0d91389's), mounted at /w. Container flags
# follow the lab's dev/q3/run_rank.sh with run.sh's kernel cache (TF_CACHE/<TF_SHA>) instead of the dev volume.
# Scores -> data/tf-dsv41/receipts/s2-recipe/tf_scores (rank 0).   usage: nll_pair.sh EVDIR
set -uo pipefail
EV=$1
IMG=tf-dsv41-flash:0.6.4-ec28f35 SHA=ec28f35c7568986666dc996f3c8b01ea59b6d0ca
WT=/home/sfxnz/projects/code/wt/recipe-engine
SNAP=/hf/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/982b70452f399814f56b46272fd30394ae10d58c
OUT=/data/receipts/s2-recipe/tf_scores
run() {  # run WHERE NAME DATA CMD
  local where=$1 name=$2 data=$3; shift 3
  local extra=(); [[ $data == 1 ]] && extra=(-v /home/sfxnz/projects/data/tf-dsv41:/data)
  local a=(docker run -d --name "$name" --gpus all --ipc=host --network host --device /dev/infiniband
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --ulimit core=1 --oom-score-adj 1000
    -v "$WT:/w:ro" -v /home/sfxnz/.cache/huggingface:/hf:ro -v /home/sfxnz/projects/data/tf-dsv41/dev:/dev-tools:ro
    "${extra[@]}" -v "/home/sfxnz/.cache/tensorfold-dsv41/$SHA:/cache/tf"
    -e TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions -e TRITON_CACHE_DIR=/cache/tf/triton -e CUDA_CACHE_PATH=/cache/tf/nv
    -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e HF_HUB_OFFLINE=1
    -e PYTHONPATH=/w/tests/cuda -e PYTHONUNBUFFERED=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    -w /tmp "$IMG" python /dev-tools/final/nll_tf.py --master 10.100.8.1 --port 29651 --model "$SNAP" "$@")
  if [[ $where == local ]]; then "${a[@]}"; else ssh spark2 "$(printf '%q ' "${a[@]}")"; fi
}
date -u +%FT%TZ >"$EV/nll_started.txt"
run spark2 tf-dsv41-s2-nll-r1 0 --sets g1 --repeat-g1 --rank 1 >/dev/null
sleep 5
run local tf-dsv41-s2-nll-r0 1 --sets g1 --repeat-g1 --rank 0 --out "$OUT" >/dev/null
echo "rank0 exit $(docker wait tf-dsv41-s2-nll-r0)"; echo "rank1 exit $(ssh spark2 docker wait tf-dsv41-s2-nll-r1)"
docker logs tf-dsv41-s2-nll-r0 >"$EV/nll_tf_rank0.log" 2>&1; docker rm tf-dsv41-s2-nll-r0 >/dev/null
ssh spark2 docker logs tf-dsv41-s2-nll-r1 >"$EV/nll_tf_rank1.log" 2>&1; ssh spark2 docker rm tf-dsv41-s2-nll-r1 >/dev/null
date -u +%FT%TZ >"$EV/nll_finished.txt"
