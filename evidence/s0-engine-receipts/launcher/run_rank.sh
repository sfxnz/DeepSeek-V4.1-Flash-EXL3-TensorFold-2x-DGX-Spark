#!/usr/bin/env bash
# Start a detached pinned-image container on this host (local) or spark2, NCCL on the private link.
# usage: run_rank.sh local|spark2 NAME WT CMD...   (CMD runs under bash -lc in /w with PYTHONPATH=src:tests/cuda)
set -euo pipefail
WHERE=$1 NAME=$2 WT=$3; shift 3
IMG=nvcr.io/nvidia/pytorch:26.07-py3
PACK=/hf/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8
ARGS=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --device /dev/infiniband
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --memory 110g --memory-swap 110g
  -v "$WT":/w -v /home/sfxnz/.cache/huggingface:/hf:ro -v /home/sfxnz/projects/data/tf-dsv41/dev:/dev-tools:ro
  -v tf-dsv41-ext:/cache -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e TF_DSV41_MODEL=$PACK
  -e PYTHONPATH=/w/src:/w/tests/cuda -e PYTHONUNBUFFERED=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  -w /w $IMG bash -lc "$*")
if [ "$WHERE" = local ]; then "${ARGS[@]}"; else ssh "$WHERE" "$(printf '%q ' "${ARGS[@]}")"; fi
