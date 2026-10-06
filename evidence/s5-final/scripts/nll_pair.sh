#!/usr/bin/env bash
# s5: teacher-forced NLL of the 40 G1 passages (+ their repeat) in-engine on both ranks, with the recipe's image
# (TensorFold 903a1e8 as pip installed it) and the 982b704 snapshot run.sh serves. s2's nll_pair.sh with the s5 image,
# commit and output dir. The scorer is the lab's data/tf-dsv41/dev/final/nll_tf.py (unchanged; it builds
# DeepSeekV41Engine directly, no server, no DSpark, and imports tests/cuda/dsv41_score.py), mounted at /w from
# `git archive 903a1e8` copied to the same path on both nodes (spark2 has no git checkout of TensorFold). The lab's
# dev/nll_tf.py is the same scorer before 7032f26 moved the score module out of the package: it imports
# tensorfold.families.deepseek_v41.cuda.score, which 903a1e8 does not have.
# Scores -> data/tf-dsv41/receipts/s5-recipe/tf_scores (rank 0).   usage: nll_pair.sh EVDIR
set -uo pipefail
EV=$1
IMG=tf-dsv41-flash:0.6.4-903a1e8 SHA=903a1e8af62c8f46eceee6b95481706ada30ae49
WT=/home/sfxnz/projects/data/tf-dsv41/s5-tree-903a1e8
SNAP=/hf/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/982b70452f399814f56b46272fd30394ae10d58c
OUT=/data/receipts/s5-recipe/tf_scores
# The export equals the commit's tree, and spark2's copy equals the head's.
diff <(git -C /home/sfxnz/projects/code/TensorFold archive "$SHA" | tar -t | grep -v '/$' | sort) <(cd "$WT" && find . -type f | sed 's|^\./||' | sort) >/dev/null || { echo "$WT is not $SHA's tree" >&2; exit 2; }
h=$(cd "$WT" && find . -type f | sort | xargs sha256sum | sha256sum)
[[ "$h" == "$(ssh spark2 "cd $WT && find . -type f | sort | xargs sha256sum | sha256sum")" ]] || { echo "spark2 $WT differs" >&2; exit 2; }
echo "tree $WT sha256-of-sha256s $h (both nodes)" >"$EV/nll_tree.txt"
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
run spark2 tf-dsv41-s5-nll-r1 0 --sets g1 --repeat-g1 --rank 1 >/dev/null
sleep 5
run local tf-dsv41-s5-nll-r0 1 --sets g1 --repeat-g1 --rank 0 --out "$OUT" >/dev/null
echo "rank0 exit $(docker wait tf-dsv41-s5-nll-r0)"; echo "rank1 exit $(ssh spark2 docker wait tf-dsv41-s5-nll-r1)"
docker logs tf-dsv41-s5-nll-r0 >"$EV/nll_tf_rank0.log" 2>&1; docker rm tf-dsv41-s5-nll-r0 >/dev/null
ssh spark2 docker logs tf-dsv41-s5-nll-r1 >"$EV/nll_tf_rank1.log" 2>&1; ssh spark2 docker rm tf-dsv41-s5-nll-r1 >/dev/null
date -u +%FT%TZ >"$EV/nll_finished.txt"
