#!/usr/bin/env bash
# Dev only (outside the PR, C5): the integration tip (A) against the tip + C5 (B), before/after/after/before, each on a
# fresh two-rank boot at --context 65538; B boots also sweep d=3 against the default policy (POLICIES).
set -uo pipefail
D=/home/sfxnz/projects/data/tf-dsv41/dev/c5 W=/home/sfxnz/projects/code/wt
i=0
for side in ${SIDES:-A B B A}; do
  i=$((i+1)); tag="$i$side"; wt=$W/C5-before; [ $side = B ] && wt=$W/C5
  echo "== $tag $wt $(date -u +%T)"
  bash /home/sfxnz/projects/data/tf-dsv41/dev/gpu_slot.sh > /dev/null || { echo "slot busy"; exit 1; }
  $D/serve_pair.sh "$tag" "$wt" $((29870+i)) --context 65538 || { $D/stop_pair.sh "$tag"; exit 1; }
  if [ $side = B ]; then POLICIES=$POLICIES $D/session.sh "$tag" "$wt"; else POLICIES= $D/session.sh "$tag" "$wt"; fi
  $D/stop_pair.sh "$tag"
done
echo "== done $(date -u +%T)"
