#!/usr/bin/env bash
# s2: long needles in increasing length on the CONTEXT=1048576 serve (scripts/long_needle.py each), while
# scripts/memwatch.sh samples MemAvailable on both nodes every 5 s into ctx1m/memwatch.tsv. After each prompt: the
# minimum MemAvailable of both nodes over that prompt's samples; stop before the next length if either fell below 4 GiB
# (4096 MiB), or if a prompt failed.
#   scripts/needles.sh "131072:0.1 131072:0.5 131072:0.9 262144:0.5 ..."   (run from evidence/s2-run-sh-ec28f35)
set -uo pipefail
OUT=ctx1m/needles.jsonl MW=ctx1m/memwatch.tsv
for cell in $1; do
  len=${cell%%:*} depth=${cell#*:}
  t0=$(date -u +%FT%TZ)
  { free -h; ssh spark2 free -h; } >"ctx1m/free-before-$len-$depth.txt" 2>&1
  python3 scripts/long_needle.py "$len" "$depth" >>"$OUT" 2>"ctx1m/needle-$len-$depth.err"
  rc=$?
  t1=$(date -u +%FT%TZ)
  read -r h w < <(awk -F'\t' -v a="$t0" -v b="$t1" 'NR>1 && $1>=a && $1<=b && $4!="NA" {
      if (h=="" || $2<h) h=$2; if (w=="" || $4<w) w=$4 } END {print h, w}' "$MW")
  echo "{\"cell\": \"$len@$depth\", \"rc\": $rc, \"from\": \"$t0\", \"to\": \"$t1\", \"min_avail_mib_head\": ${h:-null}, \"min_avail_mib_worker\": ${w:-null}}" >>"$OUT"
  tail -2 "$OUT"
  if (( rc != 0 )); then echo "stop: rc=$rc at $len@$depth"; break; fi
  if (( ${h:-0} < 4096 || ${w:-0} < 4096 )); then echo "stop: MemAvailable below 4 GiB at $len@$depth"; break; fi
done
