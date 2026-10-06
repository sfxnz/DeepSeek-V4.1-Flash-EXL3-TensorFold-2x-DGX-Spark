#!/usr/bin/env bash
# s2: MemAvailable / SwapFree (MiB) on the head and the worker every 5 s until killed: one line per sample.
#   memwatch.sh OUT.tsv [WORKER]
OUT=$1 W=${2:-spark2}
echo -e "utc\thead_avail_mib\thead_swapfree_mib\tworker_avail_mib\tworker_swapfree_mib" >"$OUT"
while :; do
  h=$(awk '/^MemAvailable:/{a=int($2/1024)} /^SwapFree:/{s=int($2/1024)} END{print a"\t"s}' /proc/meminfo)
  w=$(ssh -o BatchMode=yes -o ConnectTimeout=4 "$W" "awk '/^MemAvailable:/{a=int(\$2/1024)} /^SwapFree:/{s=int(\$2/1024)} END{print a\"\t\"s}' /proc/meminfo" 2>/dev/null || echo -e "NA\tNA")
  echo -e "$(date -u +%FT%TZ)\t$h\t$w" >>"$OUT"
  sleep 5
done
