#!/usr/bin/env bash
# DeepSeek-V4.1-Flash (EXL3 routed experts, DeepSeek FP8 elsewhere) on 2x DGX Spark (GB10) with TensorFold's
# deepseek_v41 CUDA family, TP=2. Guards and orchestration follow the lab's Qwen3.8-Flash-Next TensorFold recipe.
# The head (spark1) checks the worker, its own weights and the worker's snapshot, starts rank 1 on WORKER_HOST over ssh,
# then rank 0, which serves HTTP.
set -euo pipefail

# The HF cache is mounted read-only here in both containers.
HF_HOME_IN_CONTAINER=/cache/huggingface

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-sfxnz/DeepSeek-V4.1-Flash-EXL3}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
IMAGE="${IMAGE:-tf-dsv41-flash:0.6.4-41306d5}"
TF_REPO="${TF_REPO:-https://github.com/sfxnz/TensorFold.git}"
TF_SHA="${TF_SHA:-41306d5e2acd5651fc0954609b3a651d2ef61d6e}"
CONTAINER_NAME="${CONTAINER_NAME:-tf-dsv41-flash}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29571}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1,roceP2p1s0f1}"
TP="${TP:-2}"
CONTEXT="${CONTEXT:-1048576}"
MTP_DRAFTS="${MTP_DRAFTS:-3}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-}"
THINKING="${THINKING:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TOP_P="${TOP_P:-1.0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-982b70452f399814f56b46272fd30394ae10d58c}"
SNAPSHOT="${HF_CACHE}/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/${SNAPSHOT_SHA}"
CONFIG_SHA256="${CONFIG_SHA256:-6469adab394edead3eec148323e7471582d08acdf60a438bf0c9e815b69f36c5}"
INDEX_SHA256="${INDEX_SHA256:-91731e4af38696bd4c09e960f4b599d1d49f35d445e4f88a43cc355d9e139f03}"
SHARDS="${SHARDS:-48}"
SHARD_BYTES="${SHARD_BYTES:-357466041064}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
TF_CACHE="${TF_CACHE:-$HOME/.cache/tensorfold-dsv41}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-1000}"
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_MIN_AVAIL_MB="${MEMGUARD_MIN_AVAIL_MB:-3072}"
MEMGUARD_MIN_SWAP_FREE_MB="${MEMGUARD_MIN_SWAP_FREE_MB:-2048}"
MEM_GATE_GIB="${MEM_GATE_GIB:-100}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
BENCH_ONLY="${BENCH_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
TF_DSV41_CACHE_GIB="${TF_DSV41_CACHE_GIB:-}"
TF_DSV41_CACHE_ENTRIES="${TF_DSV41_CACHE_ENTRIES:-}"
# END generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.run-state"
# The worker runs a /tmp copy of this script: keep its memguard log with its cache, not in /tmp.
if [[ "${ROLE:-}" == worker ]]; then STATE_DIR="$TF_CACHE/run-state"; fi
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
FORCE_UNSAFE_MEM_GATE="${FORCE_UNSAFE_MEM_GATE:-0}"
# IMAGE_ONLY=1: build the image and copy it to the worker, then exit. No GPU, no weights, no memory gate.
IMAGE_ONLY="${IMAGE_ONLY:-0}"
# Every container this recipe starts carries this label; run.sh and stop.sh stop or remove only containers that do.
RECIPE_LABEL=ai-lab.recipe=dsv41-tensorfold
# The checkpoint's native window (config.json); the engine refuses a larger --context.
NATIVE_CONTEXT=1048576
# DSpark drafts at most 5 tokens a round (families/deepseek_v41/cuda BLOCK).
MAX_MTP_DRAFTS=5
# Admission grants MemAvailable less max(4 GiB, MemTotal / 10), 12.1 GiB on a Spark; the default window's startup
# estimate is 79.09 GiB (evidence/s0-engine-receipts/serve_default_rank0.log). Below this gate it cannot fit.
MIN_MEM_GATE_GIB=92
# Every NAME="${NAME:-…}" line of the generated block. The engine knobs (TF_DSV41_*) go to both ranks as
# `-e KEY=VALUE` (container_env), and the worker gets every name forwarded (worker_env).
mapfile -t GENERATED_VARS < <(sed -n '/^# BEGIN generated from recipe.yaml/,/^# END generated/s/^\([A-Z][A-Z0-9_]*\)="\${.*/\1/p' "${BASH_SOURCE[0]}")
ENGINE_VARS=()
FORWARD_VARS=(HF_HUB_DISABLE_XET FORCE_UNSAFE_MEM_GATE)
for v in "${GENERATED_VARS[@]}"; do
  [[ "$v" =~ ^TF_DSV41_ ]] && ENGINE_VARS+=("$v")
  [[ "$v" == ORCHESTRATE || "$v" == WORKER_HOST ]] || FORWARD_VARS+=("$v")
done
# Set by run.sh itself on every container; EXTRA_ENV may not override them.
FIXED_ENV="HF_HOME HF_HUB_OFFLINE TENSORFOLD_NO_UPDATE_CHECK PYTHONUNBUFFERED PYTHONDONTWRITEBYTECODE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_DEBUG"

die() {
  echo "$*" >&2
  exit 1
}

(( ${#GENERATED_VARS[@]} > 0 )) || die "run.sh found no generated block (run it as a file, not from stdin)."

# --- integers and flags. Leading zeros are refused: bash reads 010 as octal 8.
for name in PORT MASTER_PORT TP CONTEXT MAX_TOKENS SHARDS SHARD_BYTES MEMGUARD_MIN_AVAIL_MB MEMGUARD_MIN_SWAP_FREE_MB MEM_GATE_TIMEOUT READY_TIMEOUT; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name=${!name} is not a positive decimal integer."
done
for name in MTP_DRAFTS MEM_GATE_GIB; do
  [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || die "$name=${!name} is not a non-negative decimal integer."
done
for name in THINKING SKIP_DOWNLOAD HF_HUB_DISABLE_XET MEMGUARD BENCH_ONLY FORCE_UNSAFE_MEM_GATE IMAGE_ONLY; do
  [[ "${!name}" =~ ^[01]$ ]] || die "$name=${!name} must be 0 or 1."
done
[[ "$OOM_SCORE_ADJ" =~ ^(-?[1-9][0-9]*|0)$ ]] && (( OOM_SCORE_ADJ >= -1000 && OOM_SCORE_ADJ <= 1000 )) || die "OOM_SCORE_ADJ=$OOM_SCORE_ADJ must be an integer in [-1000, 1000]."
[[ -z "$MTP_CONFIDENCE" || "$MTP_CONFIDENCE" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] || die "MTP_CONFIDENCE=$MTP_CONFIDENCE must be empty or a decimal in (0, 1], e.g. 0.70."
[[ "$TOP_P" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] || die "TOP_P=$TOP_P must be a decimal in (0, 1], e.g. 1.0 or 0.95."
[[ "$TF_SHA" =~ ^[0-9a-f]{40}$ ]] || die "TF_SHA=$TF_SHA is not a 40-hex TensorFold commit."
[[ "$TF_REPO" =~ ^https://[A-Za-z0-9./_-]+\.git$ ]] || die "TF_REPO=$TF_REPO must be an https git URL ending .git."
[[ "$SNAPSHOT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SNAPSHOT_SHA=$SNAPSHOT_SHA is not a 40-hex snapshot revision."
for name in CONFIG_SHA256 INDEX_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "$name=${!name} is not a 64-hex sha256."
done
for name in HF_CACHE TF_CACHE; do
  [[ "${!name}" == /* && "${!name}" != *[[:space:]:]* ]] || die "$name=${!name} must be an absolute path without spaces or ':'."
done
# run.sh and stop.sh stop and remove this name: keep it to this recipe's own prefix.
[[ "$CONTAINER_NAME" =~ ^tf-dsv41-[a-z0-9-]+$ ]] || die "CONTAINER_NAME=$CONTAINER_NAME must match tf-dsv41-[a-z0-9-]+ (run.sh stops and removes it)."
case "$ORCHESTRATE" in auto | 0) ;; *) die "ORCHESTRATE=$ORCHESTRATE must be auto or 0." ;; esac

# --- topology, window, memory.
(( TP == 2 )) || die "TP=$TP: the deepseek_v41 family serves on two ranks (one per Spark) only."
(( CONTEXT <= NATIVE_CONTEXT )) || die "CONTEXT=$CONTEXT exceeds the native window $NATIVE_CONTEXT; the engine refuses it."
(( MTP_DRAFTS <= MAX_MTP_DRAFTS )) || die "MTP_DRAFTS=$MTP_DRAFTS exceeds $MAX_MTP_DRAFTS, the most DSpark drafts a round."
(( PORT <= 65535 && MASTER_PORT <= 65535 )) || die "PORT=$PORT / MASTER_PORT=$MASTER_PORT must be at most 65535."
(( PORT != MASTER_PORT )) || die "PORT=$PORT collides with MASTER_PORT=$MASTER_PORT."
if (( MEM_GATE_GIB < MIN_MEM_GATE_GIB )) && [[ "$FORCE_UNSAFE_MEM_GATE" != 1 ]]; then
  die "MEM_GATE_GIB=$MEM_GATE_GIB is below $MIN_MEM_GATE_GIB GiB: the engine's admission cannot fit the default window below it. FORCE_UNSAFE_MEM_GATE=1 for one boot, and record it."
fi
[[ "$SERVED_NAME" =~ ^[A-Za-z0-9._/:-]+$ ]] || die "SERVED_NAME=$SERVED_NAME must be one word of A-Z a-z 0-9 . _ / : -."

# --- engine knobs: one word each (no spaces or quotes); the family reads both as numbers.
for v in "${ENGINE_VARS[@]}"; do
  [[ "${!v}" =~ ^[^[:space:]\"\'\\]*$ ]] || die "$v='${!v}' has a space or a quote."
done
[[ -z "$TF_DSV41_CACHE_GIB" || "$TF_DSV41_CACHE_GIB" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "TF_DSV41_CACHE_GIB=$TF_DSV41_CACHE_GIB must be empty or a number of GiB, 0 or more."
[[ -z "$TF_DSV41_CACHE_ENTRIES" || "$TF_DSV41_CACHE_ENTRIES" =~ ^(0|[1-9][0-9]*)$ ]] || die "TF_DSV41_CACHE_ENTRIES=$TF_DSV41_CACHE_ENTRIES must be empty or a non-negative integer."

# --- EXTRA_ENV: KEY=VALUE words, no spaces inside a value, and not a variable run.sh sets itself.
EXTRA_ENV_KEYS=" "
for kv in $EXTRA_ENV; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "EXTRA_ENV entry '$kv' is not KEY=VALUE."
  k="${kv%%=*}"
  for g in "${GENERATED_VARS[@]}" $FIXED_ENV; do
    [[ "$k" == "$g" ]] && die "EXTRA_ENV sets $k, which run.sh sets itself. Set $k=… in the environment (or recipe.yaml) instead."
  done
  EXTRA_ENV_KEYS+="$k "
done

# --- an exported TF_DSV41_* variable that is not a recipe knob would never reach the containers. Refuse it rather
# than measure a boot without it.
for k in $(compgen -e); do
  [[ "$k" =~ ^TF_DSV41_ ]] || continue
  [[ " ${ENGINE_VARS[*]} " == *" $k "* || "$EXTRA_ENV_KEYS" == *" $k "* ]] && continue
  die "$k is exported but is not a recipe.yaml knob, so run.sh would not pass it to the containers. Pass it with EXTRA_ENV=\"$k=…\", or unset it."
done

# --- EXTRA_ARGS must not re-set a flag run.sh builds; argparse keeps the last value, so a duplicate would bypass the
# guard on its variable or desynchronise the two ranks. TensorFold's parser expands unambiguous prefixes (--cont means
# --context), so any prefix of a guarded flag is refused too.
OWNED_FLAGS="--tp --rank --master --master-port --host --port --name --context --mtp-drafts --mtp-confidence --no-drafts --thinking --no-thinking --max-tokens --top-p --no-update-check"
# The family refuses these at startup or per request, or ignores them (docs/recipes/deepseek-v4.1-flash.md).
REFUSED_FLAGS="--parallel --prefill-fp8 --no-prefill-fp8 --drafter --thinking-budget --vision --vision-urls --kv-dtype"
for w in $EXTRA_ARGS; do
  f="${w%%=*}"
  [[ "$f" == --?* ]] || continue
  for g in $OWNED_FLAGS; do
    [[ "$g" == "$f"* ]] && die "EXTRA_ARGS sets $w (argparse reads it as $g), which run.sh passes itself. Use TP, PORT, MASTER_PORT, SERVED_NAME, CONTEXT, MTP_DRAFTS, MTP_CONFIDENCE, THINKING, MAX_TOKENS or TOP_P instead."
  done
  for g in $REFUSED_FLAGS; do
    [[ "$g" == "$f"* ]] && die "EXTRA_ARGS sets $w ($g): the deepseek_v41 family refuses or ignores it (README 'Not supported')."
  done
done

API_HOST=0.0.0.0
[[ "$BENCH_ONLY" == 1 ]] && API_HOST=127.0.0.1

serve_args() {
  # The `tensorfold serve` flags after the model directory, for one rank. Both ranks get theirs from here.
  local rank="$1"
  local args=(--tp "$TP" --rank "$rank" --master "$HEAD_IP" --master-port "$MASTER_PORT" --name "$SERVED_NAME"
    --context "$CONTEXT" --mtp-drafts "$MTP_DRAFTS" --top-p "$TOP_P" --no-update-check)
  [[ -n "$MTP_CONFIDENCE" ]] && args+=(--mtp-confidence "$MTP_CONFIDENCE")
  if [[ "$THINKING" == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
  if [[ "$rank" == 0 ]]; then
    args+=(--host "$API_HOST" --port "$PORT" --max-tokens "$MAX_TOKENS")
  fi
  printf '%s\n' "${args[@]}"
}

container_env() {
  # Every KEY=VALUE of a rank's container, one a line, identical on both ranks: the fixed settings, every non-empty
  # engine knob, then EXTRA_ENV. An empty knob is left out, so the engine's own default applies.
  local v kv
  printf '%s\n' "HF_HOME=$HF_HOME_IN_CONTAINER" HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions TRITON_CACHE_DIR=/cache/tf/triton \
    CUDA_CACHE_PATH=/cache/tf/nv TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    "NCCL_SOCKET_IFNAME=$IFACE" "NCCL_IB_HCA=$HCA" NCCL_DEBUG=WARN
  for v in "${ENGINE_VARS[@]}"; do
    if [[ -n "${!v}" ]]; then printf '%s=%s\n' "$v" "${!v}"; fi
  done
  for kv in $EXTRA_ENV; do printf '%s\n' "$kv"; done
}

if [[ "${VALIDATE_ONLY:-0}" != 0 ]]; then
  case "$VALIDATE_ONLY" in
    1)
      printf '==> validate-only image=%s tf=%s repo=%s snapshot=%s tp=%s ctx=%s drafts=%s confidence=%s thinking=%s max_tokens=%s top_p=%s hca=%s host=%s mem_gate=%s engine_env=%s\n' \
        "$IMAGE" "$TF_SHA" "$TF_REPO" "$SNAPSHOT_SHA" "$TP" "$CONTEXT" "$MTP_DRAFTS" "${MTP_CONFIDENCE:-none}" "$THINKING" \
        "$MAX_TOKENS" "$TOP_P" "$HCA" "$API_HOST" "$MEM_GATE_GIB" "$(container_env | grep -c '^TF_DSV41_' || true)"
      ;;
    args)
      # Both ranks' argv and the shared container env, to diff against another launcher.
      for r in 1 0; do printf 'rank%s: tensorfold serve %s %s\n' "$r" "$SNAPSHOT_IN_CONTAINER" "$(serve_args "$r" | tr '\n' ' ' | sed 's/ $//')"; done
      container_env | sed 's/^/env: /'
      ;;
    *) die "VALIDATE_ONLY=$VALIDATE_ONLY must be 0, 1 or args." ;;
  esac
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

image_sha() {
  # The image's tensorfold.sha label, empty when the image is missing.
  docker image inspect -f '{{ index .Config.Labels "tensorfold.sha" }}' "$IMAGE" 2>/dev/null || true
}

ensure_image() {
  local have
  have="$(image_sha)"
  if [[ -z "$have" ]]; then
    [[ -f "$SCRIPT_DIR/docker/Dockerfile" ]] || die "Image $IMAGE is missing and $SCRIPT_DIR/docker/Dockerfile is not here. Build it on the head (README 'Image')."
    log "Building $IMAGE (TensorFold $TF_REPO @ $TF_SHA) from docker/Dockerfile"
    docker build --build-arg "TF_REPO=$TF_REPO" --build-arg "TF_SHA=$TF_SHA" -t "$IMAGE" "$SCRIPT_DIR/docker"
    have="$(image_sha)"
  fi
  [[ "$have" == "$TF_SHA" ]] || die "Image $IMAGE carries TensorFold '${have:-unknown}', not TF_SHA=$TF_SHA. Rebuild it from docker/Dockerfile or set IMAGE to the matching tag."
  log "Image $IMAGE (TensorFold $TF_SHA)"
}

check_snapshot() {
  # check_snapshot DIR CONFIG_SHA256 INDEX_SHA256 SHARDS SHARD_BYTES: the pinned revision is whole on this node.
  # config.json and the index have their pinned sha256, the index names SHARDS shards and they sum to SHARD_BYTES,
  # and each shard's safetensors header ends exactly at its file size (a truncated copy fails here, not mid-load).
  # Shards 47 and 48 are the Engram tables: each rank reads its half of every row from its own disk.
  python3 - "$@" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

snap, config_sha, index_sha, want_shards, want_bytes = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
bad = []


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


for name in ("tokenizer.json", "tokenizer_config.json"):
    p = snap / name
    if not p.is_file() or p.stat().st_size == 0:
        bad.append(name)
for name, want in (("config.json", config_sha), ("model.safetensors.index.json", index_sha)):
    p = snap / name
    try:
        got = sha256(p)
    except OSError as exc:
        sys.exit(f"snapshot {snap} incomplete: {name} ({exc.__class__.__name__})")
    if got != want:
        bad.append(f"{name} (sha256 {got[:12]}, pinned {want[:12]})")
try:
    shards = sorted(set(json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"].values()))
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"snapshot {snap} incomplete: model.safetensors.index.json ({exc})")
if len(shards) != want_shards:
    bad.append(f"the index names {len(shards)} shards, pinned {want_shards}")
total = 0
for shard in shards:
    p = snap / shard
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if 8 + n + end != size:
            bad.append(f"{shard} (truncated)")
        total += size
    except (OSError, ValueError, struct.error) as exc:
        bad.append(f"{shard} ({exc.__class__.__name__})")
if not bad and total != want_bytes:
    bad.append(f"shards hold {total} bytes, pinned {want_bytes}")
if bad:
    more = " ..." if len(bad) > 8 else ""
    sys.exit(f"snapshot {snap} incomplete: {', '.join(bad[:8])}{more}")
print(f"==> snapshot {snap}: config and index sha256, {len(shards)} shards, {total} bytes, headers ok")
PY
}

snapshot_ok() { check_snapshot "$SNAPSHOT" "$CONFIG_SHA256" "$INDEX_SHA256" "$SHARDS" "$SHARD_BYTES"; }

ensure_weights() {
  if snapshot_ok 2>/dev/null; then
    log "Using pinned snapshot $SNAPSHOT (config and index sha256, shard count, sizes and headers checked)"
    return 0
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    snapshot_ok || true
    die "SKIP_DOWNLOAD=1 and the pinned snapshot is incomplete."
  fi
  if [[ "$ROLE" == worker ]]; then
    # The worker may have no internet (spark2 here): a download would fail late with a misleading error.
    snapshot_ok || true
    die "The pinned snapshot is incomplete on the worker $(host_short), and the worker does not download. $(rsync_hint)"
  fi
  local HF="" free have need
  HF="$(hf_bin || true)"
  [[ -n "$HF" ]] || die "No hf CLI on PATH and snapshot $SNAPSHOT is incomplete."
  # Free space for what the download still needs: the shards plus 1 GiB, less what blobs/ already holds (a resumed
  # download; blobs of other revisions in the same cache entry count too, so this can under-count).
  mkdir -p "$HF_CACHE/hub"
  free="$(df -B1 --output=avail "$HF_CACHE/hub" | tail -1 | tr -d '[:space:]')"
  have="$(du -sb "${SNAPSHOT%/snapshots/*}/blobs" 2>/dev/null | cut -f1 || true)"
  need=$(( SHARD_BYTES + (1 << 30) - ${have:-0} ))
  (( free >= need )) || die "$HF_CACHE has $(( free >> 30 )) GiB free; the download needs $(( need >> 30 )) GiB more. Free space or set HF_CACHE."
  export HF_HUB_DISABLE_XET
  log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE; 333 GiB)"
  "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA" --cache-dir "$HF_CACHE/hub"
  snapshot_ok || die "Snapshot still incomplete after download."
}

rsync_hint() {
  # Only the pinned snapshot, links into blobs/ resolved to files: other revisions in the cache entry stay behind.
  printf 'Copy it from the head over the link, on the worker (%s GiB free needed): rsync -aL --partial --mkpath %s:%s/ %s/\n' \
    "$(( SHARD_BYTES >> 30 ))" "$HEAD_IP" "$SNAPSHOT" "$SNAPSHOT"
}

check_worker_snapshot() {
  # The same check on the worker, over ssh, before anything starts: a missing or partial copy fails here.
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" bash -s <<<"$(declare -f check_snapshot)
check_snapshot $(printf %q "$SNAPSHOT") $CONFIG_SHA256 $INDEX_SHA256 $SHARDS $SHARD_BYTES" ||
    die "The pinned snapshot is incomplete on $WORKER_HOST (above). $(rsync_hint)"
}

check_hca() {
  local h st
  for h in ${HCA//,/ }; do
    st="$(cat "/sys/class/infiniband/$h/ports/1/state" 2>/dev/null || true)"
    [[ "$st" == *ACTIVE* ]] || die "HCA $h port 1 on $(host_short) is '${st:-unreadable}', not ACTIVE (HCA=$HCA, ibv_devinfo)."
  done
  log "HCAs $HCA ACTIVE on $(host_short)"
}

# This recipe's containers on this node (running or not), by label.
ours_names() { docker ps -a --filter "label=$RECIPE_LABEL" --format '{{.Names}}' 2>/dev/null || true; }

refuse_foreign_serve() {
  local name devices ours
  ours="$(ours_names)"
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ "$name" == "$CONTAINER_NAME" ]] && grep -Fqx "$name" <<<"$ours"; then continue; fi
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if grep -Eqi 'gpu|nvidia|infiniband' <<<"$devices"; then
      die "$name is using GPUs or InfiniBand on $(host_short). This recipe needs exclusive GPUs on both Sparks. Stop that serve first (its own stop script). Do not docker rm it from this script."
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_gpu_busy() {
  # A here-string, not a pipe: under pipefail a `grep -q` that exits first can turn a match into a failure.
  if grep -q '[0-9]' <<<"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"; then
    die "A CUDA process is running on $(host_short) (nvidia-smi --query-compute-apps). Stop it first."
  fi
}

listening() { grep -qE "[:.]$1\$" <<<"$(ss -ltnH 2>/dev/null | awk '{print $4}')"; }

refuse_busy_ports() {
  local p
  command -v ss >/dev/null 2>&1 || die "ss (iproute2) is not on PATH on $(host_short): run.sh cannot check PORT / MASTER_PORT."
  for p in "$PORT" "$MASTER_PORT"; do
    if listening "$p"; then die "Port $p is already in use on $(host_short) (PORT or MASTER_PORT)."; fi
  done
}

stop_local() {
  # This recipe's rank container on this node, and only when it carries the recipe's label.
  grep -Fqx "$CONTAINER_NAME" <<<"$(ours_names)" || return 0
  log "Stopping existing container $CONTAINER_NAME"
  docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
}

memavail_gib() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' "${1:-/proc/meminfo}"; }

mem_gate() {
  # MemAvailable >= MEM_GATE_GIB before a load. The engine's admission reads MemAvailable too (on GB10 the page
  # cache counts as available), so a short node waits here instead of failing admission.
  local t0 avail
  (( MEM_GATE_GIB > 0 )) || { log "memory gate off (MEM_GATE_GIB=0)"; return 0; }
  t0="$(date +%s)"
  while :; do
    avail="$(memavail_gib /proc/meminfo)"
    if (( avail >= MEM_GATE_GIB )); then log "MemAvailable $avail GiB >= $MEM_GATE_GIB on $(host_short)"; return 0; fi
    (( $(date +%s) - t0 < MEM_GATE_TIMEOUT )) ||
      die "MemAvailable $avail GiB < MEM_GATE_GIB=$MEM_GATE_GIB on $(host_short) after ${MEM_GATE_TIMEOUT}s. Find what holds the memory (free -h, docker ps)."
    log "waiting for memory on $(host_short): MemAvailable $avail GiB, want $MEM_GATE_GIB"
    sleep 10
  done
}

prepare_node() {
  # Everything a rank needs before its container starts, on this node. Safe to repeat.
  mkdir -p "$STATE_DIR" "$HF_CACHE" "$TF_CACHE/$TF_SHA"
  ensure_image
  ensure_weights
  check_hca
  refuse_gpu_busy
  mem_gate
}

# Host memory watchdog. GB10 is unified memory: when the serve plus host load exhausts RAM and swap,
# the host thrashes for minutes before the kernel OOM killer acts, and it then picks small services
# first. Kill the serve instead once available RAM and free swap are both low for 3 samples (6 s).
# Exits when the container is gone.
memguard_loop() {
  local name="$1" min_avail_kb=$(( $2 * 1024 )) min_swap_kb=$(( $3 * 1024 )) hits=0 n=0 avail swapfree
  while :; do
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swapfree="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    if (( avail < min_avail_kb && swapfree < min_swap_kb )); then
      hits=$(( hits + 1 ))
    else
      hits=0
    fi
    if (( hits >= 3 )); then
      logger -t tf-dsv41-memguard "MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      echo "$(date -Is) MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      docker kill "$name" >/dev/null 2>&1 || true
      return 0
    fi
    n=$(( n + 1 ))
    if (( n % 8 == 0 )) && [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
      return 0
    fi
    sleep 2
  done
}

start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || { log "memguard off (MEMGUARD=$MEMGUARD)"; return 0; }
  local logf="$STATE_DIR/memguard.log"
  mkdir -p "$STATE_DIR"
  setsid bash -c "$(declare -f memguard_loop); memguard_loop \"\$@\"" memguard \
    "$CONTAINER_NAME" "$MEMGUARD_MIN_AVAIL_MB" "$MEMGUARD_MIN_SWAP_FREE_MB" >>"$logf" 2>&1 </dev/null 9>&- &
  disown || true
  log "memguard pid=$! min_avail=${MEMGUARD_MIN_AVAIL_MB}MB min_swap_free=${MEMGUARD_MIN_SWAP_FREE_MB}MB log=$logf"
}

start_local() {
  # start_local RANK [prepared]: `prepared` when prepare_node already ran for this start (the head prepares before it
  # starts rank 1); only MemAvailable is read again.
  local rank="$1" prepared="${2:-}" kv
  command -v docker >/dev/null 2>&1 || die "docker not found"
  refuse_foreign_serve
  stop_local
  if [[ "$prepared" == prepared ]]; then mem_gate; else prepare_node; fi

  local env_args=() args=()
  while IFS= read -r kv; do env_args+=(-e "$kv"); done < <(container_env)
  mapfile -t args < <(serve_args "$rank")

  # --init: a rank that installs no SIGTERM handler would ignore it as PID 1 and `docker stop` would wait out its
  # timeout. --ulimit memlock + IPC_LOCK: pinned host buffers for NCCL over RoCE. --ulimit stack: NVIDIA's
  # recommendation for this image, as the receipts ran. --ulimit core=1: no multi-GiB core dumps in host RAM.
  log "Starting $CONTAINER_NAME rank=$rank tp=$TP ctx=$CONTEXT drafts=$MTP_DRAFTS thinking=$THINKING hca=$HCA"
  # shellcheck disable=SC2086 # EXTRA_ARGS is word-split on purpose
  docker run -d \
    --name "$CONTAINER_NAME" \
    --label "$RECIPE_LABEL" \
    --init \
    --restart no \
    --oom-score-adj "$OOM_SCORE_ADJ" \
    --ulimit core=1 \
    --gpus all \
    --network host \
    --ipc host \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    --log-opt max-size=200m --log-opt max-file=3 \
    -v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro" \
    -v "${TF_CACHE}/${TF_SHA}:/cache/tf" \
    "${env_args[@]}" \
    "$IMAGE" \
    tensorfold serve "$SNAPSHOT_IN_CONTAINER" "${args[@]}" $EXTRA_ARGS >/dev/null
  start_memguard
}

worker_env() {
  # Every setting the worker rank needs, shell-quoted for one ssh command line.
  local v out="ROLE=worker ORCHESTRATE=0"
  for v in "${FORWARD_VARS[@]}"; do
    out+=" $v=$(printf '%q' "${!v}")"
  done
  printf '%s\n' "$out"
}

worker_state() {
  # Prints true, false or missing; prints nothing when ssh itself fails. docker inspect prints an
  # empty line before failing on a missing container, so strip whitespace.
  { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
    "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || echo missing" 2>/dev/null || true; } |
    tr -d '[:space:]'
}

abort_worker_dead() {
  echo "Worker $CONTAINER_NAME on $WORKER_HOST is not running ($1). Worker logs:" >&2
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker logs --tail 120 '$CONTAINER_NAME'" >&2 2>&1 || true
  exit 1   # the head's EXIT trap stops both ranks
}

sync_worker_image() {
  # The worker runs the head's exact image: same image ID, or the head's copy is sent over the link (the worker
  # has no internet, and no rank pair is built from two different builds).
  local local_id remote_id
  local_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  remote_id="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker image inspect -f '{{.Id}}' '$IMAGE' 2>/dev/null || true" | tr -d '[:space:]')" ||
    die "Lost SSH to $WORKER_HOST while comparing image IDs."
  [[ "$remote_id" == "$local_id" ]] && return 0
  log "Copying $IMAGE to $WORKER_HOST (worker has ${remote_id:-none}, head has $local_id)"
  docker save "$IMAGE" | ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" docker load >/dev/null
}

wait_ready() {
  local watch_worker="${1:-0}"
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models (up to ${READY_TIMEOUT}s)"
  local i health body state
  for i in $(seq 1 $(( (READY_TIMEOUT + 4) / 5 ))); do
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ "$health" == "200" && -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://${API_HOST}:${PORT}/v1  (context=$CONTEXT)"
      printf '%s\n' "$body"
      # The admission line: the window both ranks allocated and the memory it took.
      docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'startup estimate' | tail -1 || true
      return 0
    fi
    if ! grep -Fqx "$CONTAINER_NAME" <<<"$(docker ps --format '{{.Names}}')"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    # Every ~10 s, so a dead worker aborts the head within a minute instead of a long rendezvous wait.
    if [[ "$watch_worker" == 1 ]] && (( i % 2 == 0 )); then
      state="$(worker_state)"
      [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out after ${READY_TIMEOUT}s waiting for the API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

fail_stop() {
  trap - EXIT
  echo "$1: stopping both ranks. Logs: $STATE_DIR/fail-rank0.log, $STATE_DIR/fail-rank1.log" >&2
  docker logs --tail 300 "$CONTAINER_NAME" >"$STATE_DIR/fail-rank0.log" 2>&1 || true
  if [[ "$ORCHESTRATE" == auto ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker logs --tail 300 '$CONTAINER_NAME'" >"$STATE_DIR/fail-rank1.log" 2>&1 || true
  fi
  CONTAINER_NAME="$CONTAINER_NAME" WORKER_HOST="$WORKER_HOST" ORCHESTRATE="$ORCHESTRATE" "$SCRIPT_DIR/stop.sh" >&2 || true
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

require_worker_ssh() {
  if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
    die "Cannot SSH to $WORKER_HOST. Refusing to start a TP=2 head rank alone."
  fi
}

check_worker_free() {
  # The worker's refusals, read-only, before the head's long steps (download, image copy, memory gate). Nothing is
  # stopped or removed. Our own rank there is stopped later by its start_local, so its GPU use is not refused here.
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" bash -s <<<"$(declare -f die host_short ours_names refuse_foreign_serve refuse_gpu_busy)
RECIPE_LABEL=$(printf %q "$RECIPE_LABEL") CONTAINER_NAME=$(printf %q "$CONTAINER_NAME")
refuse_foreign_serve
grep -Fqx \"\$CONTAINER_NAME\" <<<\"\$(ours_names)\" || refuse_gpu_busy" ||
    die "$WORKER_HOST is not free for this recipe (above). Nothing was started or stopped."
}

take_lock() {
  # One start at a time on the head. The memguard does not inherit the lock (9>&- in start_memguard).
  mkdir -p "$STATE_DIR"
  exec 9>"$STATE_DIR/lock"
  flock -n 9 || die "Another ./run.sh is starting this recipe on $(host_short) (lock $STATE_DIR/lock)."
}

if [[ "$IMAGE_ONLY" == 1 ]]; then
  # Build the image and copy it to the worker, nothing else. Run it while another serve still holds the pair.
  command -v docker >/dev/null 2>&1 || die "docker not found"
  take_lock
  ensure_image
  if [[ "$ORCHESTRATE" == auto && "$ROLE" == head ]]; then
    require_worker_ssh
    sync_worker_image
  fi
  log "Image $IMAGE ready. Start the serve with ./run.sh."
  exit 0
fi

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  command -v docker >/dev/null 2>&1 || die "docker not found"
  take_lock
  refuse_foreign_serve
  refuse_busy_ports
  require_worker_ssh
  printf '%s\n' "$WORKER_HOST" >"$STATE_DIR/worker_host"
  # The worker's GPUs and containers, read-only, before anything long runs here.
  check_worker_free
  # The head's weights (downloaded when missing), then the worker's copy of them: a missing copy fails before the
  # image build, the image copy or any memory step, and the head already holds what the worker rsyncs.
  ensure_weights
  check_worker_snapshot
  # The image on both nodes, before any memory gate.
  ensure_image
  sync_worker_image
  stop_local
  # The head's weights, HCAs and memory gate, once, before rank 1 starts and waits on it.
  prepare_node
  # From here until rank 0 answers, any exit stops both ranks: rank 1 would otherwise hold its memory and wait.
  trap 'fail_stop "The start failed"' EXIT
  log "Starting rank 1 on $WORKER_HOST first"
  scp -q -o BatchMode=yes -o ConnectTimeout=5 "${BASH_SOURCE[0]}" "${WORKER_HOST}:/tmp/${CONTAINER_NAME}-run.sh"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "$(worker_env) bash /tmp/${CONTAINER_NAME}-run.sh"
  sleep 5
  state="$(worker_state)"
  [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
  start_local 0 prepared
  wait_ready 1
  trap - EXIT
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Rank 1 is up and waits for rank 0 at $HEAD_IP:$MASTER_PORT."
else
  take_lock
  refuse_busy_ports
  start_local 0
  wait_ready 0
  log "Stop with: ./stop.sh"
fi
