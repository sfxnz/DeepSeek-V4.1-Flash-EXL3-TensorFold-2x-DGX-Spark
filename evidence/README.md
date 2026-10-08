# evidence/

Receipts for every number in `README.md`. No file, no number. Never gitignore this directory.

- `recipe.yaml` names the file for each measured row. A row without one renders as "not yet measured on this pair", and `python3 kit/render.py` refuses a row that has a number but no file. `--strict` also fails on rows without a file.
- Commit raw output: logs, JSON, `free -h`, `docker inspect`. Never type a number into a file by hand. Large runs stay under the lab's data directory; copy the small verdict files here.

## Layout

One directory per session, `sK-<slug>/`, described in the session table below: date, result, setup, and how it differs from `run.sh`.

A gate session (`tools/session_gate.sh EVDIR`) writes:

| File | What |
|---|---|
| `gate.txt` | step exit codes, then `GATE=PASS` or `GATE=FAIL` |
| `health-before.json`, `health-after.json`, `health.code`, `models.json` | API state |
| `harness.sha256`, `git-head.txt` | sha256 of the clients, `run.sh`, `stop.sh` and `recipe.yaml`; the recipe commit |
| `serve-argv.txt`, `env-rank0.txt`, `env-rank1.txt` | image, argv and the full container env of both ranks |
| `free-before.txt`, `free-after.txt`, `free-*-<worker>.txt` | `free -h` on both nodes |
| `<step>.out`, `<step>.err`, `<step>.exit` | each smoke and bench |
| `docker-head.log`, `docker-<worker>.log`, `startup.txt`, `engine-needles.txt` | rank logs and the lines that matter |

## Ledgers

- `decision.tsv`: one row per choice. Columns `id	hypothesis	change	verdict	evidence	note`. Verdicts: `keep (default)`, `revert (noise)`, `revert (worse)`, `opt-in profile`, `documented option, not default`, `default (not measured)`, `pending measurement`.
- `trail.tsv`: one row per session. Columns `ts	phase	decision	why	evidence	result`; `ts` is UTC, e.g. `2026-10-04T16:25Z`.

## Sessions

| Session | What | Verdict |
|---|---|---|
| [`s0-engine-receipts`](s0-engine-receipts/) | the engine PR's final receipt run on this pair at `0d91389`, on the pinned weights | engine measured; not through `run.sh` |
| [`s1-snapshot-pins`](s1-snapshot-pins/) | the Hub at `982b704` behind `recipe.yaml`'s snapshot pins; `run.sh`'s snapshot check on the folder the receipts served | pins match |
| [`s2-run-sh-ec28f35`](s2-run-sh-ec28f35/) | engine bumped to `ec28f35`; the image and three serves through `run.sh`: decode table, sampled cells, exactness, prefill, needles to 1,039,528 tokens, quality | measured through `run.sh`; window default 1048576 |
| [`s3-default-1m`](s3-default-1m/) | `TOP_P=1.0` (`--top-p 1.0` on both ranks); two serves through `run.sh` at the shipped defaults: decode table, sampled cells with and without sampling fields, exactness of the default request, quiet prefill | default top_p 1.0; decode table at 1048576 |
| [`s4-device-nucleus`](s4-device-nucleus/) | engine bumped to `41306d5` (the device top_p cut); the image and one serve through `run.sh` at the shipped defaults: gate, sampled cells at top_p 1.0, 0.95 and 0.9, exactness at top_p 0.95 and 1.0 | top_p 0.95 and 0.9 on the device, 31.8 tok/s (was 15.5); every reply equal to `ec28f35`'s |
| [`s5-final`](s5-final/) | engine bumped to `903a1e8` (speed units C1-C5, C7, C8) and the engine's default draft policy; the image and two serves through `run.sh`: decode table, L.A.I.L, sampled cells, `bench_openai`, quiet prefill, exactness, a 1,039,528-token needle, the quality full set, NLL; the vLLM sibling the same day on the same cells | decode table and the same-session vLLM comparison |
| [`s6-concurrent`](s6-concurrent/) | engine bumped to `b86514a` (concurrent requests, units N0-N10) and the `PARALLEL` / `DECODE_SHARE` knobs; six counted `run.sh` boots in one session: `--parallel 1` against 4 in ABBA order, `--decode-share` 0.25 and 1, then the vLLM sibling on the same cells; then one gate boot at the shipped defaults | `PARALLEL=4` ships: the c=1 ruler within 0.5% of one lane (every c=1 cell within 1.31%), every concurrent reply equal to its solo run, MemAvailable at least 17.4 GiB on warm-cache boots; decode table at c=1, 2 and 4 |

### s0-engine-receipts (2026-10-04)

**Result.** TensorFold `0d91389` (PR #391's tip, the `deepseek_v41` family before the device draw) served the pinned weights on this pair. bench_decode at `--context 65538`, c=1: prose 45.5, structured 63.5, prose_long 34.0 tok/s (`s65538/bench_decode.log`). A second boot without `--context` admitted the native 1,048,576-token window and answered the smoke. This is the engine's receipt run, not a `run.sh` run: the recipe's image and launcher were not used.

The files are copies of the engine PR's final receipts (lab path `data/tf-dsv41/receipts/final/`), unchanged. Only the files the README cites are here. `launcher/run_rank.sh` is the dev launcher's container command (lab `data/tf-dsv41/dev/run_rank.sh`, unchanged, last modified 04:44Z, before the session). The pair script that called it (`dev/final/serve_pair.sh`) changed at 17:10Z, after the session, so it is not copied; `environment.txt` records the serve command it ran.

Setup (`environment.txt`):

- Engine: a worktree of `0d91389b1fe738dd24c49cc045977e2ac5324ea4`, the same tree on both nodes, installed editable (`pip install --no-deps -e .`) into the base image at each start. `s65538/commit.txt` is the commit the session ran from.
- Image: `nvcr.io/nvidia/pytorch:26.07-py3` at the digest this recipe's `docker/Dockerfile` builds FROM. torch `2.13.0a0+9186a08b2c.nv26.07`; TensorFold reports 0.6.4 (`serve_*_rank*.log`).
- Weights: `sfxnz/DeepSeek-V4.1-Flash-EXL3` revision `982b70452f399814f56b46272fd30394ae10d58c`, mounted read-only from the HF cache. The folder was the hand-built `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8` (named after the branch; its files link into `../2.0bpw-mcg/`), not the `snapshots/982b704…` folder `hf download` makes and `run.sh` reads. On the head that folder passes `run.sh`'s snapshot check with `recipe.yaml`'s pins (`s1-snapshot-pins/head_receipts_dir_check.txt`); its shards were not re-hashed. The lab's earlier file-by-file Hub comparison of both nodes (`data/tf-dsv41/receipts/q4/revision_check.*`, `revision_rank1_compare.txt`) is not copied.
- Link: `NCCL_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, master `10.100.8.1`, rank 1 on spark2.
- Serve, both ranks, rank 1 first: `--tp 2 --rank R --master 10.100.8.1 --master-port 29640 --name deepseek-ai/DeepSeek-V4.1-Flash --no-thinking --no-update-check --context 65538`; rank 0 adds `--host 127.0.0.1 --port 8080`. The default-window boot left out `--context`.
- The dev launcher's container (`launcher/run_rank.sh`): `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, `PYTHONUNBUFFERED=1`, `PYTHONPATH=/w/src:/w/tests/cuda`, the two NCCL variables, `--ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK`, a cgroup cap `--memory 110g --memory-swap 110g`, and one named volume, `tf-dsv41-ext`, for the kernel caches of every run it started. Rank 0 reports `loaded in 48.5s` (`serve_s65538_rank0.log`). It also sets `TF_DSV41_MODEL`, which the engine source at `0d91389` does not read (only tests do).

How it differs from `run.sh`:

- `run.sh` serves from an image built with `pip install` at the same commit (not editable).
- `run.sh` passes `--mtp-drafts 3` and `--max-tokens 4096` explicitly (the engine's and the CLI's defaults, the values this session ran with), port 8000, master port 29571, and rank 0 on 0.0.0.0 unless `BENCH_ONLY=1`.
- `run.sh` sets no `--memory` cgroup cap. It uses `--oom-score-adj 1000`, `--ulimit core=1`, `--init` and its memguard. The engine's admission reads `/proc/meminfo`, not the cgroup, so the cap does not change the admission lines.
- `run.sh` adds `HF_HUB_OFFLINE=1`, `NCCL_DEBUG=WARN`, and caches under `/cache/tf`, a fresh host directory per `TF_SHA`: its first boot compiles the extensions cold, which the receipts never timed.
- `run.sh` serves `snapshots/<SNAPSHOT_SHA>` from `hf download`, not the branch-named folder.

Files:

| File | What |
|---|---|
| `environment.txt` | commit, image, machines, link, checkpoint, serve command |
| `session_s65538.log`, `s65538/started.txt`, `s65538/finished.txt` | the session's steps and UTC times (16:25:12Z to 16:46:55Z) |
| `s65538/bench_decode.log` | the frozen ruler (the vLLM sibling's `bench_decode.py`, the same bytes as this repo's), `--concurrency 1 --phase all`, 3 runs |
| `s65538/smoke.txt`, `s65538/smoke.json` | `smoke_chat.py`: `323` |
| `free_before_s65538.txt`, `s65538/free_after.txt` | `free -h` on both nodes before the boot and after the session |
| `serve_s65538_rank0.log`, `serve_s65538_rank1.log` | both ranks' logs: admission, weights, one `done req-…` line per request |
| `free_before_default.txt`, `serve_default_rank*.log`, `default/smoke.*`, `free_after_default.txt` | the boot without `--context`: admission at 1,048,576 tokens, the smoke, `free -h` after |
| `launcher/run_rank.sh` | the dev launcher's `docker run` (container flags, env, mounts) |

Not here: the session also ran exactness pairs, tools, refusals, `bench_openai`, cold prefill, the NCCL two-rank check and the quality set. The README does not cite them, so they stay in the lab's receipt directory.

### s1-snapshot-pins (2026-10-04)

**Result.** The Hub at revision `982b70452f399814f56b46272fd30394ae10d58c` gives the pins in `recipe.yaml` (`model`): `config.json` sha256 `6469adab…`, index sha256 `91731e4a…`, 48 shards in the index, 357,466,041,064 shard bytes (`pins.json`). All 56 files hold 357,486,881,279 bytes (333 GiB). `run.sh`'s `check_snapshot` passes with those pins on the folder the receipts served on the head (`head_receipts_dir_check.txt`). Read-only: no serve, no GPU.

| File | What |
|---|---|
| `listing.py` | the script: `HfApi().model_info(…, files_metadata=True)` at the revision, and `hf_hub_download` of `config.json` and the index into a scratch dir to hash them |
| `pins.json` | its output: the revision the Hub resolved, both sha256, shard count, shard bytes, all-file bytes, UTC time, `huggingface_hub` version |
| `hub_files.tsv` | every file at the revision: path, bytes, LFS sha256, git blob id |
| `head_receipts_dir_check.txt` | `check_snapshot` from `run.sh` on spark1's `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8`, rc=0 |

Not checked here: spark2's folder (no ssh in this session), and the shard bytes against their LFS sha256 (not re-hashed).

### s2-run-sh-ec28f35 (2026-10-04)

**Result.** TensorFold `ec28f35` (v0.6.4 + PRs #390 and #391 + the device keyed draw for top-k-off rows) built into `tf-dsv41-flash:0.6.4-ec28f35` by `IMAGE_ONLY=1 ./run.sh` and served by `./run.sh` on this pair. Decode, median of two boots at `--context 65538`: prose 45.7, structured 64.2, prose_long 34.4 tok/s (`decode.txt`). The native window, 1048576, booted through `run.sh`, ran the ruler (46.2 / 64.9 / 34.8) and found needles at 130k to 1,039,528 prompt tokens, with MemAvailable at least 22.1 GiB on the head and 26.7 GiB on the worker: it is now the default. Quality and NLL equal the `0d91389` receipts exactly. Default sampling (top_p 0.95, top-k off) is unchanged at about 24 tok/s; top_p 1.0 now decodes like top_k 20.

Setup: spark1 head, spark2 worker, both GPUs exclusive (only `conduit` ran on spark1, no GPU); `run.sh` defaults with no env override except `CONTEXT=1048576` for the third boot; weights from `snapshots/982b704…` on both nodes (`run.sh`'s snapshot check passed on both at each boot, `*/run.txt`). The client tools ran on the head's host: the recipe's own (`smoke_chat.py`, `smoke_count.py`, `bench_decode.py`, `tools/session_gate.sh` with RUNS=9 RUNS_LONG=9), the scripts in `scripts/`, the engine's `tools/bench_openai.py` and `tools/prefill_cold.py` from a checkout of `ec28f35` in the lab's host venv (the image does not ship `tools/`), and the vLLM sibling's `tests/quality_eval.py` through the lab's `data/tf-dsv41/dev/q3/qe_tf.py`. The lab's local-ai-lab app polled `GET /v1/models` on port 8000 throughout (no inference).

Order (UTC): validate 18:49; image 18:49-18:53; boot A 18:53 (cold kernel cache), gate, pairs, sampled cells, bench_openai, prefill_cold, stop 19:19; boot B 19:19, gate, sampled cells, stop 19:26; boot `CONTEXT=1048576` 19:26, gate, needles 19:30-20:39, quality 20:40-20:52, stop 20:52; NLL 20:52-20:54 (its own two containers from the recipe image, no server); both GPUs idle at 20:54 (`gpus_idle_end.txt`).

| File | What |
|---|---|
| `step1-validate.txt` | unit tests, `render --check`, `VALIDATE_ONLY=1` and `=args`, shellcheck, after the pin bump |
| `image-only.txt`, `image-ids.txt` | `IMAGE_ONLY=1 ./run.sh`: the build (pip's commit check, the unchanged-freeze check), the copy; the image ID and labels on both nodes |
| `bootA/`, `bootB/` | `run.txt` (`./run.sh` output), `gate/` (`tools/session_gate.sh`: smokes, the frozen ruler, receipts), `sampled-*.txt`, `rank*-full.log`, `free-*.txt`, `stop.txt` |
| `derived.txt` | `scripts/derived.py`: each README number that no file holds literally (MiB to GiB, rounding, durations, the gaps to vLLM, the admission reserve, the `thinking=False` count), with its inputs |
| `decode.txt` | `scripts/decode_compute.py` over both boots' `gate/bench-*.out`: the README rows, per-boot and pooled medians |
| `bootA/pairs.jsonl`, `bootA/pairs_vs_s0.txt` | `scripts/pairs.py`: drafted vs `"draft": false`; the 15 requests the s0 receipts also sent, hash by hash |
| `bootA/bench_openai.*`, `bootA/prefill_cold.*` | the engine's tools at `ec28f35`; prefill prompts `data/tf-dsv41/receipts/final/prompts.json` (the s0 set, sha256 in the log) |
| `ctx1m/` | the `CONTEXT=1048576` boot: `run.txt`, `gate/`, `needles.jsonl` (`scripts/needles.sh`, `scripts/long_needle.py`), `memwatch.tsv` (`scripts/memwatch.sh`, both nodes every 5 s, 19:26-20:52), `quality_quick.*`, `quality_vs_s0.txt` |
| `nll/` | `scripts/nll_pair.sh` (the lab's `dev/final/nll_tf.py` on both ranks in the recipe image), `nll_compare.txt` (`scripts/nll_compare.py`: mean NLL and a bitwise comparison with s0's 80 G1 score files); the score files stay in `data/tf-dsv41/receipts/s2-recipe/tf_scores` |
| `vllm-sibling/` | copies of the vLLM numbers the README compares with, and of s0's `bench_openai` / `prefill_cold` JSON; `SOURCES.txt` names each source and its sha256 |
| `gpus_idle_end.txt` | containers, GPU use and `free -h` on both nodes at the end |

s2's `gate/harness.sha256` files hash the clients only, and `git-head.txt` reads `HEAD` (nothing was committed): the gate added `run.sh`, `stop.sh` and `recipe.yaml` after s2. The launcher's provenance for s2 is each boot's `run.txt` (`./run.sh` output) and `gate/serve-argv.txt` / `env-rank*.txt`.

Not here: a `0d91389` boot through `run.sh` (s0 is the dev launcher), so the top_p 1.0 speed before the device draw is not measured; c=2 (the family serves one request at a time); MMLU and the 128k quality needles of the full set.

### s3-default-1m (2026-10-04)

**Result.** `run.sh` now passes `--top-p 1.0` to both ranks (`TOP_P=1.0` in `recipe.yaml`; the engine's default is 0.95). Rank 0 logs `sampling: top_p 1.0` (`bootC/gate/startup.txt`); a request that omits `temperature` gets the engine's 1.0 (`src/tensorfold/cuda/server.py:92` at `ec28f35`), and the family keeps top-k off. Two boots at the shipped defaults (`--context 1048576`, `--top-p 1.0`), median of the per-boot medians: prose 46.0, structured 64.1, prose_long 34.6 tok/s (`decode.txt`). A request with no sampling field decodes at 36.76 / 37.10 tok/s (boots C / D; s2's server default top_p 0.95 gave 15.3 / 15.4), next to an explicit top_p 1.0 at 37.03 / 37.27, top_k 20 at 37.12 / 37.63 and top_p 0.95 at 15.23 / 15.72. On boot D the default request equals `"draft": false` on 3 of 3 prompts and reproduces s2's explicit top_p 1.0 token hashes. Quiet `prefill_cold`: 526 / 551.7 / 561.3 / 563.5 / 570.6 prompt tok/s at 2k / 8k / 16k / 32k / 64k.

Why: DeepSeek's model card recommends temperature 1.0 and top_p "0.95 or 1.0" (`model-card-sampling.txt`, the card's lines 168-173; its evaluations ran at 0.95). At `ec28f35` a top-k-off row with no top_p cut draws on the device; a cut reads each rank's vocabulary half to the host. Decision `TOPP1` in `../decision.tsv`.

Setup: as s2 (spark1 head, spark2 worker, both GPUs exclusive, only `conduit` on spark1, no GPU), the image `tf-dsv41-flash:0.6.4-ec28f35` already on both nodes, the warm kernel cache. `./run.sh` with no env override for both boots, through `scripts/boot.sh` (one boot: `free -h`, `./run.sh`, the gate with RUNS=9 RUNS_LONG=9, the cells, the pairs, on boot D `prefill_cold`, both ranks' logs, `./stop.sh`). The client tools ran on the head's host: the recipe's own, s2's `scripts/sampled_cell.py` (unchanged), `scripts/pairs_default.py`, and the engine's `tools/prefill_cold.py` from the host worktree of `ec28f35` in the lab's host venv, with s2's prompts (`data/tf-dsv41/receipts/final/prompts.json`, sha256 in the log). The lab's local-ai-lab app polled `GET /v1/models` throughout (no inference; `bootD/prefill-clients.txt`).

Order (UTC): validate 21:11; boot C 21:11:58, ready 21:12:50, gate, cells, stop 21:19:41; boot D 21:20:08, ready 21:20:59, gate, cells, pairs, `prefill_cold` 21:28:13-21:39:17, stop 21:39:18; both GPUs idle at 21:39:41 (`gpus_idle_end.txt`).

| File | What |
|---|---|
| `step1-validate.txt` | unit tests, `render --check` and `--strict --check`, `VALIDATE_ONLY=1` and `=args` (both ranks with `--top-p 1.0`), shellcheck, after the `TOP_P` change |
| `model-card-sampling.txt` | the model card's "Recommended sampling parameters" lines and its top_p evaluation lines, with the card's sha256 |
| `bootC/`, `bootD/` | `run.txt` (`./run.sh` output), `gate/` (`tools/session_gate.sh`: smokes, the frozen ruler, receipts), `sampled-{nofields,top_p1,top_k20,top_p095}.txt`, `rank*-full.log`, `free-*.txt`, `stop.txt` |
| `bootD/pairs.jsonl` | `scripts/pairs_default.py`: no sampling field, seed 1234, 160 tokens, drafted vs `"draft": false`, and each hash against s2's explicit top_p 1.0 reply |
| `bootC/pairs-client-error.txt` | boot C's pairs run: the script failed to parse s2's file (its trailing `rc=0` line) before sending any request; fixed, run on boot D |
| `bootD/prefill_cold.*`, `bootD/prefill-clients.txt` | the quiet prefill run; the API's connections and the head's GPU processes just before it |
| `decode.txt` | `scripts/decode_compute.py` (s2's with boots C and D) over both boots' `gate/bench-*.out`: the README rows, per-boot and pooled medians |
| `derived.txt` | `scripts/derived.py`: the serving line's sampling, the sampled cells and their medians, the pairs checks, the gaps to vLLM (s2's `bench_openai`, this session's ruler and prefill), the quiet-prefill check, boot durations, the `thinking=False` count |
| `bootC-driver.txt`, `bootD-driver.txt` | `scripts/boot.sh`'s own output |
| `gpus_idle_end.txt` | containers, GPU use and `free -h` on both nodes at the end |
| `final-validate.txt` | the same checks after the README, `recipe.yaml` and ledger updates |

Not here: `bench_openai` (not rerun: at t=1 it sends top_k 20 and top_p 0.95, so the server's top_p does not reach it; the README keeps s2's), needles, quality and NLL (not rerun; s2's 1048576 boot at the engine default top_p 0.95 holds them).

### s4-device-nucleus (2026-10-05)

**Result.** TensorFold `41306d5` (fork branch `dsv41-recipe-engine2`: `ec28f35`'s tree, which is `44338a2`, plus `41306d5`, the patch of `43ebdce` on branch `cuda-keyed-draw`: a top_p cut found on the device) built into `tf-dsv41-flash:0.6.4-41306d5` by `IMAGE_ONLY=1 ./run.sh` (208 s; pip changed nothing else) and served once by `./run.sh` at the shipped defaults. Gate: `GATE=PASS`; the ruler, one boot: prose 45.26, structured 63.74, prose_long 34.18 tok/s (s3's two-boot rows 46.0 / 64.1 / 34.6; the decode table stays s3's). Sampled cells (one boot, 9 runs, median): no field 36.59, top_p 1.0 36.55, top_k 20 37.04, top_p 0.95 31.76 (s3: 15.23 / 15.72), top_p 0.9 31.76 tok/s. Each cell's reply has the same token hash as the same request in both s3 boots. The top_p 0.95 and 0.9 reply is one text (`dae740142762`) that takes 99 rounds against the top_p 1.0 reply's 86; scaled by rounds the three cells match (36.56 / 36.56 / 36.55). Pairs at temperature 1.0, top_k omitted, seed 1234, 160 tokens, three prompts: drafted == `"draft": false` 6 of 6, each drafted reply equal to s2's reply to the same request at `ec28f35` 6 of 6 (top_p 0.95: the host path then); the sampling stage fell from about 68 ms a round (s2) to 3.4-3.8 ms at top_p 0.95, and those requests decoded at 47.0-50.1 tok/s against 23.2-24.8 in the same rounds (`derived.txt`).

Why: at `ec28f35` a top_k-off row with a top_p cut whose nucleus ran past each rank's 1,024 best candidates read each rank's vocabulary half to the host. The engine commit finds the cut by radix passes over exact int64 masses on the device, so it draws the host rule's tokens.

Setup: as s3 (spark1 head, spark2 worker, both GPUs exclusive, only `conduit` on spark1, no GPU). The bump to `41306d5` in `recipe.yaml`, `docker/Dockerfile`, the render and the test constants was uncommitted during the boot (`gate/git-head.txt` is the previous commit `5cbb262`); `run.sh` already had its committed bytes (its sha256 in `gate/harness.sha256`). `recipe.yaml` changed afterwards in comments and the decode conditions only, so its hash in `gate/harness.sha256` is the pre-edit one. A fresh kernel cache for the new `TF_SHA`: the boot compiled the extensions (182 s to ready, rank 0 `loaded in 169.2s`). `./run.sh` with no env override, through `scripts/boot.sh` (`free -h`, `./run.sh`, the gate with RUNS=9 RUNS_LONG=9, the five cells, the pairs, both ranks' logs, `free -h`, `./stop.sh`). Client tools on the head's host: the recipe's own, `scripts/sampled_cell.py` (s2's, plus each reply's `token_sha` from the stream) and `scripts/pairs_nucleus.py`. Rank 0 logged 89 `POST /v1/chat/completions` and 89 `done` lines: the gate's 32, the cells' 45 and the pairs' 12; nothing else sent inference (the `GET /v1/models` lines are polls).

Order (UTC): validate 02:46; image 03:24:23-03:27:51; boot 03:28:01, ready 03:31:03, gate to 03:33:35, cells, pairs, stop 03:39:12; both GPUs idle at 03:39:30 (`gpus_idle_end.txt`).

| File | What |
|---|---|
| `step1-validate.txt` | unit tests, `render --check` and `--strict --check`, `VALIDATE_ONLY=1` and `=args`, shellcheck, after the pin bump |
| `image-only.txt`, `image-ids.txt` | `IMAGE_ONLY=1 ./run.sh`: the build (pip's commit check, the unchanged-freeze check), the copy; the image ID and labels on both nodes |
| `run.txt`, `gate/`, `stop.txt`, `free-*.txt`, `rank*-full.log` | `./run.sh` output, `tools/session_gate.sh` (smokes, the frozen ruler, receipts), `./stop.sh`, `free -h` on both nodes, both ranks' full logs |
| `sampled-{nofields,top_p1,top_k20,top_p095,top_p09}.txt` | `scripts/sampled_cell.py`: s3's four cells and top_p 0.9, 9 runs each, with token hashes |
| `pairs.jsonl` | `scripts/pairs_nucleus.py`: drafted vs `"draft": false` at top_p 0.95 and 1.0 on s2's three prompts, and each hash against s2's reply at `ec28f35` |
| `derived.txt` | `scripts/derived.py`: the serving line, the ruler against s3, the cells against s3 (speeds, token hashes from s3's `rank0-full.log`, text heads), rounds of each reply, the pair checks and the before/after per request, durations, the `thinking=False` count |
| `boot-driver.txt` | `scripts/boot.sh`'s own output |
| `gpus_idle_end.txt` | containers, GPU use and `free -h` on both nodes at the end |
| `final-validate.txt` | the same checks as `step1-validate.txt` after the README, `recipe.yaml`, `AGENTS.md` and ledger updates |

Not here: a second boot (the decode table and its two-boot rule stay with s3; the ruler is greedy and this commit changes only sampling with a top_p cut); `bench_openai`, `prefill_cold`, needles, quality and NLL (not rerun: the commit does not touch prompts, logits or top_k 20 sampling); rows where the device cut falls back to the host rule (past `TMAX`, or where the draw cannot decide) are not counted in the logs.

### s5-final (2026-10-06)

**Result.** TensorFold `903a1e8` (fork branch `dsv41-recipe-engine3`: `41306d5`'s tree, which `db7f53e` has, plus the speed units C1, C3, C2, C7, C4, C5 and C8, merged in that order after each unit's receipts in the lab's `data/tf-dsv41/receipts/c1..c8`) built into `tf-dsv41-flash:0.6.4-903a1e8` by `IMAGE_ONLY=1 ./run.sh` (214 s, pip changed nothing else, the same image ID on both nodes). `run.sh` now passes `--mtp-drafts 5 --mtp-confidence 0.15`, the engine's default policy, instead of `--mtp-drafts 3` (decision `DRAFTS`, from the C5 receipts copied to `c5-draft-policy/`). Two boots at the shipped defaults, median of the per-boot medians: ruler prose 55.81, structured 97.25, prose_long 40.68 tok/s (`decode.txt`; s3 at `ec28f35`: 46.0 / 64.1 / 34.6); L.A.I.L 41.23; sampled no field 42.77, top_p 0.95 37.70, top_k 20 42.55, each reply equal to s4's; `prefill_cold` 1146.5 / 1542.5 / 1684.8 / 1709.0 / 1790.1 prompt tok/s at 2k / 8k / 16k / 32k / 64k. Exactness, each boot: pairs drafted == `"draft": false` 6 of 6 and equal to s2's replies 6 of 6; `bench_concurrent --alone --serial` 48 of 48 and 10 of 10. Boot F: the 1,039,528-token needle prefilled in 754.3 s and found (s2: 2077.8 s), MemAvailable at least 18.1 / 25.2 GiB during it and 15.0 / 23.7 GiB over the boot; the quality full set passes the round-36 gate (GSM8K 97/100, with thinking 39/40, MMLU 204/228, tools 22/22 / 22/22 / 8/8, needles 9/9, selfcons 12/12, golden hazard 0.01307). NLL 0.137069, its repeat equal, 80 of 80 score files bit-identical to s2's. The vLLM sibling, booted the same day with its canonical command and the same clients: TF/vLLM prose 1.165, structured 1.178, prose_long 0.939, L.A.I.L 1.026, sampled 1.017 / 0.894 / 0.979, `bench_openai` t=1 0.973 / 0.962, the t=0 chat reply 0.876, prefill 1.446-2.353 (`derived.txt`).

Why: upstream froze its Python engine at 0.6.5 (ashhart/TensorFold issue 286) and closed PRs 390, 391 and 408; the maintainer rewrites the useful parts in Zig with credit. The fork carries the speed units, so the recipe pins the fork. Two knobs moved in this session, the engine and the draft policy: the policy's A/B is the C5 receipts' (four dev boots, A B B A, plus a d=3 / default sweep forward and reversed inside the B boots), not a `run.sh` A/B; s5 measures the shipped combination.

Setup: spark1 head, spark2 worker, both GPUs exclusive (only `conduit` on spark1, no GPU). The bump was committed (`13a26f7`) before the boots, so each `gate/git-head.txt` names it and `gate/harness.sha256` hashes its `run.sh` and `recipe.yaml`. Afterwards `run.sh` changed outside the generated block (`MIN_MEM_GATE_GIB` 92 -> 94, from boot E's admission line: 81.32 + 12.1 GiB) and `recipe.yaml` in comments and the measured rows. `./run.sh` with no env override, through `scripts/boot.sh` (`free -h`, `./run.sh`, the gate with RUNS=9 RUNS_LONG=9, L.A.I.L, the cells, `bench_openai`, `bench_t0_chat.py`, the pairs, `bench_concurrent`, quiet `prefill_cold`; with `long`: `memwatch.sh` from before the boot to the stop, the needle, the quality set; both ranks' logs, `free -h`, `./stop.sh`). Client tools on the head's host: the recipe's own; the vLLM recipe's `tools/measure_lail_prose.py` and `tests/quality_eval.py` (through the lab's `dev/q3/qe_tf.py`), unchanged; the engine's `tools/bench_openai.py`, `bench_concurrent.py` and `prefill_cold.py` from the host checkout at `903a1e8` (clean) in the lab's host venv; s4's `sampled_cell.py` and `pairs_nucleus.py`, s2's `long_needle.py` and `memwatch.sh`, the oracle session's `bench_t0_chat.py`, all unchanged. The lab's local-ai-lab app polled `GET /v1/models` throughout (no inference; `*/prefill-clients.txt`). NLL: `scripts/nll_pair.sh`, s2's with the s5 image and a `git archive` of `903a1e8` mounted at `/w` on both nodes (spark2 has no TensorFold checkout; `nll/nll_tree.txt`); the lab's `dev/final/nll_tf.py` (the lab's `dev/nll_tf.py` imports the score module from the package, which `7032f26` moved to `tests/cuda/`). vLLM: `scripts/vllm.sh` in the vLLM recipe's clean main checkout (`3de9146`), `env AUDIT=strict ./run.sh` (`audit ok` on head and worker), the ruler with the gate's exact commands, L.A.I.L, the cells, `bench_openai` at t=1 (its t=0 pass fails on vLLM) and `bench_t0_chat.py`, quiet `prefill_cold` on the oracle session's `G7_prefill_prompts.json`, then `docker stop -t 60` and `docker rm` on both nodes (its `stop.sh` uses `docker rm -f`).

Order (UTC): validate 16:23:46; image 16:24:24-16:27:58; boot E 16:29:41, ready 16:33:09, cells, stop 16:51:51; boot F 17:00:16, ready 17:01:22, cells to 17:19:16, needle 17:19:21-17:31:57, quality 17:31:58-17:48:16, stop 17:48:17; NLL 17:52:27-17:54:09 (`nll/nll_started.txt`, `nll_finished.txt`); vLLM 17:54:26, ready 18:05:08, cells, stop 18:20:40-18:21:13; both GPUs idle at 18:21:38 (`gpus_idle_end.txt`).

| File | What |
|---|---|
| `step1-validate.txt`, `final-validate.txt` | unit tests, `render --check` and `--strict --check`, `VALIDATE_ONLY=1` and `=args`, shellcheck: after the pin bump, and after the README, `recipe.yaml`, `AGENTS.md`, `run.sh` floor and ledger updates |
| `image-only.txt`, `image-ids.txt` | `IMAGE_ONLY=1 ./run.sh` (the build, pip's commit check, the unchanged-freeze check, the copy); the image ID and labels on both nodes |
| `c5-draft-policy/` | copies of the C5 unit receipts behind decision `DRAFTS`: `abba_table.txt`, `abba.log`, the B boots' sweeps, the dev scripts, each boot's commit; `SOURCES.txt` with sha256 |
| `bootE/`, `bootF/` | `run.txt`, `gate/` (`tools/session_gate.sh`), `lail.log`, `sampled-{nofields,top_p095,top_k20}.txt`, `bench_openai.*`, `bench_t0_chat.*`, `pairs.jsonl`, `bench_concurrent.*`, `prefill_cold.*`, `prefill-clients.txt`, `rank*-full.log`, `free-*.txt`, `steps.txt`, `stop.txt` |
| `bootF/needle.jsonl`, `bootF/memwatch.tsv`, `bootF/quality_full.*` | the needle, MemAvailable / SwapFree on both nodes every 5 s over boot F, the quality full set |
| `nll/` | `nll_tf_rank*.log`, `nll_compare.txt` (`scripts/nll_compare.py`: mean NLL, repeat, bitwise against s2's 80 score files); the score files stay in `data/tf-dsv41/receipts/s5-recipe/tf_scores` |
| `vllm/` | the vLLM boot: `run.txt`, `vllm-git.txt`, `serve-argv.txt`, `harness.sha256`, `smoke-chat.*`, `bench-frozen.*`, `bench-prose-long.*`, `lail.log`, `sampled-*.txt`, `bench_openai.*`, `bench_t0_chat.*`, `prefill_cold.*`, `prefill-clients.txt`, `metrics-end.txt`, both containers' logs, `stop.txt` |
| `decode.txt` | `scripts/decode_compute.py` (s3's with boots E and F): the README rows, per-boot and pooled medians |
| `derived.txt` | `scripts/derived.py`: argv and admission, every cell per boot and its two-boot row, TF / vLLM ratios, token hashes against s4, exactness, needle and memory minima, quality, NLL, durations, the `thinking=False` count |
| `*-driver.txt` | the drivers' own output |
| `gpus_idle_end.txt` | containers, GPU compute apps and use, `free -h` on both nodes at the end |

Not here: c=2 (the family serves one request at a time); a `run.sh` A/B of the draft policy (the C5 receipts hold it, on the dev launcher at `--context 65538`); needles between 128k and 1M at `903a1e8` (s2's at `ec28f35` stand); the quality set's `nll`, `decode`, `c2` and `vision` components (`prompt_logprobs`, two streams, images).

### s6-concurrent (2026-10-08)

**Result.** TensorFold `b86514a` (fork branch `dsv41-recipe-engine4`: `903a1e8`'s tree plus concurrent requests, units N0-N10, each merged after its receipts in the lab's `data/tf-dsv41/receipts/n0..n10`) built into `tf-dsv41-flash:0.6.4-b86514a` by `IMAGE_ONLY=1 ./run.sh` (the same image ID on both nodes, `1A/image.txt`). `run.sh` passes `--parallel 4 --decode-share 0.5` (`PARALLEL`, `DECODE_SHARE`). In one session, A = `PARALLEL=1` and B = the defaults, both at `CONTEXT=1048576`, booted 1A 2B 3B 4A. c=1, B against A (gate: within 3%): ruler -0.39% / -0.33% / -0.43%, cold prefill 2k-64k -1.31% to +0.66%: pass. B, median of the per-boot medians: prose 56.3 / 82.2 / 115.7 tok/s aggregate at c=1 / 2 / 4, structured 97.7 / 144.0 / 200.7, prose_long 41.2 / 61.0 / 87.1 (`decode.txt`). The vLLM sibling the same day (7V, `MAX_NUM_SEQS=2`): prose 49.3 / 81.3 / 102.1, structured 84.0 / 155.9 / 201.8, prose_long 46.2 / 69.4 / 87.6 (`derived.txt`). Every concurrent reply equals its solo run: `bench_concurrent` 102 of 102 a boot, `pairs_concurrent` 4 of 4 against both A boots, 68 of 68 TTFT-cell streams. MemAvailable at least 17.42 GiB on the head and 21.90 on the worker at 4 x 1048576 (`startup estimate 82.55 GiB`) over the warm-cache boots; 13.01 / 14.74 GiB during W1's first start, while the kernels compiled (`W1/memwatch.tsv`). So `PARALLEL=4` ships (the orchestrator's rule: 4 if the memory floors hold at 4 windows and c=1 holds, else 2).

Why: before `b86514a` the family served one request at a time and ignored `--parallel`; a second request waited for the first. The design and the unit receipts are in the lab's `data/tf-dsv41/design/conc-design.md` and `receipts/n0..n10`.

Setup: spark1 head, spark2 worker, both GPUs exclusive (only `conduit` on spark1, no GPU; the Qwen serve was already stopped, `previous_serve_before.txt`, and stayed stopped at the user's request, so nothing was restored, `restore_log.txt`). The pin bump and knobs were committed (`b55aac8`) before the boots, so each `gate/git-head.txt` names it. The image was built before the session (`image-only.txt`; `step1-validate.txt` holds the checks after the bump). Every TF boot ran `./run.sh` through `scripts/boot.sh` with one env override, `EXTRA_ENV=PYTHONPATH=/cache/tf/n11probe`: it loads `scripts/n11probe/usercustomize.py`, which only logs memory and timing around graph warm-up (`<boot>/startup.txt`), since the engine logs neither. Two warm-up boots, W1 (`PARALLEL=4`) and W2 (`PARALLEL=1`), filled the new kernel cache first and are not counted. `scripts/boot.sh` was edited while W1 ran, so bash stopped after W1's `prefill_cold` and W1 was stopped by hand (`W1.driver.txt`, `W1/steps.txt`); the scripts were frozen before W2 (`scripts.sha256`). Only `scripts/table.py` changed after the session (`scripts-final.sha256`). The vLLM sibling: `env AUDIT=strict ./run.sh` in its clean checkout at `3de9146` (`7V/vllm-git.txt`), `audit ok` on both ranks, stopped with `docker stop -t 60` and `docker rm` on both nodes (`7V/stop.txt`).

Order (UTC, `session.log`): W1 02:50-03:22; W2 03:22-03:44; 1A 03:44-04:16; 2B 04:16-05:13 (with the needle 04:49-05:12); 3B 05:13-05:46; 4A 05:46-06:19; 5C (`DECODE_SHARE=0.25`) 06:19-06:25; 6D (`DECODE_SHARE=1`) 06:25-06:30; 7V (vLLM) 06:31-06:58; both GPUs idle at 06:58:25 (`end_state.txt`).

| File | What |
|---|---|
| `step1-validate.txt`, `image-only.txt` | the checks after the pin bump (tests, `render --check`, `VALIDATE_ONLY=1` and `=args`, shellcheck) and `IMAGE_ONLY=1 ./run.sh` (the build, pip's commit check, the unchanged-freeze check, the copy) |
| `abba_table.txt` | `scripts/table.py` over every boot: the c=1 gate, the concurrent aggregates against vLLM and the A arm, exactness, the TTFT cells, memory and the probe, the gate floor |
| `decode.txt` | `scripts/decode_compute.py` over boots 2B and 3B: the README rows, each boot's medians |
| `derived.txt` | `scripts/derived.py`: TF against vLLM per cell at c=1, 2 and 4 (per-stream decode, aggregate, TTFT p50, inter-chunk p50 / p90), the aggregate gain over c=1 |
| `1A/`, `4A/` (`PARALLEL=1`), `2B/`, `3B/` (defaults) | `run.txt`, `gate/` (`tools/session_gate.sh`, RUNS=9 RUNS_LONG=9: smokes, the ruler at c=1, 2 and 4, `bench_concurrent` plain and mixed, `conc-check`, `pairs_concurrent`), `sampled-{nofields,top_p095}.txt` (`scripts/sampled_conc.py`; c=1 only on A), `pairs_nucleus.jsonl` (s5's pairs), `pairs_q2.jsonl` (the c7 receipts' pairs), `prefill_cold.*`, `prefill-clients.txt`, `memwatch.tsv` (both nodes every 2 s), `startup.txt` (admission and the probe), `rank*-full.log`, `free-*.txt`, `steps.txt`, `stop.txt`; A: `ttft_alone.jsonl` (each TTFT request alone); B: `ttft_cells.jsonl` (`scripts/ttft.py`) |
| `2B/needle_lanes.jsonl`, `2B/free-*-needle.txt` | `scripts/needle_lanes.py`: the 1,039,528-token needle while 3 lanes decode, every live reply |
| `5C/`, `6D/` | `--decode-share` 0.25 and 1: `ttft_cells.jsonl`, `pairs_concurrent.jsonl`, `memwatch.tsv`, `startup.txt`, logs |
| `7V/` | the vLLM boot: `run.txt`, `vllm-git.txt`, `serve-argv.txt`, `harness.sha256`, `smoke-chat.*`, `bench-frozen.*` and `bench-prose-long.*` (c=1, 2, 4), `sampled-*.out`, `metrics-end.txt`, both containers' logs, `memwatch.tsv`, `stop.txt` |
| `W1/`, `W2/`, `W1.driver.txt` | the warm-up boots (not counted) |
| `scripts/` | the session's scripts (`session.sh`, `boot.sh`, `vllm.sh`, `memwatch.sh`, the clients, `n11probe/`), `table.py`, and N12b's `decode_compute.py` and `derived.py` |
| `session.log`, `session.driver.txt`, `previous_serve_before.txt`, `restore_log.txt`, `end_state.txt` | the session's steps, the drivers' output, the state before and after |

Not here: c above 4 (`--parallel` takes 1 to 4; a fifth request waits for a lane, not measured); a client disconnecting while its prompt fills (documented, not measured); L.A.I.L, `bench_openai`, quality and NLL at `b86514a` (s5's at `903a1e8` stand: the units change how requests share a forward, and every reply here equals its solo run and s5's hashes); `prefill_s` under lanes (the stats field counts only span launches: the needle read 48.4 s of 1370.6 s, `abba_table.txt`).
