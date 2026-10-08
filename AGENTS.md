# AGENTS.md — DeepSeek-V4.1-Flash EXL3 · TensorFold · 2× DGX Spark

Serve `sfxnz/DeepSeek-V4.1-Flash-EXL3` (revision `982b704`, branch `2.0bpw-mcg-viterbi-lmhead-mxfp8`) at TP=2 with TensorFold's own `deepseek_v41` CUDA family: commit `b86514a` on `sfxnz/TensorFold` branch `dsv41-recipe-engine4` (v0.6.4 + the family of upstream PRs #390 and #391 + the device keyed draw for top_k-off rows + the speed units C1-C5, C7, C8 + concurrent requests, units N0-N10; upstream froze its Python engine, issue #286, so the recipe pins the fork), inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned in `docker/Dockerfile`). Not vLLM. The vLLM recipe is a separate repo (`sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`); one repo per engine.

Humans read [README.md](README.md).

## Standing orders

1. **PR only.** Commit on `agent/**` branches. Open a PR against `main`. Never push `main`. Never merge. Never force-push.
2. **Evidence.** Every number in `README.md` has a file under `evidence/` that proves it. No file, no number. Never gitignore `evidence/`. `recipe.yaml` names the file per measured row. A row without one renders as "not yet measured on this pair", and `kit/render.py` refuses a number without one.
3. **Exclusive GPUs.** One serve per pair. Before a boot, stop any other GPU serve on both Sparks with its own stop script (gracefully: `docker stop`, not `docker rm -f`). Restore it afterwards and run its smokes. `run.sh` refuses to start next to a foreign GPU container and never removes it. `run.sh` and `stop.sh` stop only containers named `tf-dsv41-*` that carry the label `ai-lab.recipe=dsv41-tensorfold`.
4. **Validate first.** `VALIDATE_ONLY=1 ./run.sh` before any serve. It needs no Docker.
5. **Source of truth.** Values live in `recipe.yaml`. Edit it and run `python3 kit/render.py`. Never edit inside the generated markers by hand.
6. **Voice.** Short sentences. Numbers first. No marketing language. No emojis.
7. **Secrets.** Never commit tokens, Tailscale IPs, or auth headers.

## Working rules

- **One engine.** This recipe serves TensorFold's own family only. No patches, no vendored launcher code, no other packs. To change the engine, bump `TF_REPO` + `TF_SHA` in `recipe.yaml` (the fork while upstream's Python engine stays frozen, issue #286), render, rebuild with `IMAGE_ONLY=1 ./run.sh`, and measure again.
- **Window.** `CONTEXT=1048576`, the native window: s2 ran the bench and needles up to 1,039,528 prompt tokens through `run.sh` at it (`evidence/s2-run-sh-ec28f35/ctx1m`); s5 ran the 1,039,528-token needle again at `903a1e8` (`evidence/s5-final/bootF/needle.jsonl`). Do not advertise a window that was not run through `run.sh`.
- **One knob per session.** Change one knob at a time against the gate (`tools/session_gate.sh EVDIR`). Final numbers come from two boots: report the median of the per-boot medians.
- **Frozen ruler.** `bench_decode.py` is byte-identical to the vLLM sibling's frozen ruler (sha256 `3172cbc4…`, enforced by `tests/`). Do not edit it. Its `/metrics` acceptance fields stay empty on this engine; drafting shows in rank 0's `done req-…` log lines. `tools/bench_concurrent.py` is TensorFold's at `TF_SHA`, byte-identical too (enforced by `tests/`); copy it again on an engine bump.
- **Memory.** Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- **HCAs.** Pin `NCCL_IB_HCA`. Two Sparks on their direct cable expose two RoCE devices for the one port (TensorFold `RUNBOOK.md` at the pin); the default lists both (`rocep1s0f1,roceP2p1s0f1`).
- **Same settings on both ranks.** The ranks compare their settings at startup and refuse to serve on a mismatch. `run.sh` builds both argvs from one `serve_args()` and both container envs from one `container_env()`; the worker gets every setting forwarded shell-quoted. Keep it that way. `VALIDATE_ONLY=args` prints both.
- **Weights.** Order: download on the head by revision sha (`main` holds only the model card), rsync to the worker, run. The worker has no internet: rsync only the pinned snapshot dir with `-aL` (README Weights), never the whole cache entry, which holds other revisions. `run.sh` checks the snapshot on both nodes (config and index sha256, 48 shards, total bytes, headers) before anything starts. The pins come from the Hub (`evidence/s1-snapshot-pins`).
- **Image.** Built on the head from `docker/Dockerfile` (spark2 has no internet). Run `IMAGE_ONLY=1 ./run.sh` before the downtime. `run.sh` refuses an image whose `tensorfold.sha` label differs from `TF_SHA`, and copies the head's image to the worker when the image IDs differ.
- **TF32.** Keep `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` in the container env, as the receipts ran.

## Known engine limits (deepseek_v41 at `b86514a`, two ranks)

- Two ranks only. Requests decode concurrently at `--parallel N` (`PARALLEL`, 1 to 4; each lane holds a whole `CONTEXT` window, allocated at startup), and each reply equals its solo run. Both ranks decode each request to `max_tokens` or an end token; a client disconnect stops decoding it after its next round.
- A client that disconnects while its prompt fills is noticed at its first token: the fill runs to its end first.
- This export only. The family checks the checkpoint at startup and refuses EXL3 outside the routed experts or a BF16 LM head.
- Refused with HTTP 400: images, `response_format` / guided / structured outputs, `logprobs`, `tool_choice: "required"` or named, `thinking_budget`, `n > 1`.
- An effort name turns thinking on, even on a `--no-thinking` server; `chat_template_kwargs.thinking: false` keeps it off.
- Sampling with top-k off is exact and draws on the device, with or without a top_p cut: on the ruler's prose prompt top_p 0.95 decodes at 37.7 tok/s against 42.8 with no sampling field (top_p 1.0) at `903a1e8`, the gap being that reply's extra rounds, and replies equal the old host path's (`evidence/s5-final/derived.txt`, `evidence/s4-device-nucleus/derived.txt`). At `ec28f35` a cut read each rank's vocabulary half to the host (15.5 tok/s there), which is why `run.sh` passes `--top-p 1.0` (`TOP_P` in `recipe.yaml`, decision `TOPP1`). Past the cut's `TMAX` bound, or where the device draw cannot decide, the host rule still runs. Greedy is not affected.
- What happens when one rank dies mid-request is not tested here. Restart both with `./stop.sh && ./run.sh`.
- A failure on one rank in the middle of a round (the other rank still in its collectives) is not recovered: the pair stops serving. Restart both with `./stop.sh && ./run.sh`.

## Host memory safety

- **Watchdogs.** Keep `OOM_SCORE_ADJ=1000`, `MEMGUARD=1` and `--ulimit core=1`.
- **Load gate.** MemAvailable ≥ `MEM_GATE_GIB=100` on each node before a load. `run.sh` refuses less than a floor it computes from `PARALLEL` and `CONTEXT`, unless `FORCE_UNSAFE_MEM_GATE=1`; record that boot. The floor is resident 75.99 + max(5.33, 3.71 + 0.95 a further lane per 1048576-token window) + 12.1 reserve GiB, rounded up: 95 at the defaults, 94 at `--parallel 1`.
- **Admission.** The engine grants MemAvailable less max(4 GiB, MemTotal / 10) and refuses an explicit `--context` that does not fit, on both ranks. Startup estimate at `903a1e8` 81.32 GiB at the native 1048576 (`evidence/s5-final/bootE/gate/startup.txt`); at `0d91389` it was 78.46 GiB at 65538 and 79.09 GiB at 1048576 (`evidence/s0-engine-receipts/serve_*_rank0.log`).
- **No sudo.** No `drop_caches`. If MemAvailable stays short, find what holds it (`free -h`, `docker ps`).
- **Before a heavy step** (a long prompt at a large window), run `free -h` on both nodes.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
VALIDATE_ONLY=args ./run.sh
shellcheck -S warning run.sh stop.sh tools/session_gate.sh
```

After `./run.sh` is up:
- `GET /health` is 200, and `GET /v1/models` lists `deepseek-ai/DeepSeek-V4.1-Flash`.
- `run.sh` printed rank 0's `startup estimate … allocated prompt/reply window <CONTEXT>` line.
- `tools/session_gate.sh evidence/<id>` ends `GATE=PASS`.

## Never touch

- Live HF tokens
- An unpinned base image, an unpinned TensorFold ref, or an unpinned model revision
- Hand-edited generated README / `run.sh` blocks
- Advertising a window, concurrency level or speed that was not run here
