# AGENTS.md — DeepSeek-V4.1-Flash EXL3 · TensorFold · 2× DGX Spark

Serve `sfxnz/DeepSeek-V4.1-Flash-EXL3` (revision `982b704`, branch `2.0bpw-mcg-viterbi-lmhead-mxfp8`) at TP=2 with TensorFold's own `deepseek_v41` CUDA family: commit `ec28f35` on `sfxnz/TensorFold` branch `dsv41-recipe-engine` (v0.6.4 + upstream PRs #390 and #391 + the device keyed draw for top_k-off rows, the patch of `be2cc80` on branch `cuda-keyed-draw`), inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned in `docker/Dockerfile`). Not vLLM. The vLLM recipe is a separate repo (`sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`); one repo per engine.

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

- **One engine.** This recipe serves TensorFold's own family only. No patches, no vendored launcher code, no other packs. To change the engine, bump `TF_REPO` + `TF_SHA` in `recipe.yaml` (to an upstream release tag's commit once the family is released there), render, rebuild with `IMAGE_ONLY=1 ./run.sh`, and measure again.
- **Window.** `CONTEXT=1048576`, the native window: s2 ran the bench and needles up to 1,039,528 prompt tokens through `run.sh` at it (`evidence/s2-run-sh-ec28f35/ctx1m`). Do not advertise a window that was not run through `run.sh`.
- **One knob per session.** Change one knob at a time against the gate (`tools/session_gate.sh EVDIR`). Final numbers come from two boots: report the median of the per-boot medians.
- **Frozen ruler.** `bench_decode.py` is byte-identical to the vLLM sibling's frozen ruler (sha256 `3172cbc4…`, enforced by `tests/`). Do not edit it. Its `/metrics` acceptance fields stay empty on this engine; drafting shows in rank 0's `done req-…` log lines.
- **Memory.** Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- **HCAs.** Pin `NCCL_IB_HCA`. Two Sparks on their direct cable expose two RoCE devices for the one port (TensorFold `RUNBOOK.md` at the pin); the default lists both (`rocep1s0f1,roceP2p1s0f1`).
- **Same settings on both ranks.** The ranks compare their settings at startup and refuse to serve on a mismatch. `run.sh` builds both argvs from one `serve_args()` and both container envs from one `container_env()`; the worker gets every setting forwarded shell-quoted. Keep it that way. `VALIDATE_ONLY=args` prints both.
- **Weights.** Order: download on the head by revision sha (`main` holds only the model card), rsync to the worker, run. The worker has no internet: rsync only the pinned snapshot dir with `-aL` (README Weights), never the whole cache entry, which holds other revisions. `run.sh` checks the snapshot on both nodes (config and index sha256, 48 shards, total bytes, headers) before anything starts. The pins come from the Hub (`evidence/s1-snapshot-pins`).
- **Image.** Built on the head from `docker/Dockerfile` (spark2 has no internet). Run `IMAGE_ONLY=1 ./run.sh` before the downtime. `run.sh` refuses an image whose `tensorfold.sha` label differs from `TF_SHA`, and copies the head's image to the worker when the image IDs differ.
- **TF32.** Keep `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` in the container env, as the receipts ran.

## Known engine limits (deepseek_v41 at `ec28f35`, two ranks)

- Two ranks only, one request at a time (`--parallel` is ignored). Both ranks decode each request to `max_tokens` or an end token.
- This export only. The family checks the checkpoint at startup and refuses EXL3 outside the routed experts or a BF16 LM head.
- Refused with HTTP 400: images, `response_format` / guided / structured outputs, `logprobs`, `tool_choice: "required"` or named, `thinking_budget`, `n > 1`.
- An effort name turns thinking on, even on a `--no-thinking` server; `chat_template_kwargs.thinking: false` keeps it off.
- Sampling with top-k off is exact. Without a top_p cut (`top_p` 1.0) the draw runs on the device. With a cut (the engine default `top_p` 0.95, or a request's top_p under 1) a row whose nucleus runs past 1,024 candidates still reads each rank's vocabulary half to the host: about 15-24 tok/s against about 37-50 with `top_k` 20 or `top_p` 1.0 (`evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl`, `evidence/s3-default-1m/derived.txt`). So `run.sh` passes `--top-p 1.0` (`TOP_P` in `recipe.yaml`, decision `TOPP1`). Greedy is not affected.
- What happens when one rank dies mid-request is not tested here. Restart both with `./stop.sh && ./run.sh`.

## Host memory safety

- **Watchdogs.** Keep `OOM_SCORE_ADJ=1000`, `MEMGUARD=1` and `--ulimit core=1`.
- **Load gate.** MemAvailable ≥ `MEM_GATE_GIB=100` on each node before a load. `run.sh` refuses less than 92 unless `FORCE_UNSAFE_MEM_GATE=1`; record that boot.
- **Admission.** The engine grants MemAvailable less max(4 GiB, MemTotal / 10) and refuses an explicit `--context` that does not fit, on both ranks. Startup estimate 78.46 GiB at 65538, 79.09 GiB at the native 1048576 (`evidence/s0-engine-receipts/serve_*_rank0.log`).
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
