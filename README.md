# DeepSeek-V4.1-Flash EXL3 · TensorFold · 2× DGX Spark

Serve [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/tree/2.0bpw-mcg-viterbi-lmhead-mxfp8) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2 with [TensorFold](https://github.com/ashhart/TensorFold)'s `deepseek_v41` CUDA family. Not vLLM. The vLLM recipe for the same weights is [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark); one repo per engine.

**Status: measured through `run.sh` on this pair.** Session [`s2-run-sh-ec28f35`](evidence/README.md#s2-run-sh-ec28f35-2026-10-04) built the image from git at `ec28f35`, booted `./run.sh` three times (twice at `--context 65538` for the decode table, once at the native 1048576), and ran the frozen ruler, sampled cells, the engine's `bench_openai` and `prefill_cold`, exactness pairs, needles up to 1,039,528 prompt tokens, and a quality spot check. Session [`s3-default-1m`](evidence/README.md#s3-default-1m-2026-10-04) booted `./run.sh` twice more at the shipped defaults (`--context 1048576`, `--top-p 1.0`) for the decode table, the sampled cells, exactness of the default request and a quiet `prefill_cold`. Session [`s4-device-nucleus`](evidence/README.md#s4-device-nucleus-2026-10-05) bumped the engine to `41306d5` (the device top_p cut), built the image with `IMAGE_ONLY=1 ./run.sh` and booted `./run.sh` once at the shipped defaults for the gate, the sampled cells and exactness at top_p 0.95 and 1.0. The window default is the native 1048576. The snapshot is the pinned `snapshots/982b704…` folder on both nodes; `run.sh`'s check passed on both at every boot.

- **Engine.** TensorFold `41306d5` from [sfxnz/TensorFold `dsv41-recipe-engine2`](https://github.com/sfxnz/TensorFold/tree/dsv41-recipe-engine2): release v0.6.4 (`6ea5ade`) plus upstream PRs [#390](https://github.com/ashhart/TensorFold/pull/390) and [#391](https://github.com/ashhart/TensorFold/pull/391), which add the family (tip `0d91389`), plus two commits that draw top-k-off rows on the device, without a top_p cut (`44338a2`) and with one (`41306d5`): the same patches as `be2cc80` and `43ebdce` on branch `cuda-keyed-draw`. s2 and s3 ran `ec28f35`, the tree of `44338a2`. Apache-2.0. The image builds from [`docker/Dockerfile`](docker/Dockerfile) on `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned).
- **Weights.** Revision `982b70452f399814f56b46272fd30394ae10d58c` (branch `2.0bpw-mcg-viterbi-lmhead-mxfp8`), 48 shards. The routed experts are EXL3 at 2 bits (`mcg` codebook). Everything else is DeepSeek's own bytes: FP8 with 32x32 block scales, an MXFP8 LM head, MXFP4 DSpark draft experts, and the two Engram tables in shards 47 and 48.
- **Engram.** Each rank reads its half of every Engram row from the pack on its own local disk. No repacking, no second copy. Both nodes need the whole revision.
- **Ranks.** Rank 1 runs on `spark2`, rank 0 serves HTTP on `spark1`. NCCL over the QSFP RoCE link.

## Measured on 2× DGX Spark (L.A.I.L lab)

`python3 bench_decode.py`, byte-identical to the vLLM sibling's frozen ruler (sha256 `3172cbc4…`). It sends `ignore_eos` and `chat_template_kwargs {"thinking": false, "reasoning_effort": "low"}`; the server log shows `thinking=False` for every bench request (32 of 32 `done` lines in each boot's `docker-head.log`, [`bootC`](evidence/s3-default-1m/bootC/gate/docker-head.log), [`bootD`](evidence/s3-default-1m/bootD/gate/docker-head.log)). Do not copy community tok/s into this table.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: frozen bench_decode.py (sha256 3172cbc4…), streamed greedy, thinking off, ignore_eos, max_tokens 200, 9 runs, two boots at the shipped defaults (s3 bootC and bootD, 2026-10-04: `--context 1048576`, `--top-p 1.0`, 3 drafts a round, no env override); each row is the median of the two per-boot medians (evidence/s3-default-1m/decode.txt, which also gives each boot's and the pooled 18-run medians); TensorFold `ec28f35` at TP=2 through `./run.sh`, `tools/session_gate.sh` with RUNS_LONG=9; runs 2 to 9 of each cell resume the kept prompt, so TTFT p50 is a resumed prompt. Also measured: two boots at `--context 65538` (s2 bootA and bootB): prose 45.7, structured 64.2, prose_long 34.4 (evidence/s2-run-sh-ec28f35/decode.txt); one boot at the pinned `41306d5` and the shipped defaults (s4, 2026-10-05; its one commit over `ec28f35` changes only top-k-off sampling with a top_p cut, which the greedy ruler does not reach): prose 45.26, structured 63.74, prose_long 34.18 (evidence/s4-device-nucleus/gate/bench-*.out, evidence/s4-device-nucleus/derived.txt).

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose (note 1) | 1 | 46.0 | 46.0 | 0.07 s |
| structured | 1 | 64.1 | 64.1 | 0.07 s |
| prose_long | 1 | 34.6 | 34.6 | 0.07 s |

1. The prose reply ends at its end token after 74 tokens; ignore_eos decodes the other 126 (post_eos_fraction 0.63).
<!-- END generated measured -->

One request runs at a time: the family ignores `--parallel`, so there are no c=2 rows.

- Per boot (C / D): prose 45.74 / 46.29, structured 63.58 / 64.58, prose_long 34.37 / 34.86 tok/s. Pooled over both boots' 18 runs: 46.065, 64.12, 34.555 ([`decode.txt`](evidence/s3-default-1m/decode.txt), from `gate/bench-*.out` of [`bootC`](evidence/s3-default-1m/bootC/gate/) and [`bootD`](evidence/s3-default-1m/bootD/gate/)). Inter-chunk p50 about 61 ms. The ruler is greedy, so `--top-p` does not touch it.
- Also measured, s2 at `--context 65538` (boots A / B): prose 45.75 / 45.59, structured 63.86 / 64.48, prose_long 34.43 / 34.43; rows 45.7, 64.2, 34.4 ([`s2 decode.txt`](evidence/s2-run-sh-ec28f35/decode.txt)). s2's boot at 1048576, before its long prompts: 46.24, 64.90, 34.75 ([`ctx1m/gate`](evidence/s2-run-sh-ec28f35/ctx1m/gate/)). The window does not slow decode.
- The engine's own receipt run at `0d91389` (dev launcher, 3 runs, [`s0`](evidence/s0-engine-receipts/s65538/bench_decode.log)) gave 45.5, 63.5, 34.0.

### Sampled decode (top-k off)

The ruler is greedy. [`sampled_cell.py`](evidence/s4-device-nucleus/scripts/sampled_cell.py) (s2's, plus each reply's token hash) sends the ruler's prose prompt with sampling: streamed, 200 tokens, ignore_eos, no seed (the server seeds from the prompt, so each run draws the same reply), 9 runs a boot. Median decode tok/s at the shipped defaults (`TOP_P=1.0`): s4, one boot at `41306d5`, against s3's boots C / D at `ec28f35`:

| Request sends | s4, `41306d5` | s3, `ec28f35` (C / D) | Reply equal to s3 |
|---|---:|---:|:---:|
| no sampling field (server default: temperature 1.0, top_p 1.0, top-k off) | 36.6 | 36.76 / 37.10 | yes |
| temperature 1.0, top_p 1.0 (top-k off) | 36.6 | 37.03 / 37.27 | yes |
| temperature 1.0, top_k 20 | 37.0 | 37.12 / 37.63 | yes |
| temperature 1.0, top_p 0.95 (top-k off) | 31.8 | 15.23 / 15.72 | yes |
| temperature 1.0, top_p 0.9 (top-k off) | 31.8 | not run | - |

([`s4 sampled-*.txt`](evidence/s4-device-nucleus/), [`s4 derived.txt`](evidence/s4-device-nucleus/derived.txt); [`s3 bootC`](evidence/s3-default-1m/bootC/), [`bootD`](evidence/s3-default-1m/bootD/), [`s3 derived.txt`](evidence/s3-default-1m/derived.txt)). Each s4 reply has the same token hash as the same request in both s3 boots (rank 0's `done` lines): the device cut draws exactly what the host path drew. top_p 0.95 is 2.05 times s3's.

**top_p under 1 now draws on the device too.** At `ec28f35` a row whose nucleus ran past each rank's 1,024 best candidates read the whole vocabulary half to the host, about 68 ms of sampling a round (s2's top_p 0.95 rows, [`pairs.jsonl`](evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl)). At `41306d5` the same requests spend 3.4-3.8 ms a round sampling, against 2.8-3.0 at top_p 1.0 ([`s4 derived.txt`](evidence/s4-device-nucleus/derived.txt)). The top_p 0.95 and 0.9 cells are slower than top_p 1.0 because their reply is a different text that takes more rounds: the same reply at both cuts, 99 rounds against 86 for the top_p 1.0 reply (100 of 293 drafts accepted against 113 of 253). Scaled by rounds, the three cells decode at the same speed (36.56, 36.56, 36.55). The engine still takes the host path past the cut's `TMAX` bound and wherever the device draw cannot decide (its slack check); the logs do not count those rows.

### Exactness

Drafted replies equal `"draft": false` replies, token for token, in 13 of 13 checks ([`pairs.jsonl`](evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl), [`pairs.py`](evidence/s2-run-sh-ec28f35/scripts/pairs.py)): three prompts at top_k 20 (temperature 0 and 1), the s0 top_k-omitted triple (drafted, `"draft": false`, a resend with 17 of 18 prompt tokens cached), and three prompts at temperature 1 with top_k omitted, at the default top_p 0.95 and at top_p 1.0 (the device draw). All 15 replies of the requests the s0 receipts also sent have the same token hash as at `0d91389` ([`pairs_vs_s0.txt`](evidence/s2-run-sh-ec28f35/bootA/pairs_vs_s0.txt)).

At the shipped defaults (s3 boot D), a request with no sampling field (seed 1234, 160 tokens) on the same three prompts: drafted equals `"draft": false` in 3 of 3, and each reply's token hash equals s2's explicit temperature 1.0 / top_p 1.0 reply (`7d1dc35680a6`, `22480b3cd4c8`, `65f2c13cd1dc`; [`pairs.jsonl`](evidence/s3-default-1m/bootD/pairs.jsonl), [`pairs_default.py`](evidence/s3-default-1m/scripts/pairs_default.py)). Drafted, those requests decoded at 45-52 tok/s.

At `41306d5` (s4), temperature 1.0 with top_k omitted, seed 1234, 160 tokens, on the same three prompts, at top_p 0.95 and 1.0: drafted equals `"draft": false` in 6 of 6, and each drafted reply's token hash equals s2's reply to the same request at `ec28f35` in 6 of 6: the host path's at top_p 0.95 (`2e35c52cda08`, `75ad34c67f94`, `14f0fa7d7629`; s2 left top_p to the server's 0.95), the device draw's at 1.0 ([`pairs.jsonl`](evidence/s4-device-nucleus/pairs.jsonl), [`pairs_nucleus.py`](evidence/s4-device-nucleus/scripts/pairs_nucleus.py)). Drafted at top_p 0.95 they decoded at 47.0-50.1 tok/s, against 23.2-24.8 in s2, in the same number of rounds ([`derived.txt`](evidence/s4-device-nucleus/derived.txt)).

### Against the vLLM sibling (different sessions)

Same pack, same machines, same client tools; not interleaved. vLLM numbers: the sibling recipe's round 36 (2026-09-27, 2 boots, median of the per-boot medians, [`round36-headline.json`](evidence/s2-run-sh-ec28f35/vllm-sibling/round36-headline.json)) and the lab's vLLM oracle session (2026-10-03, [`G7_*`](evidence/s2-run-sh-ec28f35/vllm-sibling/SOURCES.txt)). This recipe: s3 (bench_decode, two boots; prefill_cold, boot D) and s2 (bench_openai, boot A), 2026-10-04.

| Cell | This recipe | vLLM sibling |
|---|---:|---:|
| bench_decode prose c=1, tok/s | 46.0 | 50.7 |
| bench_decode structured c=1 | 64.1 | 84.3 |
| bench_decode prose_long c=1 | 34.6 | 43.8 |
| bench_openai fibonacci-raw, t=1, median tok/s | 41.2 | 47.3 |
| bench_openai gpu-chat-no-think, t=1 | 37.1 | 44.3 |
| bench_openai gpu-chat-no-think, t=0 | 41.2 | 43.1 |
| prefill_cold, prompt tok/s at 2k / 8k / 16k / 32k / 64k | 526 / 552 / 561 / 564 / 571 | 778 / 769 / 772 / 777 / 774 |

This recipe's `bench_openai` and `prefill_cold` are the engine's `tools/` at `ec28f35`, run from the host: `bench_openai` against s2 boot A ([`bench_openai.log`](evidence/s2-run-sh-ec28f35/bootA/bench_openai.log)); at t=1 it sends top_k 20 and top_p 0.95, 64 tokens, 5 reps, so the server's top_p does not change it. `prefill_cold` ran quiet on s3 boot D at the shipped defaults, with s2's command and prompts: nothing else sent a request meanwhile (the 16 `done` lines after the last pairs request are its 16 requests, none cached; only the lab app's `/v1/models` polls were connected; [`prefill_cold.log`](evidence/s3-default-1m/bootD/prefill_cold.log), [`prefill-clients.txt`](evidence/s3-default-1m/bootD/prefill-clients.txt), [`derived.txt`](evidence/s3-default-1m/derived.txt)). s2 boot A, after other cells and not checked for quiet, gave 499 / 527 / 549 / 534 / 537 ([`s2 prefill_cold.log`](evidence/s2-run-sh-ec28f35/bootA/prefill_cold.log)). vLLM's t=0 fibonacci-raw row did not run (the tool fails there on vLLM). The prefill prompt sets differ: the same corpus and builder, counted by each server's tokenizer. Decode is 4.5-24.0% lower, prompt tok/s 26-32% lower ([`derived.txt`](evidence/s3-default-1m/derived.txt)). The engine receipts at `0d91389` gave 42.0 / 37.6 / 41.1 tok/s and 519-568 prompt tok/s on the same tools.

### Window and long prompts

`CONTEXT=1048576` booted through `run.sh` (`startup estimate 79.09 GiB within 99.54 GiB` / `101.55 GiB`). Needles from the vLLM sibling's `quality_eval.py` builder, one at a time, greedy ([`long_needle.py`](evidence/s2-run-sh-ec28f35/scripts/long_needle.py), [`needles.jsonl`](evidence/s2-run-sh-ec28f35/ctx1m/needles.jsonl)); MemAvailable sampled every 5 s on both nodes ([`memwatch.tsv`](evidence/s2-run-sh-ec28f35/ctx1m/memwatch.tsv)); MiB to GiB, prompt tok/s and minutes in [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt):

| Prompt tokens | Depth | Prefill s | Prompt tok/s | Found | Min MemAvailable head / worker |
|---:|---:|---:|---:|---|---:|
| 130,171 | 0.1 | 214.3 | 607 | yes | 24.8 / 26.7 GiB |
| 130,541 | 0.5 | 215.1 | 607 | yes | 24.6 / 27.4 GiB |
| 129,076 | 0.9 | 212.8 | 607 | yes | 24.8 / 27.4 GiB |
| 257,955 | 0.5 | 439.8 | 587 | yes | 24.6 / 27.0 GiB |
| 523,681 | 0.5 | 946.9 | 553 | yes | 23.4 / 26.7 GiB |
| 1,039,528 | 0.5 | 2077.8 | 500 | yes | 22.1 / 26.7 GiB |

A 1M-token prompt takes 35 minutes to prefill, and the server serves nothing else meanwhile. Memory stays flat after admission: the window is allocated at startup. Tested windows: 65538 (two boots, s2) and 1048576 (three boots: s2's, with the needles, and s3's two at the shipped defaults). The default is 1048576.

### Quality

On the 1048576 serve after the needles ([`quality_quick.txt`](evidence/s2-run-sh-ec28f35/ctx1m/quality_quick.txt), the vLLM sibling's `quality_eval.py --quick --only selfcons,gsm8k,tools,needle`, gated against its round-36 quick baseline): pass. selfcons 12 of 12 identical, golden hazard 0.01307; GSM8K 97/100; tools exact_args 22/22, json_valid 22/22, no_call 8/8; needles 6/6 at 8k and 32k. Every reply, miss and cell equals the s0 run at `0d91389` ([`quality_vs_s0.txt`](evidence/s2-run-sh-ec28f35/ctx1m/quality_vs_s0.txt)). Teacher-forced NLL on the 40 G1 passages (19,828 positions, both ranks, the recipe's image): 0.137069, the repeat equal, and all 80 score files bit-identical to `0d91389`'s ([`nll_compare.txt`](evidence/s2-run-sh-ec28f35/nll/nll_compare.txt)). The device draw does not touch logits. s2 ran the quality set before `TOP_P=1.0`, with the engine's default top_p 0.95 for any request that sent none; it was not rerun at 1.0.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (`10.100.8.1` / `10.100.8.2` in this lab), both RoCE ports `ACTIVE`.
- Docker + NVIDIA Container Toolkit on both nodes. SSH from the head to the worker (`spark2` here).
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up; `run.sh` refuses to.
- Disk on each node: the whole revision, 333 GiB ([`pins.json`](evidence/s1-snapshot-pins/pins.json), [family page](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/docs/recipes/deepseek-v4.1-flash.md#download-on-both-machines)), in the HF cache (`HF_CACHE`, default `~/.cache/huggingface`) on local NVMe, never NFS. Other revisions already in the same cache entry add to that. Plus the kernel cache under `TF_CACHE`.

## Weights

Order: download on the head, copy to the worker, then `./run.sh`.

1. **Head.** `tensorfold pull` and a plain `hf download` take the repository's `main`, which holds only the model card, so name the revision:

   ```bash
   hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 982b70452f399814f56b46272fd30394ae10d58c
   ```

   `./run.sh` runs that download itself on the head when the snapshot is missing or incomplete, after it checks the free space, and before it checks the worker's copy.
2. **Worker.** `spark2` has no internet and never downloads. Copy only the pinned snapshot from the head over the link, on the worker:

   ```bash
   S=$HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/982b70452f399814f56b46272fd30394ae10d58c
   rsync -aL --partial --mkpath 10.100.8.1:$S/ $S/
   ```

   `-L` copies the files the snapshot links to, so 333 GiB cross the link, not the whole cache entry with its other revisions. The worker's copy holds plain files and no `blobs/`. It never downloads, so that is enough.

Before anything starts, `run.sh` checks the snapshot on both nodes, the worker over ssh:

- `config.json` and `model.safetensors.index.json` have their pinned sha256;
- the index names 48 shards, and their sizes sum to the pinned total;
- each shard's safetensors header ends exactly at its file size (a truncated copy fails here, not mid-load);
- `tokenizer.json` and `tokenizer_config.json` exist.

The pins are in `recipe.yaml` (`model`), from the Hub at this revision ([`pins.json`](evidence/s1-snapshot-pins/pins.json), [`hub_files.tsv`](evidence/s1-snapshot-pins/hub_files.tsv)). The containers mount the HF cache read-only and serve the snapshot path inside it.

## Image

`run.sh` builds `tf-dsv41-flash:0.6.4-41306d5` from [`docker/Dockerfile`](docker/Dockerfile) on the head when it is missing. It copies the image to the worker over the link when the worker's image ID differs (`docker save | ssh docker load`). Do both before the downtime, while another serve still holds the pair:

```bash
IMAGE_ONLY=1 ./run.sh   # build on the head, copy to the worker; no GPU, no weights, no memory gate
```

- The build runs `pip install "tensorfold @ git+TF_REPO@TF_SHA"` under [`docker/constraints.txt`](docker/constraints.txt). It then checks the commit pip recorded (`direct_url.json`) against `TF_SHA` and `TF_REPO`.
- The constraints pin TensorFold's runtime deps at the versions expected in the base image. The build fails if pip changes any installed package: a wrong pin fails the build, not a serve, and torch and triton stay the base's. s2's build passed that check: pip changed nothing ([`image-only.txt`](evidence/s2-run-sh-ec28f35/image-only.txt)). `IMAGE_ONLY=1 ./run.sh` took 3 min 34 s, most of it the copy to the worker ([`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). s4's build at `41306d5` passed the same checks and took 208 s ([`image-only.txt`](evidence/s4-device-nucleus/image-only.txt), [`image-ids.txt`](evidence/s4-device-nucleus/image-ids.txt), [`derived.txt`](evidence/s4-device-nucleus/derived.txt)).
- No extras: the family serves no grammars and no images.
- Label `tensorfold.sha`. `run.sh` refuses an image whose label differs from `TF_SHA`.
- The four CUDA extensions compile at the first start, into `TF_CACHE/<TF_SHA>` on each node.
- To bump the engine to an upstream release, set `TF_REPO` and `TF_SHA` in `recipe.yaml` (`engine`), render, rebuild, and measure again.

`VALIDATE_ONLY=args ./run.sh` prints both ranks' argv and the full container env.

## Quick start

On the head Spark (`spark1`):

```bash
VALIDATE_ONLY=1 ./run.sh   # checks every setting; no Docker
IMAGE_ONLY=1 ./run.sh      # before the downtime: build and copy the image
./run.sh
```

`./run.sh` does this, in order:

1. **Refusals.** It takes a start lock (one `./run.sh` or `IMAGE_ONLY=1 ./run.sh` at a time). It refuses another GPU container (and never removes it), a busy `PORT` / `MASTER_PORT`, and a missing SSH link to the worker.
2. **Worker refusals.** Over ssh, read-only: another GPU container, or a CUDA process while no container of this recipe is there. Nothing is stopped.
3. **Weights.** The head's snapshot (downloaded if missing, see Weights), then the worker's, over ssh.
4. **Image.** It builds or checks the image (label) and copies it to the worker if the IDs differ, before any memory step.
5. **Head node.** HCA ports `ACTIVE`, no running CUDA process, then the memory gate: MemAvailable ≥ `MEM_GATE_GIB` (100).
6. **Start.** It starts rank 1 on `spark2` (the same node checks there), then rank 0 here. If anything fails from here until rank 0 answers, it stops both ranks.
7. **Ready.** It waits for `/health` and `/v1/models`, up to `READY_TIMEOUT` (1800 s), checks the worker every ~10 s, and prints rank 0's admission line (`startup estimate …`).

Both ranks get their flags from one function and their container env from another. The worker's settings are forwarded shell-quoted and rebuilt by the same code.

If SSH is not set up, start the worker yourself, then the head without orchestration (the same settings on both):

```bash
# spark2
ROLE=worker ./run.sh
# spark1
ORCHESTRATE=0 ./run.sh
```

Smoke test (thinking is off by default):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Probes and the gate pack against the live API:

```bash
python3 smoke_chat.py                      # 17*19 -> 323, thinking off
python3 smoke_count.py                     # count 1..200 without a gap
python3 bench_decode.py --concurrency 1 --runs 9   # the frozen ruler
tools/session_gate.sh evidence/<id>        # all of the above, prose_long, and the receipts
```

Stop both ranks from the head:

```bash
./stop.sh
```

`stop.sh` stops rank 0 first with SIGTERM, then waits up to `STOP_TIMEOUT` (30 s) for rank 1 to exit before it stops and removes rank 1's container. It touches only containers named `CONTAINER_NAME` (`tf-dsv41-flash`) with the label `ai-lab.recipe=dsv41-tensorfold`, which `run.sh` sets on every container it starts.

## Defaults

`recipe.yaml` is the source of truth. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Engine | TensorFold `903a1e8af62c8f46eceee6b95481706ada30ae49` (v0.6.4 + the deepseek_v41 CUDA family and its speed units) from `https://github.com/sfxnz/TensorFold.git`, built into `tf-dsv41-flash:0.6.4-903a1e8` from `docker/Dockerfile` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-EXL3` at `982b70452f399814f56b46272fd30394ae10d58c`, served from the HF cache `$HOME/.cache/huggingface` (read-only) on each node |
| Snapshot check | `config.json` sha256 `6469adab394edead3eec148323e7471582d08acdf60a438bf0c9e815b69f36c5`, index sha256 `91731e4af38696bd4c09e960f4b599d1d49f35d445e4f88a43cc355d9e139f03`, 48 shards, 357466041064 shard bytes, every shard's header ending at its size |
| Ranks | `--tp 2`: rank 1 on `spark2` first, then rank 0 (HTTP) on the head; rendezvous `10.100.8.1:29571` |
| `--context` | 1048576, the model's native window (tested through run.sh: needles at 130k, 258k, 524k and 1,039,528 prompt tokens, all found) |
| Drafting | `--mtp-drafts 5 --mtp-confidence 0.15`, the engine's default policy: up to 5 DSpark drafts a round, the chain stopping where the drafts' confidence product falls under 0.15 (`MTP_CONFIDENCE` empty: 5 drafts every round; 0 drafts: one token a round) |
| Thinking | off (`--no-thinking`, `THINKING=0`) on both ranks; a request turns it on with `chat_template_kwargs.thinking` or an effort name |
| `--max-tokens` | 4096 (the reply cap when a request sets none) |
| Sampling | `--top-p 1.0` on both ranks (the top_p when a request sends none); temperature not passed (the engine's default, 1.0); top-k off (the family's default). DeepSeek's model card: temperature 1.0, top_p 0.95 or 1.0 |
| NCCL | `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1` |
| Memory | MemAvailable >= 100 GiB on each node before a start; memguard 3072 MB RAM and 2048 MB swap |
| Kernel cache | `$HOME/.cache/tensorfold-dsv41/<TF_SHA>` on each node, mounted at `/cache/tf` |
| API | `http://<head>:8000/v1`, served as `deepseek-ai/DeepSeek-V4.1-Flash` |
| Container | `tf-dsv41-flash` on each node |
<!-- END generated defaults -->

The serve flags match the receipt run's (`--tp 2 --master … --name deepseek-ai/DeepSeek-V4.1-Flash --no-thinking --no-update-check` on both ranks, and `--context`, 65538 there, [`environment.txt`](evidence/s0-engine-receipts/environment.txt)). `run.sh` also passes `--mtp-drafts 3` to both ranks, and rank 0 also gets `--max-tokens 4096`: the engine's and the CLI's defaults, written out so a bump cannot change them silently. Both ranks also get `--top-p 1.0`, which the receipts did not pass (their server default was 0.95; see Sampled decode). Rank 0 prints it: `serving … (sampling: top_p 1.0; …)` ([`startup.txt`](evidence/s3-default-1m/bootC/gate/startup.txt)). A request that omits `temperature` gets the engine's 1.0 ([`cuda/server.py`](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/src/tensorfold/cuda/server.py#L92) at the pin; the family sets top-k off, [`app.py`](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/src/tensorfold/families/deepseek_v41/cuda/app.py#L36)). The container env adds `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, as the receipt run had: it keeps fp32 matmuls in fp32 whatever TF32 default the NGC image sets.

`run.sh` refuses, before `VALIDATE_ONLY` exits:

- non-decimal or zero-padded integers, a `TF_SHA` / `SNAPSHOT_SHA` that is not 40 hex, a pinned sha256 that is not 64 hex, a `TF_REPO` that is not an https `.git` URL, a cache path that is relative or has a space or `:`
- `TP` other than 2, `CONTEXT` above the native 1048576, `MTP_DRAFTS` above 5, `MTP_CONFIDENCE` outside (0, 1], `TOP_P` outside (0, 1], `PORT` equal to `MASTER_PORT`
- `MEM_GATE_GIB` below 92, unless `FORCE_UNSAFE_MEM_GATE=1` for one boot
- a `CONTAINER_NAME` that does not match `tf-dsv41-[a-z0-9-]+`
- `TF_DSV41_CACHE_GIB` / `TF_DSV41_CACHE_ENTRIES` that are not numbers; an exported `TF_DSV41_*` variable that is not a `recipe.yaml` knob or in `EXTRA_ENV` (it would never reach the containers)
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds (`--tp`, `--rank`, `--master*`, `--host`, `--port`, `--name`, `--context`, `--mtp-*`, `--no-drafts`, `--thinking`, `--no-thinking`, `--max-tokens`, `--top-p`, `--no-update-check`), or a prefix of one; flags the family refuses or ignores (`--parallel`, `--prefill-fp8`, `--drafter`, `--thinking-budget`, `--vision*`, `--kv-dtype`)
- `EXTRA_ENV` that is not `KEY=VALUE`, or that sets a variable `run.sh` already sets (every `recipe.yaml` key, `NCCL_IB_HCA`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, …)

`BENCH_ONLY=1` binds the API to 127.0.0.1. The server's top_p is `TOP_P=1.0`, and `EXTRA_ARGS` may not set `--top-p`; `--temperature` and `--top-k` go through `EXTRA_ARGS`; other engine variables through `EXTRA_ENV` (`EXTRA_ENV="TENSORFOLD_MEMORY_RESERVE_GIB=8"`).

## Not supported (this engine, two ranks)

From the family's [recipe page](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/docs/recipes/deepseek-v4.1-flash.md) at the pinned commit:

- Other exports of the model. The family checks the checkpoint at startup and refuses EXL3 outside the routed experts or a BF16 LM head; other packs do not load.
- One rank, one GPU, or a separate `--drafter`.
- Concurrent streams. `--parallel` is ignored with a note; requests run one at a time and queue.
- Images, `response_format` and the `guided_*` and `structured_outputs` fields, `logprobs`, `tool_choice: "required"` or a named function, `thinking_budget`, and `n` above 1: HTTP 400.
- `--prefill-fp8`: refused. Prompt matmuls take bf16 activations.
- Tool calls arrive whole in the final chunk of a streamed reply, with `finish_reason: "tool_calls"`.
- An effort name turns thinking on. A client that sends `reasoning_effort: "low"` to this `--no-thinking` server gets thinking. Send `chat_template_kwargs.thinking: false` (or `reasoning_effort: "none"`) to keep it off. The frozen ruler sends both `thinking: false` and `reasoning_effort: "low"` inside `chat_template_kwargs`, and the server keeps thinking off.
- Sampling defaults to DeepSeek's recommendation (model card: temperature 1.0, `top_p` 0.95 or 1.0, [`model-card-sampling.txt`](evidence/s3-default-1m/model-card-sampling.txt); the card's own evaluations ran at 0.95), no top-k. The engine's own default is `top_p` 0.95; `run.sh` passes `--top-p 1.0` (`TOP_P=1.0`), so a request that sends no `top_p` draws at 36.6 tok/s on the ruler's prose prompt (s4). A request that sends a top_p under 1 also draws on the device at `41306d5`, exactly as the host path drew: 31.8 tok/s at 0.95 and at 0.9 on that prompt, about the cost a round of top_p 1.0; its reply takes more rounds (see Sampled decode). At `ec28f35` it read each rank's vocabulary half to the host (15.5 tok/s). Greedy requests are not affected. The family page at the pin predates the device draw: it still says `top_p` 1.0 reads every row to the host and is slow; Sampled decode measures otherwise.
- Both ranks decode each request to `max_tokens` or an end token, one request at a time. Size `max_tokens` per request.
- What happens when one rank dies mid-request is not tested here. Restart both with `./stop.sh && ./run.sh`.

## Memory

GB10 is unified memory: the page cache, the host and the GPU share one pool on each node.

- **Admission.** At startup each rank sizes its caches for the window and allocates everything once; serving allocates no device memory. The engine grants MemAvailable less a reserve of max(4 GiB, a tenth of MemTotal) ([`capacity.py`](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/src/tensorfold/cuda/capacity.py#L160-L165)): 12.1 GiB of 121 GiB on these Sparks ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). The page cache counts as available. An explicit `--context` that does not fit is refused on both ranks.
- **Receipts.** At `--context 65538`: `startup estimate 78.46 GiB within 99.76 GiB` on rank 0 and `within 101.46 GiB` on rank 1, with 73.23 GiB of weights ([`serve_s65538_rank0.log`](evidence/s0-engine-receipts/serve_s65538_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_s65538_rank1.log)). MemAvailable was 115 / 116 GiB before that boot ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt)) and 28 / 30 GiB after the receipt session ([`free_after.txt`](evidence/s0-engine-receipts/s65538/free_after.txt)).
- **Native window.** Without `--context`, both ranks admitted 1,048,576 tokens: `startup estimate 79.09 GiB within 98.95 GiB` / `101.55 GiB` ([`serve_default_rank0.log`](evidence/s0-engine-receipts/serve_default_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_default_rank1.log)). One smoke ran; 23 / 26 GiB were available after it ([`free_after_default.txt`](evidence/s0-engine-receipts/free_after_default.txt)). Through `run.sh` (s2): the same admission, then the bench and needles up to 1,039,528 tokens; the lowest MemAvailable was 22.1 GiB on the head and 26.7 GiB on the worker ([`memwatch.tsv`](evidence/s2-run-sh-ec28f35/ctx1m/memwatch.tsv)).
- **Before a load.** `run.sh` waits until MemAvailable ≥ `MEM_GATE_GIB=100` on each node, up to `MEM_GATE_TIMEOUT` (600 s). It refuses a gate below 92 GiB: the native window's 79.09 GiB plus the 12.1 GiB reserve, rounded up. It does not evict the page cache: the engine counts it as available, and a warm cache loads faster.
- **While serving.** `run.sh` adds `--oom-score-adj` (`OOM_SCORE_ADJ=1000`), `--ulimit core=1` and a memguard. The memguard kills the container when MemAvailable < `MEMGUARD_MIN_AVAIL_MB` (3072) MB and SwapFree < `MEMGUARD_MIN_SWAP_FREE_MB` (2048) MB for 6 s.
- The Engram reads go through the page cache. Rows are never resident and never cross the link.

Read memory with `free -h`, never `nvidia-smi`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1,roceP2p1s0f1
export PORT=8000
```

- **HCAs.** Pin `NCCL_IB_HCA`. Two Sparks on their direct cable expose two RoCE devices for the one port: list both ([TensorFold `RUNBOOK.md`](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/RUNBOOK.md) at the pin). `run.sh` checks that each listed device's port 1 is `ACTIVE`.
- **Rendezvous.** `MASTER_PORT=29571`, distinct from the lab's other recipes. The rendezvous port and the link are not authenticated: keep them on the private link.
- **Caches.** `TF_CACHE/<TF_SHA>` holds the compiled extensions and Triton kernels. A new `TF_SHA` starts a new cache.

## Logs

```bash
docker logs -f tf-dsv41-flash
ssh spark2 docker logs -f tf-dsv41-flash
```

Each rank prints `[tensorfold] CUDA rank R startup estimate …` once admitted. Rank 0 prints one `done req-… tokens=… tok/s=… rounds=… accepted=…` line per request: that is where drafting shows.

## Evidence

Every number above has a file under [`evidence/`](evidence/). `recipe.yaml` names the file per measured row; a row without one renders as "not yet measured on this pair". Sessions:

| Session | What | Verdict |
|---|---|---|
| [`s0-engine-receipts`](evidence/README.md#s0-engine-receipts-2026-10-04) | the engine PR's final receipt run on this pair, at `0d91389`, on the pinned weights: bench_decode at `--context 65538`, a boot at the native window | engine measured; not through `run.sh` |
| [`s1-snapshot-pins`](evidence/README.md#s1-snapshot-pins-2026-10-04) | the Hub listing at `982b704` behind the snapshot pins, and `run.sh`'s snapshot check on the folder the receipts served | pins match the Hub and the receipts' folder |
| [`s2-run-sh-ec28f35`](evidence/README.md#s2-run-sh-ec28f35-2026-10-04) | the engine bumped to `ec28f35`; image built and shipped by `run.sh`; three `run.sh` boots: the decode table, sampled cells, exactness, `bench_openai`, `prefill_cold`, needles to 1,039,528 tokens, quality | measured through `run.sh`; window default 1048576 |
| [`s3-default-1m`](evidence/README.md#s3-default-1m-2026-10-04) | `TOP_P=1.0` (`--top-p 1.0` on both ranks); two `run.sh` boots at the shipped defaults: the decode table, sampled cells with and without sampling fields, exactness of the default request, a quiet `prefill_cold` | default top_p 1.0; decode table from these boots |
| [`s4-device-nucleus`](evidence/README.md#s4-device-nucleus-2026-10-05) | the engine bumped to `41306d5` (top_p cut on the device); image built by `run.sh`; one `run.sh` boot at the shipped defaults: the gate, sampled cells at top_p 1.0, 0.95 and 0.9, exactness at top_p 0.95 and 1.0 against s2 and s3 | top_p 0.95 at 31.8 tok/s (was 15.5); replies equal |

## Gotchas

- **Name the revision.** The repository's `main` holds only the model card. `run.sh` downloads by revision sha.
- **Both nodes hold the whole pack.** Each rank reads its own Engram half from shards 47 and 48 on its own disk.
- **The first start compiles.** Four CUDA extensions build at the first start on each node, into `TF_CACHE/<TF_SHA>`. s2's first boot with an empty cache took 3 min 1 s from `./run.sh` to ready (rank 0 `loaded in 167.9s`); warm boots took about a minute ([`bootA/run.txt`](evidence/s2-run-sh-ec28f35/bootA/run.txt), [`bootB/run.txt`](evidence/s2-run-sh-ec28f35/bootB/run.txt), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). s4's first boot at `41306d5` took 182 s ([`s4 run.txt`](evidence/s4-device-nucleus/run.txt)). Rank 1 waits up to 600 s for rank 0's rendezvous ([`comm.py`](https://github.com/sfxnz/TensorFold/blob/41306d5e2acd5651fc0954609b3a651d2ef61d6e/src/tensorfold/cuda/comm.py#L54)).
- **The kernel cache is root-owned.** The containers run as root, so files under `TF_CACHE/<TF_SHA>` belong to root. Remove an old one through the image, not with sudo: `docker run --rm -v "$TF_CACHE:/c" "$IMAGE" rm -rf /c/<old TF_SHA>`.
- TensorFold opens HTTP only after the model is loaded: "connection refused" means still loading.

## Credits

- Engine: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), with the `deepseek_v41` family of PRs #390 and #391 and the device keyed draw of the fork's `cuda-keyed-draw` branch.
- Weights: [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3), quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT per its model card).
- Harness: `bench_decode.py` and `smoke_chat.py` come from the vLLM sibling recipe, unchanged. The guards, `kit/render.py`, `stop.sh` and the gate follow the lab's Qwen3.8-Flash-Next TensorFold recipe.

## License

The recipe's own files are MIT ([`LICENSE`](LICENSE)). [`NOTICE`](NOTICE) lists the files that come from the lab's other recipes. TensorFold is not included; the image build fetches it at the pinned commit. Model weights follow their own licenses on Hugging Face.
