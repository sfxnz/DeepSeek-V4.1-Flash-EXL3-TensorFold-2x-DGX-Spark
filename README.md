# DeepSeek-V4.1-Flash EXL3 · TensorFold · 2× DGX Spark

Serve [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/tree/2.0bpw-mcg-viterbi-lmhead-mxfp8) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2 with [TensorFold](https://github.com/ashhart/TensorFold)'s `deepseek_v41` CUDA family. Not vLLM. The vLLM recipe for the same weights is [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark); one repo per engine.

**Status: measured through `run.sh` on this pair, against the previous engine and against vLLM in the same session.** Session [`s7-speed`](evidence/README.md#s7-speed-2026-10-10) bumped the engine from `b86514a` to `19f5478`. In one session it booted `./run.sh` at the shipped defaults, with no knob or debug env: a warm-up boot of the new image (W1, not counted), then the old recipe (`38a0607`, engine `b86514a`, boots 1A and 4A) against this branch (engine `19f5478`, boots 2B and 3B) in ABBA order, then the vLLM sibling (5V) on the same cells. All five TensorFold boots ended `GATE=PASS`. Every reply of the new engine equals the old engine's and s6's ([Exactness](#exactness)).

`19f5478` against `b86514a`, same session, decode tok/s at c=1 and aggregate tok/s at c=2 and 4 ([`abba_table.txt`](evidence/s7-speed/abba_table.txt)):

| Cell | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| Distinct prompts, the diverse cell (code / prose) | +3.54% / +2.50% | +11.01% / +14.81% | +16.40% / +23.28% |
| One prompt in every stream, the ruler (prose, structured, prose_long) | +5.59% to +7.39% | +15.76% to +16.13% | +26.67% to +28.59% |
| One prompt, sampled (no field / top_p 0.95) | +3.34% / +4.16% | +15.08% / +14.85% | +28.75% / +28.14% |

- Read the distinct-prompt row first. Streams that all send one prompt route to the same experts and share their weight reads, so same-prompt cells overstate what decoding requests together gains on real traffic (upstream issue [#479](https://github.com/ashhart/TensorFold/issues/479), finding 1). The diverse cell sends a different prompt to every stream ([Before and after](#before-and-after-s7-2026-10-10)).
- Diverse c=1: boot 1A ran slow on those waves (its rounds ran longer than 4A's on 17 of 18 waves; A noise 4.00-5.08%). Against 4A alone the c=1 gain is +0.97% (code) and +0.49% (prose).
- The gain is round time. Every reply has the same rounds and accepted drafts in both arms; on the diverse cell a round takes 10.37-13.21% less time at c=2 and 14.17-19.58% less at c=4.

**What changed in the engine.** Upstream's Python engine ends at 0.6.6, and its `main` is now the Zig engine 1.0.x, which has no DeepSeek-V4 family and no CUDA tensor parallel ([Upstream status](#upstream-status)). So the recipe stays on the fork. `19f5478` is `b86514a` plus upstream's final Python line (`python-0.6` at `ed78d6f`: v0.6.6, API keys, `--name-priority`, the tool-JSON rewrite) and five decode units. They target the decode round, which sets the speed at every concurrency. Each was kept only after its own ABBA in the lab against the fork tip of the day, with every reply's sha, rounds and accepted drafts equal ([`units/`](evidence/s7-speed/units/), decision rows in [`decision.tsv`](evidence/decision.tsv)):

- **P4.** `wo_a` runs its groups in one grouped launch. It ran as one stacked launch over every group that kept only the diagonal blocks.
- **P2.** While a decode forward's all-gathers run, a side-stream kernel warms the dense weights read after each gather into L2, capped at 200 GB/s (it runs at about 89 GB/s) and up to 8 MB a gather.
- **P10.** A greedy request's tokens are drawn on the device inside the verify graph: each row's largest value at its lowest id, read back once. It replaces a host top-k, gather and copy.
- **P1.** Decode rows of most FP8 projections run on tiles picked by shape. Each output keeps its K slices and their order.
- **P6b.** Two or more drafting requests run their DSpark proposals in one block forward, each on its own ring and positions. Its scratch and graphs are made after the startup warm-up. In the lab, c=1 prose read -0.99% (noise 0.92%) in one ABBA, +0.96% in the other and +0.02% pooled over 8 boots ([`P6b-summary.txt`](evidence/s7-speed/units/P6b-summary.txt)); the c=1 rows below are s7's.

Each unit is the engine's default ([family page](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/docs/recipes/deepseek-v4.1-flash.md)). Seven other units were dropped: P3, P5, P7, P8, P9, P11, P12 (one line each in `decision.tsv`).

Earlier sessions, one line each ([evidence/README.md](evidence/README.md#sessions)):

- [`s6-concurrent`](evidence/README.md#s6-concurrent-2026-10-08) (2026-10-08, `b86514a`): concurrent requests, `--parallel 4` against 1, `--decode-share`, the needle beside 3 lanes, vLLM at c=1, 2 and 4.
- [`s5-final`](evidence/README.md#s5-final-2026-10-06) (2026-10-06, `903a1e8`): the speed units C1-C5, C7, C8 and the default draft policy; L.A.I.L, `bench_openai`, quality, NLL, vLLM at c=1.
- [`s4-device-nucleus`](evidence/README.md#s4-device-nucleus-2026-10-05) (2026-10-05, `41306d5`): the device top_p cut.
- [`s3-default-1m`](evidence/README.md#s3-default-1m-2026-10-04) (2026-10-04, `ec28f35`): `--top-p 1.0`, two boots at 1048576.
- [`s2-run-sh-ec28f35`](evidence/README.md#s2-run-sh-ec28f35-2026-10-04) (2026-10-04, `ec28f35`): the first `run.sh` boots, needles to 1,039,528 tokens, quality.
- [`s0-engine-receipts`](evidence/README.md#s0-engine-receipts-2026-10-04), [`s1-snapshot-pins`](evidence/README.md#s1-snapshot-pins-2026-10-04): the engine PR's receipts at `0d91389` and the snapshot pins.

The window default is the native 1048576. The snapshot is the pinned `snapshots/982b704…` folder on both nodes; `run.sh`'s check passed on both at every boot.

- **Engine.** TensorFold `19f5478` from [sfxnz/TensorFold `dsv41-recipe-engine5`](https://github.com/sfxnz/TensorFold/tree/dsv41-recipe-engine5): release v0.6.4 (`6ea5ade`); the `deepseek_v41` family of upstream PRs [#390](https://github.com/ashhart/TensorFold/pull/390) and [#391](https://github.com/ashhart/TensorFold/pull/391) (`e174ee0` has the tree of #391's tip `0d91389`); the device keyed draw for top-k-off rows (`0582d5f`, `3ab53da`; `db7f53e` has the tree of s4's pin `41306d5`); then seven speed units, each merged after its own receipts: C1 skips the decoder rows a prompt's result does not read, C3 a prompt-window EXL3 expert kernel, C2 decode fusion, C7 packed caches and a split decode index selection, C4 Engram I/O off the decode path with resident scale rows, C5 drafts drawn on the device and a confidence draft policy, C8 prompt chunks in two row halves with the kept snapshot taken inside its chunk; then concurrent requests (units N0-N10): up to 4 requests share one verify forward a round, each in its own lane with a whole window, prompts fill between rounds, and every reply equals its solo run (`b86514a`); then upstream's final Python line, `python-0.6` at `ed78d6f` (v0.6.6: API keys, `--name-priority`, the tool-JSON rewrite), merged as `dc971a2`; then five decode units, each kept on its own ABBA: P4 one grouped `wo_a` launch, P10 greedy target draws on the device, P2 a paced L2 warm of the next dense weights, P1 shape-picked MXFP8 decode tiles, P6b DSpark proposals batched across lanes. Each unit is bit-exact, so replies equal `b86514a`'s, and each is the engine's default. s2 and s3 ran `ec28f35`, s4 `41306d5`, s5 `903a1e8`, s6 `b86514a`, s7 `b86514a` against `19f5478`; the decode table below is s7's, at `19f5478`. Apache-2.0. The image builds from [`docker/Dockerfile`](docker/Dockerfile) on `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned).
- **Weights.** Revision `982b70452f399814f56b46272fd30394ae10d58c` (branch `2.0bpw-mcg-viterbi-lmhead-mxfp8`), 48 shards. The routed experts are EXL3 at 2 bits (`mcg` codebook). Everything else is DeepSeek's own bytes: FP8 with 32x32 block scales, an MXFP8 LM head, MXFP4 DSpark draft experts, and the two Engram tables in shards 47 and 48.
- **Engram.** Each rank reads its half of every Engram row from the pack on its own local disk. No repacking, no second copy. Both nodes need the whole revision.
- **Ranks.** Rank 1 runs on `spark2`, rank 0 serves HTTP on `spark1`. NCCL over the QSFP RoCE link.

## Upstream status

TensorFold's Python engine ends at 0.6.6, kept on its [`python-0.6`](https://github.com/ashhart/TensorFold/tree/python-0.6) branch ([ashhart/TensorFold#286](https://github.com/ashhart/TensorFold/issues/286)); new work goes to its Zig engine. For that reason the maintainer closed [#390](https://github.com/ashhart/TensorFold/pull/390), [#391](https://github.com/ashhart/TensorFold/pull/391) and [#408](https://github.com/ashhart/TensorFold/pull/408), and will rewrite the useful parts in Zig, with credit. Upstream `main` is now the Zig engine 1.0.x. It has no DeepSeek-V4 family: its [CHANGELOG](https://github.com/ashhart/TensorFold/blob/main/CHANGELOG.md) (1.0.0) lists DeepSeek-V4 among the families that "remain ports for 1.0.x". On CUDA the native binary registers only Nemotron ([docs/recipes/cuda.md](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/cuda.md)); its CUDA docs show single-host serving only ([RUNBOOK](https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md#linux-and-cuda)), and `--speed-up` two-rank serving is for two Macs ([README](https://github.com/ashhart/TensorFold/blob/main/README.md#serve-flags)), so it has no CUDA tensor parallel either. So this recipe pins the fork: `sfxnz/TensorFold` branch `dsv41-recipe-engine5` at `19f5478`, which `docker/Dockerfile` installs from git at the full commit. It carries upstream's final Python line (`python-0.6` at `ed78d6f`). A later bump goes to another fork commit, or to an upstream release once one serves this family, and needs new evidence.

## Measured on 2× DGX Spark (L.A.I.L lab)

`python3 bench_decode.py`, byte-identical to the vLLM sibling's frozen ruler (sha256 `3172cbc4…`). It sends `ignore_eos` and `chat_template_kwargs {"thinking": false, "reasoning_effort": "low"}`; the server log shows `thinking=False` for every gate request (335 of 335 `done` entries in each of boots 2B and 3B, [`2B`](evidence/s7-speed/2B/gate/docker-head.log), [`3B`](evidence/s7-speed/3B/gate/docker-head.log)). Do not copy community tok/s into this table.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: frozen bench_decode.py (sha256 3172cbc4…), streamed, thinking off, ignore_eos, max_tokens 200, 9 runs, two boots at the shipped defaults (s7 boots 2B and 3B, 2026-10-10: TensorFold `19f5478`, `--context 1048576 --parallel 4 --decode-share 0.5`, `--top-p 1.0`, `--mtp-drafts 5 --mtp-confidence 0.15`, no env override); each row is the median of the two per-boot medians (evidence/s7-speed/decode.txt, which gives each boot's); TP=2 through `./run.sh`, `tools/session_gate.sh` with RUNS=9 RUNS_LONG=9. Concurrency N sends N streams at once, 9 waves: decode is the median stream's rate after its first token, aggregate the median wave's tokens over its wall time, TTFT p50 over every stream (runs 2 to 9 resume the kept prompt). prose, structured and prose_long are the ruler's greedy cells. The sampled rows send the ruler's prose prompt with no sampling field (the server's temperature 1.0, top_p 1.0, top-k off) or top_p 0.95, no seed, so every stream draws the same reply (evidence/s7-speed/scripts/sampled_conc.py). Every stream sends one prompt, which overstates what lanes gain on distinct prompts; the diverse cell below the table sends a different prompt to each stream. Before (s7 boots 1A and 4A, TensorFold `b86514a`, the same session; c=1 decode, c=2 and 4 aggregate): prose 56.12 / 82.35 / 116.01, structured 97.88 / 144.36 / 200.62, prose_long 40.93 / 60.81 / 86.56 (evidence/s7-speed/abba_table.txt).

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose (note 1) | 1 | 59.3 | 59.2 | 0.07 s |
| prose | 2 | 48.0 | 95.3 | 0.13 s |
| prose | 4 | 37.3 | 148.7 | 0.25 s |
| structured | 1 | 105.1 | 105.1 | 0.07 s |
| structured | 2 | 84.5 | 167.6 | 0.13 s |
| structured | 4 | 63.8 | 254.1 | 0.25 s |
| prose_long | 1 | 43.3 | 43.3 | 0.07 s |
| prose_long | 2 | 35.3 | 70.6 | 0.13 s |
| prose_long | 4 | 28.0 | 111.3 | 0.25 s |
| sampled (no field) | 1 | 44.6 | 44.6 | 0.07 s |
| sampled (no field) | 2 | 36.5 | 73.0 | 0.13 s |
| sampled (no field) | 4 | 28.9 | 115.8 | 0.26 s |
| sampled (top_p 0.95) | 1 | 39.5 | 39.5 | 0.07 s |
| sampled (top_p 0.95) | 2 | 33.0 | 65.8 | 0.13 s |
| sampled (top_p 0.95) | 4 | 26.5 | 106.0 | 0.26 s |

1. The prose reply ends at its end token after 74 tokens; ignore_eos decodes the other 126 (post_eos_fraction 0.63).
<!-- END generated measured -->

`run.sh` serves `--parallel 4`: up to 4 requests decode together, each in its own lane holding a whole 1,048,576-token window, and each reply equals the same request served alone ([Concurrent requests](#concurrent-requests)). A fifth request waits for a free lane.

- Per boot (2B / 3B), c=1: prose 59.40 / 59.12, structured 104.92 / 105.32, prose_long 43.16 / 43.41 tok/s ([`decode.txt`](evidence/s7-speed/decode.txt)). The ruler is greedy, so `--top-p` does not touch it.
- Inter-chunk time, p50 / p90 over the ruler's cells: 49-56 / 56-58 ms at c=1, 60-68 / 68-74 ms at c=2, 76-90 / 86-98 ms at c=4. A chunk carries a round's accepted tokens. TTFT p50 0.07 s at c=1, 0.13 s at c=2, 0.25-0.26 s at c=4: each prompt fills between the others' rounds.
- Earlier tables, c=1 prose / structured / prose_long: s6 at `b86514a` 56.3 / 97.7 / 41.2 ([`s6 decode.txt`](evidence/s6-concurrent/decode.txt)); s5 at `903a1e8` 55.8 / 97.3 / 40.7 ([`s5 decode.txt`](evidence/s5-final/decode.txt)); s3 at `ec28f35` 46.0 / 64.1 / 34.6 ([`s3 decode.txt`](evidence/s3-default-1m/decode.txt)); the engine's receipts at `0d91389` 45.5 / 63.5 / 34.0 ([`s0`](evidence/s0-engine-receipts/s65538/bench_decode.log)).

### Before and after (s7, 2026-10-10)

One session, the same clients and the same serve argv in both arms ([`1A/serve-argv.txt`](evidence/s7-speed/1A/serve-argv.txt), [`2B/serve-argv.txt`](evidence/s7-speed/2B/serve-argv.txt)). A = `b86514a` (boots 1A, 4A, `./run.sh` at `38a0607`), B = `19f5478` (2B, 3B, this branch). Each arm is the median of its two per-boot medians; noise is |1A − 4A| / A ([`abba_table.txt`](evidence/s7-speed/abba_table.txt), made by [`table.py`](evidence/s7-speed/scripts/table.py)). Each arm builds and loads only its own kernels, under `TF_CACHE/<TF_SHA>` ([`extensions.txt`](evidence/s7-speed/extensions.txt)).

**Distinct prompts: the diverse cell.** Each stream of a wave gets a different prompt: 8 prose openings to continue and 8 small code tasks, greedy, `ignore_eos`, 200 tokens, 9 waves at c=1, 2 and 4 ([`bench_diverse.py`](evidence/s7-speed/scripts/bench_diverse.py)). The waves alternate between fixed prompt sets, so a boot's plain median can land in either set (3B's code c=4 reads 143.07 against 2B's 153.69). Each boot is therefore read as the mean over prompt sets of each set's median wave (section 1b). Aggregate tok/s:

| Phase | c | `b86514a` | `19f5478` | Change | Noise | Time a round |
|---|---:|---:|---:|---:|---:|---:|
| code | 1 | 75.94 | 78.62 | +3.54% | 5.08% | −3.43% |
| code | 2 | 101.03 | 112.15 | +11.01% | 0.03% | −10.37% |
| code | 4 | 127.03 | 147.86 | +16.40% | 0.06% | −14.17% |
| prose | 1 | 40.93 | 41.96 | +2.50% | 4.00% | −2.43% |
| prose | 2 | 58.59 | 67.27 | +14.81% | 0.43% | −13.21% |
| prose | 4 | 77.54 | 95.60 | +23.28% | 0.81% | −19.58% |

- c=1 is the one cell where A's noise exceeds 2.67%: 1A's rounds ran longer than 4A's on 17 of 18 waves. Against 4A alone: code +0.97%, prose +0.49%.
- At c=2 and c=4 the diverse gain is smaller than the same-prompt gain below (code against structured: +11.01% / +16.40% against +16.08% / +26.67%). The plan expected this: distinct prompts load more distinct experts a round, so a round costs more and the units' savings are a smaller share of it.
- Every prompt drew one reply at every c and in every boot (`SHA_CHECK` 16 of 16, [`2B/diverse.txt`](evidence/s7-speed/2B/diverse.txt) per boot).

**One prompt in every stream: the ruler and the sampled cells.** Decode tok/s at c=1, aggregate tok/s at c=2 and 4:

| Cell | c | `b86514a` | `19f5478` | Change | Noise |
|---|---:|---:|---:|---:|---:|
| prose (greedy, 200 tokens) | 1 | 56.12 | 59.26 | +5.59% | 0.22% |
| | 2 | 82.35 | 95.33 | +15.76% | 0.02% |
| | 4 | 116.01 | 148.65 | +28.14% | 0.75% |
| structured | 1 | 97.88 | 105.12 | +7.39% | 0.26% |
| | 2 | 144.36 | 167.58 | +16.08% | 0.37% |
| | 4 | 200.62 | 254.13 | +26.67% | 0.10% |
| prose_long | 1 | 40.93 | 43.29 | +5.76% | 0.34% |
| | 2 | 60.81 | 70.62 | +16.13% | 0.97% |
| | 4 | 86.56 | 111.31 | +28.59% | 2.67% |
| sampled, no sampling field | 1 | 43.17 | 44.61 | +3.34% | 0.23% |
| | 2 | 63.40 | 72.96 | +15.08% | 0.18% |
| | 4 | 89.96 | 115.82 | +28.75% | 0.52% |
| sampled, top_p 0.95 | 1 | 37.96 | 39.55 | +4.16% | 0.23% |
| | 2 | 57.28 | 65.79 | +14.85% | 0.11% |
| | 4 | 82.69 | 105.96 | +28.14% | 0.24% |

Files: `<boot>/gate/bench-*.out`, `<boot>/sampled-*.txt`, `<boot>/diverse.txt` under [`evidence/s7-speed/`](evidence/s7-speed/).

### Against vLLM, same session (s7)

The vLLM sibling booted after the TensorFold boots with its canonical command: `env AUDIT=strict ./run.sh` at `3de9146`, image `dsv41-flash-exl3-sm121:canonical-e14`, `audit ok` on both ranks, `MAX_NUM_SEQS=2`. One boot, the same clients, then `docker stop -t 60` and `docker rm` on both nodes ([`5V/`](evidence/s7-speed/5V/)). vLLM decodes 2 sequences at a time, so in its c=4 waves 2 streams wait for the first 2. TensorFold is `19f5478` (2B, 3B). Decode tok/s at c=1, aggregate at c=2 and 4; per stream is the median stream's decode rate (diverse: the plain median over waves):

| Cell | c | TensorFold | vLLM | TF / vLLM | TF per stream | vLLM per stream |
|---|---:|---:|---:|---:|---:|---:|
| prose (greedy, 200 tokens) | 1 | 59.26 | 49.68 | 1.193 | | |
| | 2 | 95.33 | 78.35 | 1.217 | 48.02 | 40.84 |
| | 4 | 148.65 | 104.22 | 1.426 | 37.27 | 40.47 |
| structured | 1 | 105.12 | 83.79 | 1.255 | | |
| | 2 | 167.58 | 156.30 | 1.072 | 84.49 | 78.19 |
| | 4 | 254.13 | 201.83 | 1.259 | 63.76 | 78.14 |
| prose_long | 1 | 43.29 | 44.90 | 0.964 | | |
| | 2 | 70.62 | 71.82 | 0.983 | 35.31 | 35.91 |
| | 4 | 111.31 | 89.04 | 1.250 | 27.99 | 34.70 |
| sampled, no sampling field | 1 | 44.61 | 48.15 | 0.926 | | |
| | 2 | 72.96 | 68.76 | 1.061 | 36.52 | 36.52 |
| | 4 | 115.82 | 78.32 | 1.479 | 28.90 | 34.16 |
| sampled, top_p 0.95 | 1 | 39.55 | 42.80 | 0.924 | | |
| | 2 | 65.79 | 63.90 | 1.029 | 33.03 | 34.44 |
| | 4 | 105.96 | 88.36 | 1.199 | 26.50 | 36.24 |
| diverse code (matched sets) | 1 | 78.62 | 71.08 | 1.106 | | |
| | 2 | 112.15 | 102.99 | 1.089 | 62.66 | 54.42 |
| | 4 | 147.86 | 131.31 | 1.126 | 40.86 | 55.68 |
| diverse prose (matched sets) | 1 | 41.96 | 42.41 | 0.989 | | |
| | 2 | 67.27 | 62.14 | 1.082 | 34.30 | 31.95 |
| | 4 | 95.60 | 78.22 | 1.222 | 24.90 | 31.78 |

- c=4: TensorFold leads on every cell, x1.126 to x1.479 aggregate. Per stream it decodes slower than vLLM's 2 running streams; vLLM queues the other 2, which shows in its TTFT.
- c=2: TensorFold leads on every cell but prose_long (0.983), x1.029 to x1.217.
- c=1: TensorFold leads on prose (1.193), structured (1.255) and diverse code (1.106). It trails on prose_long (0.964), diverse prose (0.989) and the sampled cells (0.926, 0.924).
- TTFT p50: TensorFold 0.07 s at c=1, 0.13 s at c=2, 0.25-0.26 s at c=4. vLLM 0.21-0.27 s, 0.23-0.32 s and 1.6-3.2 s ([`5V/bench-frozen.out`](evidence/s7-speed/5V/bench-frozen.out), [`5V/bench-prose-long.out`](evidence/s7-speed/5V/bench-prose-long.out)).
- vLLM's greedy diverse replies change with the batch: 11 of 16 prompts drew more than one reply across c=1, 2 and 4 ([`5V/diverse.out`](evidence/s7-speed/5V/diverse.out)). Its speeds are still read.
- vLLM draws a new reply each sampled run. TensorFold seeds an unseeded request from its prompt, so every sampled stream draws one reply. The sampled cells time different text.
- s6 and s5 ran the same comparison at `b86514a` and `903a1e8` ([`s6 derived.txt`](evidence/s6-concurrent/derived.txt), [`s5 derived.txt`](evidence/s5-final/derived.txt)). s5 also has L.A.I.L, `bench_openai` and prefill against vLLM, at c=1.

### Exactness

The decode units are bit-exact, so every reply at `19f5478` must equal `b86514a`'s. In s7 none differed ([`abba_table.txt`](evidence/s7-speed/abba_table.txt) section 2):

- **Every reply of a boot.** Rank 0's `done` entries ([`rank0-full.log`](evidence/s7-speed/3B/rank0-full.log) per boot): 603 replies in each of 1A, 3B and 4A, the same multiset of (prompt tokens, completion tokens, sha), and the same rounds and accepted drafts. 2B served the same and 31 more: the needle and its lanes.
- **Against s6.** Every sha at a (prompt, tokens) key that s6 also served is in s6's 2B and 3B logs (24 to 26 shared keys a boot).
- **Concurrent against alone.** `tools/bench_concurrent.py`, plain and `--mixed`: every request equals its alone run, in every boot ([`conc-check.out`](evidence/s7-speed/2B/gate/conc-check.out)). `tools/pairs_concurrent.py` (4 staggered prompts, greedy and keyed): 0 of 4 unequal for 2B and 3B against 1A, 4A and s6's 2B.
- **Sampled cells.** Token hash `4e0fa4daff89` (no field) and `dae740142762` (top_p 0.95) at c=1, 2 and 4 in every boot, the same as s6.
- **Diverse cell.** Each prompt drew one sha at every c in every boot (16 of 16), and the same sha across boots.

Earlier checks: drafted replies equal `"draft": false` at `903a1e8` (s5, 6 of 6 a boot, [`bootE/pairs.jsonl`](evidence/s5-final/bootE/pairs.jsonl)) and at `ec28f35` (s2, 13 of 13, [`pairs.jsonl`](evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl)); s4 the device top_p cut against the host path ([`pairs.jsonl`](evidence/s4-device-nucleus/pairs.jsonl)).

### Concurrent requests

Lanes: `PARALLEL=4` lanes, each holding a whole `CONTEXT` window (4 x 1,048,576 tokens), allocated at startup. A request takes a free lane and keeps it to its end; while all 4 are busy, further requests wait in the queue. Each round runs one shared verify forward over every decoding lane, and since `19f5478` one DSpark block forward for every drafting lane. A new prompt fills between those rounds, and the rounds take `DECODE_SHARE` (0.5) of each prompt span's time.

- **The needle beside 3 lanes** (s7, boot 2B, [`needle_lanes.jsonl`](evidence/s7-speed/2B/needle_lanes.jsonl)): 1,039,528 prompt tokens, found (`CZP-3128-QL`), token hash `4dfbacf6d86b`, the same as s6's. Answered in 1367.3 s (s6: 1370.6 s; s5, alone at `903a1e8`: 756.6 s). The 3 lanes ran 30 replies with 0 errors, all hash `b12cd40d927d` (s6's). Those overlapping the needle decoded at a median 10.04 tok/s (s6: 8.43).
- **A prompt arriving mid-decode** (s6, `b86514a`; not re-run at `19f5478`): a 16,386-token prompt while 3 lanes decode reaches its first token in 17.98 / 18.66 s at `--decode-share 0.5` (9.58 s alone), 14.98 s at 0.25 and 24.04 s at 1. The live lanes decode at about 7.2, 4.2 and 11.3 tok/s meanwhile ([`s6 abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 4).
- A request sent with `"priority": "background"` (or a session-title request) yields its lane when all 4 are busy and a prompt waits. It replays its whole reply later and sends only the tokens it had not sent; its reply equals its solo run (s6).

**Limits.**

- At most 4 requests decode at once (the family's limit). More wait for a free lane.
- A client that disconnects stops its request after the next round. One that disconnects while its prompt fills is noticed only at its first token: the fill runs to its end and holds its lane until then.
- `thinking_budget` stays refused (HTTP 400) at every `--parallel`, 2 to 4 included.
- Under `--parallel` 2 or more, a reply's `prefill_s` (its `tensorfold` stats and rank 0's `done … prefill=` field) counts only the time spent launching prompt spans: the needle above read 48.19 s of 1367.3 s. Time prompts from the client, or read `ttft=`.
- A routed-expert fp16 overflow in one lane, while the first forwards still check for it, fails that round for every live request (the family page). A failure on one rank in the middle of a round is not recovered: restart both with `./stop.sh && ./run.sh`.

### Prefill

The engine's `tools/prefill_cold.py` ([copy](evidence/s7-speed/scripts/prefill_cold.py)) in every counted s7 boot, median of 3 prompts a length, prompt tok/s, each arm the median of its two boots ([`abba_table.txt`](evidence/s7-speed/abba_table.txt) section 4, `<boot>/prefill_cold.json`):

| Prompt tokens | `b86514a` (1A / 4A) | `19f5478` (2B / 3B) | Change |
|---:|---:|---:|---:|
| 2,048 | 1138.8 (1135.3 / 1142.3) | 1139.8 (1139.7 / 1139.8) | +0.08% |
| 8,192 | 1586.3 (1573.8 / 1598.9) | 1572.7 (1567.6 / 1577.8) | −0.86% |
| 16,384 | 1716.0 (1708.2 / 1723.8) | 1682.7 (1708.0 / 1657.3) | −1.94% |
| 32,768 | 1770.6 (1746.3 / 1794.9) | 1759.0 (1778.1 / 1740.0) | −0.65% |
| 65,536 | 1783.2 (1751.9 / 1814.4) | 1704.4 (1795.7 / 1613.1) | −4.42% |

- No regression is established. The 64k shortfall comes from 3B alone (1613.1); 2B's 1795.7 sits inside A's 1751.9-1814.4 and next to s6's boots (1781.5-1830.1). The decode units do not target the prompt path.
- Against vLLM (s5, `903a1e8`): x1.45 at 2k to x2.35 at 64k ([`s5 derived.txt`](evidence/s5-final/derived.txt)). Against s3 at `ec28f35`: x2.2 to x3.1; C1, C3 and C8 are the units on the prompt path.

### Long prompts

`CONTEXT=1048576` boots through `run.sh` at `19f5478`: `startup estimate 82.56 GiB within 98.59 GiB` on rank 0 and `within 102.01 GiB` on rank 1 in boot 2B ([`2B startup.txt`](evidence/s7-speed/2B/startup.txt)); 98.52-98.59 and 101.73-102.01 GiB over W1, 2B and 3B ([`abba_table.txt`](evidence/s7-speed/abba_table.txt) section 6). The 1,039,528-token needle was found alone at `903a1e8` after 754.3 s of prefill (s5, [`needle.jsonl`](evidence/s5-final/bootF/needle.jsonl)), and beside 3 lanes at `b86514a` (s6) and `19f5478` (s7, above). s2 found needles at 130k, 258k, 524k and 1,039,528 tokens at `ec28f35` ([`needles.jsonl`](evidence/s2-run-sh-ec28f35/ctx1m/needles.jsonl), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). Memory stays flat after admission: every lane's window is allocated at startup. Tested windows: 65538 (s2) and 1048576 (s2 to s7).

### Draft policy

`run.sh` passes `--mtp-drafts 5 --mtp-confidence 0.15`: the engine's default since C5, written out so a bump cannot change it silently. Each round DSpark drafts up to 5 tokens, and the chain stops where the drafts' confidence product falls under 0.15. Against 3 drafts every round, the C5 receipts (dev launcher, `--context 65538`, not `run.sh`) measured prose x1.098, structured x1.394, L.A.I.L x1.095, code x1.110 and non-English text x1.072 to x1.142; one 64-token greedy chat reply lost (x0.926). Every case drew the same reply under both policies (decision `DRAFTS` in [`decision.tsv`](evidence/decision.tsv), [`c5-draft-policy/abba_table.txt`](evidence/s5-final/c5-draft-policy/abba_table.txt)).

### Sampled decode (top-k off)

The ruler is greedy. The sampled rows send the ruler's prose prompt with sampling ([`sampled_conc.py`](evidence/s7-speed/scripts/sampled_conc.py)): no sampling field (the server's temperature 1.0, top_p 1.0) decodes at 44.6 tok/s at c=1, top_p 0.95 at 39.5, a reply that takes more rounds (99 against 86 in s4). top_p under 1 draws on the device since `41306d5`; at `ec28f35` it read each rank's vocabulary half to the host (15.5 tok/s on this prompt, s3). The engine still takes the host path past the cut's `TMAX` bound and wherever the device draw cannot decide; the logs do not count those rows ([`s4 derived.txt`](evidence/s4-device-nucleus/derived.txt)). top_k 20: 42.55 tok/s at `903a1e8` (s5, [`derived.txt`](evidence/s5-final/derived.txt)).

### Quality

s5, boot F at `903a1e8` ([`quality_full.txt`](evidence/s5-final/bootF/quality_full.txt)): the vLLM sibling's `quality_eval.py --full` through the lab's `qe_tf.py`, gated against vLLM's round-36 baseline on this pack: pass. GSM8K 97/100, with thinking 39/40; MMLU 204/228; tools exact_args 22/22; needles 9/9 at 8k, 32k and 128k; selfcons 12 of 12. Teacher-forced NLL on 40 passages: 0.137069, all 80 score files bit-identical to s2's ([`nll_compare.txt`](evidence/s5-final/nll/nll_compare.txt)). Not re-run since; every reply s6 and s7 checked equals the earlier pins' on the cells both served.

### Where TensorFold trails

- c=1 against vLLM (s7): prose_long 0.964, diverse prose 0.989, sampled 0.926 (no field) and 0.924 (top_p 0.95).
- c=2: prose_long 0.983 (aggregate); per stream at c=2 also prose_long (35.31 against 35.91) and sampled top_p 0.95 (33.03 against 34.44). Per stream at c=4 on every cell: vLLM's 2 running streams decode faster while it queues the other 2.
- Short chat replies at c=1 (s5, `903a1e8`): `bench_openai` t=1 0.973 and 0.962, a 64-token greedy chat reply 0.876.
- No images, structured outputs, `logprobs` or named tool choice ([Not supported](#not-supported-this-engine-two-ranks)). vLLM serves images on the same pack.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (`10.100.8.1` / `10.100.8.2` in this lab), both RoCE ports `ACTIVE`.
- Docker + NVIDIA Container Toolkit on both nodes. SSH from the head to the worker (`spark2` here).
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up; `run.sh` refuses to.
- Disk on each node: the whole revision, 333 GiB ([`pins.json`](evidence/s1-snapshot-pins/pins.json), [family page](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/docs/recipes/deepseek-v4.1-flash.md#download-on-both-machines)), in the HF cache (`HF_CACHE`, default `~/.cache/huggingface`) on local NVMe, never NFS. Other revisions already in the same cache entry add to that. Plus the kernel cache under `TF_CACHE`.

## Weights

Order: download on the head, copy to the worker, then `./run.sh`.

1. **Head.** `tensorfold pull` and a plain `hf download` take the repository's `main`, which holds only the model card, so name the revision:

   ```bash
   hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 982b70452f399814f56b46272fd30394ae10d58c
   ```

   `./run.sh` runs that download itself on the head when the snapshot is missing or incomplete, after it checks the free space, and before it checks the worker's copy.
2. **Worker.** The worker needs no internet and never downloads: images and weights come from the head. Copy only the pinned snapshot from the head over the link, on the worker:

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

`run.sh` builds `tf-dsv41-flash:0.6.6-19f5478` from [`docker/Dockerfile`](docker/Dockerfile) on the head when it is missing. It copies the image to the worker over the link when the worker's image ID differs (`docker save | ssh docker load`). Do both before the downtime, while another serve still holds the pair:

```bash
IMAGE_ONLY=1 ./run.sh   # build on the head, copy to the worker; no GPU, no weights, no memory gate
```

- The build runs `pip install "tensorfold @ git+TF_REPO@TF_SHA"` under [`docker/constraints.txt`](docker/constraints.txt). It then checks the commit pip recorded (`direct_url.json`) against `TF_SHA` and `TF_REPO`.
- The constraints pin TensorFold's runtime deps at the versions expected in the base image. The build fails if pip changes any installed package: a wrong pin fails the build, not a serve, and torch and triton stay the base's. s2's build passed that check: pip changed nothing ([`image-only.txt`](evidence/s2-run-sh-ec28f35/image-only.txt)). `IMAGE_ONLY=1 ./run.sh` took 3 min 34 s, most of it the copy to the worker ([`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). s4's build at `41306d5` took 208 s ([`image-only.txt`](evidence/s4-device-nucleus/image-only.txt)); s5's at `903a1e8` passed the same checks and took 214 s, the same image ID on both nodes ([`image-only.txt`](evidence/s5-final/image-only.txt), [`image-ids.txt`](evidence/s5-final/image-ids.txt), [`derived.txt`](evidence/s5-final/derived.txt)). s7's at `19f5478` installed tensorfold 0.6.6 and changed no other package, in 3 min 27 s, the same image ID on both nodes ([`image-only.txt`](evidence/s7-speed/image-only.txt)).
- No extras: the family serves no grammars and no images.
- Label `tensorfold.sha`. `run.sh` refuses an image whose label differs from `TF_SHA`.
- Seven CUDA extensions compile at the first start, into `TF_CACHE/<TF_SHA>` on each node (five at `b86514a`; `19f5478` adds `gemv_v2` and `l2warm_v1` and moves `nvfp4` from v3 to v4, [`extensions.txt`](evidence/s7-speed/extensions.txt)).
- To bump the engine, set `TF_REPO` and `TF_SHA` in `recipe.yaml` (`engine`), render, rebuild, and measure again (Upstream status, above).

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
| Engine | TensorFold `19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb` (v0.6.6 + the deepseek_v41 CUDA family, its speed units, concurrent requests and the decode units P4, P2, P10, P1, P6b) from `https://github.com/sfxnz/TensorFold.git`, built into `tf-dsv41-flash:0.6.6-19f5478` from `docker/Dockerfile` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-EXL3` at `982b70452f399814f56b46272fd30394ae10d58c`, served from the HF cache `$HOME/.cache/huggingface` (read-only) on each node |
| Snapshot check | `config.json` sha256 `6469adab394edead3eec148323e7471582d08acdf60a438bf0c9e815b69f36c5`, index sha256 `91731e4af38696bd4c09e960f4b599d1d49f35d445e4f88a43cc355d9e139f03`, 48 shards, 357466041064 shard bytes, every shard's header ending at its size |
| Ranks | `--tp 2`: rank 1 on `spark2` first, then rank 0 (HTTP) on the head; rendezvous `10.100.8.1:29571` |
| `--context` | 1048576, the model's native window (tested through run.sh: needles at 130k, 258k, 524k and 1,039,528 prompt tokens, all found) |
| Concurrency | `--parallel 4`: up to 4 requests decode together, each lane holding a whole `--context` window allocated at startup; more requests wait for a free lane. `--decode-share 0.5`: a new prompt fills between the other lanes' decode rounds, which take 0.5 of each prompt span's time (passed only at `PARALLEL` 2 or more) |
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

The serve flags match the receipt run's (`--tp 2 --master … --name deepseek-ai/DeepSeek-V4.1-Flash --no-thinking --no-update-check` on both ranks, and `--context`, 65538 there, [`environment.txt`](evidence/s0-engine-receipts/environment.txt)). `run.sh` also passes `--parallel 4 --decode-share 0.5` and `--mtp-drafts 5 --mtp-confidence 0.15` to both ranks, and rank 0 also gets `--max-tokens 4096`. The drafting values and the reply cap are the engine's and the CLI's defaults, written out so a bump cannot change them silently ([`serve-argv.txt`](evidence/s7-speed/2B/serve-argv.txt); the receipts ran 3 drafts a round, the default before C5, see Draft policy). Both ranks also get `--top-p 1.0`, which the receipts did not pass (their server default was 0.95; see Sampled decode). Rank 0 prints it: `serving … (sampling: top_p 1.0; …)` ([`startup.txt`](evidence/s7-speed/2B/gate/startup.txt)). A request that omits `temperature` gets the engine's 1.0 ([`cuda/server.py`](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/src/tensorfold/cuda/server.py#L95) at the pin; the family sets top-k off, [`app.py`](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/src/tensorfold/families/deepseek_v41/cuda/app.py#L36)). The container env adds `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, as the receipt run had: it keeps fp32 matmuls in fp32 whatever TF32 default the NGC image sets.

`run.sh` refuses, before `VALIDATE_ONLY` exits:

- non-decimal or zero-padded integers, a `TF_SHA` / `SNAPSHOT_SHA` that is not 40 hex, a pinned sha256 that is not 64 hex, a `TF_REPO` that is not an https `.git` URL, a cache path that is relative or has a space or `:`
- `TP` other than 2, `CONTEXT` above the native 1048576, `PARALLEL` outside 1 to 4, `DECODE_SHARE` outside [0, 1], `MTP_DRAFTS` above 5, `MTP_CONFIDENCE` outside (0, 1], `TOP_P` outside (0, 1], `PORT` equal to `MASTER_PORT`
- `MEM_GATE_GIB` below the floor `run.sh` computes from `PARALLEL` and `CONTEXT`, unless `FORCE_UNSAFE_MEM_GATE=1` for one boot: it refuses below 95 at the defaults, 94 at `--parallel 1` (Memory, below)
- a `CONTAINER_NAME` that does not match `tf-dsv41-[a-z0-9-]+`
- `TF_DSV41_CACHE_GIB` / `TF_DSV41_CACHE_ENTRIES` that are not numbers; an exported `TF_DSV41_*` variable that is not a `recipe.yaml` knob or in `EXTRA_ENV` (it would never reach the containers)
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds (`--tp`, `--rank`, `--master*`, `--host`, `--port`, `--name`, `--context`, `--parallel`, `--decode-share`, `--mtp-*`, `--no-drafts`, `--thinking`, `--no-thinking`, `--max-tokens`, `--top-p`, `--no-update-check`), or a prefix of one; flags the family refuses or ignores (`--prefill-fp8`, `--drafter`, `--thinking-budget`, `--vision*`, `--kv-dtype`)
- `EXTRA_ENV` that is not `KEY=VALUE`, or that sets a variable `run.sh` already sets (every `recipe.yaml` key, `NCCL_IB_HCA`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, …)

`BENCH_ONLY=1` binds the API to 127.0.0.1. The server's top_p is `TOP_P=1.0`, and `EXTRA_ARGS` may not set `--top-p`; `--temperature` and `--top-k` go through `EXTRA_ARGS`; other engine variables through `EXTRA_ENV` (`EXTRA_ENV="TENSORFOLD_MEMORY_RESERVE_GIB=8"`).

## Not supported (this engine, two ranks)

From the family's [recipe page](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/docs/recipes/deepseek-v4.1-flash.md) at the pinned commit:

- Other exports of the model. The family checks the checkpoint at startup and refuses EXL3 outside the routed experts or a BF16 LM head; other packs do not load.
- One rank, one GPU, or a separate `--drafter`.
- More than 4 requests decoding at once. `--parallel` takes 1 to 4; further requests queue for a free lane.
- Images, `response_format` and the `guided_*` and `structured_outputs` fields, `logprobs`, `tool_choice: "required"` or a named function, `thinking_budget` (at every `--parallel`, 2 to 4 included), and `n` above 1: HTTP 400.
- `--prefill-fp8`: refused. Prompt matmuls take bf16 activations.
- Tool calls arrive whole in the final chunk of a streamed reply, with `finish_reason: "tool_calls"`.
- An effort name turns thinking on. A client that sends `reasoning_effort: "low"` to this `--no-thinking` server gets thinking. Send `chat_template_kwargs.thinking: false` (or `reasoning_effort: "none"`) to keep it off. The frozen ruler sends both `thinking: false` and `reasoning_effort: "low"` inside `chat_template_kwargs`, and the server keeps thinking off.
- Sampling defaults to DeepSeek's recommendation (model card: temperature 1.0, `top_p` 0.95 or 1.0, [`model-card-sampling.txt`](evidence/s3-default-1m/model-card-sampling.txt); the card's own evaluations ran at 0.95), no top-k. The engine's own default is `top_p` 0.95; `run.sh` passes `--top-p 1.0` (`TOP_P=1.0`), so a request that sends no `top_p` draws at 42.8 tok/s on the ruler's prose prompt (s5). A request that sends a top_p under 1 also draws on the device (since `41306d5`), exactly as the host path drew: 37.7 tok/s at 0.95 on that prompt, a reply that takes more rounds (see Sampled decode). At `ec28f35` it read each rank's vocabulary half to the host (15.5 tok/s). Greedy requests are not affected. The family page at the pin predates the device draw: it still says `top_p` 1.0 reads every row to the host and is slow; Sampled decode measures otherwise.
- Both ranks decode each request to `max_tokens` or an end token. A client that disconnects stops its request after the next round; one that disconnects while its prompt fills is noticed at its first token, so the fill runs to its end first. Size `max_tokens` per request.
- What happens when one rank dies mid-request is not tested here. Restart both with `./stop.sh && ./run.sh`.
- A failure on one rank in the middle of a round (the other rank still in its collectives) is not recovered: the pair stops serving. Restart both with `./stop.sh && ./run.sh`.

## Memory

GB10 is unified memory: the page cache, the host and the GPU share one pool on each node.

- **Admission.** At startup each rank sizes its caches for the window and allocates everything once; serving allocates no device memory. The engine grants MemAvailable less a reserve of max(4 GiB, a tenth of MemTotal) ([`capacity.py`](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/src/tensorfold/cuda/capacity.py#L163-L168)): 12.1 GiB of 121 GiB on these Sparks ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). The page cache counts as available. An explicit `--context` that does not fit is refused on both ranks.
- **Receipts.** At `--context 65538`: `startup estimate 78.46 GiB within 99.76 GiB` on rank 0 and `within 101.46 GiB` on rank 1, with 73.23 GiB of weights ([`serve_s65538_rank0.log`](evidence/s0-engine-receipts/serve_s65538_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_s65538_rank1.log)). MemAvailable was 115 / 116 GiB before that boot ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt)) and 28 / 30 GiB after the receipt session ([`free_after.txt`](evidence/s0-engine-receipts/s65538/free_after.txt)).
- **Native window.** Without `--context`, both ranks admitted 1,048,576 tokens: `startup estimate 79.09 GiB within 98.95 GiB` / `101.55 GiB` ([`serve_default_rank0.log`](evidence/s0-engine-receipts/serve_default_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_default_rank1.log)). One smoke ran; 23 / 26 GiB were available after it ([`free_after_default.txt`](evidence/s0-engine-receipts/free_after_default.txt)). Through `run.sh` (s2): the same admission, then the bench and needles up to 1,039,528 tokens; the lowest MemAvailable was 22.1 GiB on the head and 26.7 GiB on the worker ([`memwatch.tsv`](evidence/s2-run-sh-ec28f35/ctx1m/memwatch.tsv)). At `903a1e8` (s5): `startup estimate 81.32 GiB within 99.54 GiB` / `102.12 GiB`, with 76.07 GiB of weights (C4 keeps the Engram scale rows resident); over boot F, the bench cells, the 1,039,528-token needle and the full quality set, the lowest MemAvailable was 15.0 GiB on the head and 23.7 GiB on the worker ([`startup.txt`](evidence/s5-final/bootE/gate/startup.txt), [`memwatch.tsv`](evidence/s5-final/bootF/memwatch.tsv), [`derived.txt`](evidence/s5-final/derived.txt)). At `b86514a` with `--parallel 4` (s6): `startup estimate 82.55 GiB` for 4 windows of 1,048,576, 81.32 GiB at one lane; the lowest MemAvailable over the 4-lane boots was 17.42 GiB on the head, during the needle beside 3 lanes, and 21.90 GiB on the worker ([`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 5). At `19f5478` (s7): `startup estimate 82.56 GiB` for 4 windows of 1,048,576 on both ranks, 0.01 GiB more for the batched proposals' scratch (8.81 MiB, [`P6b-summary.txt`](evidence/s7-speed/units/P6b-summary.txt)). MemAvailable, every 2 s on both nodes ([`memwatch.tsv`](evidence/s7-speed/2B/memwatch.tsv) per boot): at least 16.79 GiB on the head (at the needle's start) and 21.54 GiB on the worker over 2B and 3B; 18.58 / 22.09 GiB over the `b86514a` boots 1A and 4A; 12.13 / 15.76 GiB during W1's first start, while the kernels compiled ([`abba_table.txt`](evidence/s7-speed/abba_table.txt) section 5). Before the needle, `free -h` showed 19 Gi available on the head and 22 Gi on the worker ([`free-before-needle.txt`](evidence/s7-speed/2B/free-before-needle.txt)).
- **Before a load.** `run.sh` waits until MemAvailable ≥ `MEM_GATE_GIB=100` on each node, up to `MEM_GATE_TIMEOUT` (600 s). It refuses a gate below 95 GiB at the defaults: the startup estimate plus the 12.1 GiB reserve, rounded up, where the estimate is resident 75.99 + max(5.33, 3.71 + 0.95 GiB a further lane per 1048576-token window + 0.01 GiB at 2 or more lanes). That gives 82.56 GiB at 4 x 1048576, the estimate s7 measured at `19f5478` ([`2B startup.txt`](evidence/s7-speed/2B/startup.txt)), and 81.32 GiB at one lane (94 at `--parallel 1`), s6's at `b86514a` ([`1A startup.txt`](evidence/s6-concurrent/1A/startup.txt)). The 0.01 is the batched proposals' scratch, which the engine allocates only at 2 or more lanes. It does not evict the page cache: the engine counts it as available, and a warm cache loads faster.
- **While serving.** `run.sh` adds `--oom-score-adj` (`OOM_SCORE_ADJ=1000`), `--ulimit core=1` and a memguard. The memguard kills the container when MemAvailable < `MEMGUARD_MIN_AVAIL_MB` (3072) MB and SwapFree < `MEMGUARD_MIN_SWAP_FREE_MB` (2048) MB for 6 s.
- The Engram reads go through the page cache. Rows are never resident (only their scale rows, since C4) and never cross the link.

Read memory with `free -h`, never `nvidia-smi`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1,roceP2p1s0f1
export PORT=8000
```

- **HCAs.** Pin `NCCL_IB_HCA`. Two Sparks on their direct cable expose two RoCE devices for the one port: list both ([TensorFold `RUNBOOK.md`](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/RUNBOOK.md) at the pin). `run.sh` checks that each listed device's port 1 is `ACTIVE`.
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
| [`s5-final`](evidence/README.md#s5-final-2026-10-06) | the engine bumped to `903a1e8` (speed units C1-C5, C7, C8) and the engine's default draft policy; image built by `run.sh`; two `run.sh` boots at the shipped defaults: the decode table, L.A.I.L, sampled cells, `bench_openai`, quiet `prefill_cold`, exactness, a 1,039,528-token needle, the full quality set, NLL; the vLLM sibling the same day on the same cells | decode table and headline from these boots; ahead of vLLM on prose, structured, L.A.I.L and prefill, behind on prose_long and short chat |
| [`s6-concurrent`](evidence/README.md#s6-concurrent-2026-10-08) | the engine bumped to `b86514a` (concurrent requests); six `run.sh` boots in one session: `--parallel 1` against 4 (ABBA), `--decode-share` 0.25 and 1, then the vLLM sibling; the decode table at c=1, 2 and 4, exactness under load, TTFT of prompts arriving mid-decode, a 1,039,528-token needle beside 3 lanes, memory; the gate at the shipped defaults | `PARALLEL=4` ships: the c=1 ruler within 0.5% of one lane (every c=1 cell within 1.31%), memory floors hold; the decode table from boots 2B and 3B |
| [`s7-speed`](evidence/README.md#s7-speed-2026-10-10) | the engine bumped to `19f5478` (v0.6.6 + the decode units P4, P2, P10, P1, P6b); the old recipe (`b86514a`) against the new in ABBA order through `run.sh`, then the vLLM sibling: the decode table at c=1, 2 and 4, the diverse cell, sampled cells, exactness against `b86514a` and s6, the needle beside 3 lanes, prefill, memory | `19f5478` ships: every reply equal to `b86514a`'s; diverse cell +2.50% to +3.54% at c=1, +11.01% to +14.81% at c=2, +16.40% to +23.28% at c=4; decode table from boots 2B and 3B |

## Gotchas

- **Name the revision.** The repository's `main` holds only the model card. `run.sh` downloads by revision sha.
- **Both nodes hold the whole pack.** Each rank reads its own Engram half from shards 47 and 48 on its own disk.
- **The first start compiles.** Seven CUDA extensions build at the first start on each node, into `TF_CACHE/<TF_SHA>` ([`extensions.txt`](evidence/s7-speed/extensions.txt)). s7's first boot at `19f5478` took 4 min 19 s from `./run.sh` to ready (rank 0 `loaded in 245.5s`), and MemAvailable fell to 12.13 GiB on the head while it compiled; warm boots at `19f5478` took 56 s ([`W1/run.txt`](evidence/s7-speed/W1/run.txt), [`2B/run.txt`](evidence/s7-speed/2B/run.txt)). Earlier first boots: 3 min 1 s at `ec28f35` (s2, [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)), 208 s at `903a1e8` (s5, [`derived.txt`](evidence/s5-final/derived.txt)). Rank 1 waits up to 600 s for rank 0's rendezvous ([`comm.py`](https://github.com/sfxnz/TensorFold/blob/19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb/src/tensorfold/cuda/comm.py#L54)).
- **The kernel cache is root-owned.** The containers run as root, so files under `TF_CACHE/<TF_SHA>` belong to root. Remove an old one through the image, not with sudo: `docker run --rm -v "$TF_CACHE:/c" "$IMAGE" rm -rf /c/<old TF_SHA>`.
- TensorFold opens HTTP only after the model is loaded: "connection refused" means still loading.

## Credits

- Engine: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) by its maintainer, with the `deepseek_v41` family of PRs #390 and #391, the device keyed draw, the speed units C1-C5, C7 and C8, concurrent requests and the decode units P4, P2, P10, P1 and P6b from the fork `sfxnz/TensorFold` (Upstream status, above).
- Weights: [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3), quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT per its model card).
- Harness: `bench_decode.py` and `smoke_chat.py` come from the vLLM sibling recipe, unchanged. The guards, `kit/render.py`, `stop.sh` and the gate follow the lab's Qwen3.8-Flash-Next TensorFold recipe.

## License

The recipe's own files are MIT ([`LICENSE`](LICENSE)). [`NOTICE`](NOTICE) lists the files that come from the lab's other recipes. TensorFold is not included; the image build fetches it at the pinned commit. Model weights follow their own licenses on Hugging Face.
