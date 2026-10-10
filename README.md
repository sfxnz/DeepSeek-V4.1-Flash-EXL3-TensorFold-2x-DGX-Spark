# DeepSeek-V4.1-Flash EXL3 · TensorFold · 2× DGX Spark

Serve [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/tree/2.0bpw-mcg-viterbi-lmhead-mxfp8) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2 with [TensorFold](https://github.com/ashhart/TensorFold)'s `deepseek_v41` CUDA family. Not vLLM. The vLLM recipe for the same weights is [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark); one repo per engine.

**Status: measured through `run.sh` on this pair, and against vLLM in the same session.** Session [`s6-concurrent`](evidence/README.md#s6-concurrent-2026-10-08) bumped the engine to `b86514a` (concurrent requests) and booted `./run.sh` six counted times in one session, after two warm-up boots: `--parallel 1` (boots 1A, 4A) against the shipped `--parallel 4` (2B, 3B) in ABBA order, then `--decode-share` 0.25 and 1 (5C, 6D), then the vLLM sibling (7V) on the same cells: the decode table at c=1, 2 and 4, exactness under load, prompts arriving while other requests decode, a 1,039,528-token needle beside three decoding requests, memory. Session [`s5-final`](evidence/README.md#s5-final-2026-10-06) bumped the engine to `903a1e8` (the speed units below), switched `run.sh` to the engine's default draft policy, built the image with `IMAGE_ONLY=1 ./run.sh`, and booted `./run.sh` twice at the shipped defaults (boots E and F) for the frozen ruler, L.A.I.L, sampled cells, `bench_openai`, a quiet `prefill_cold` and exactness; boot F also ran a 1,039,528-token needle and the full quality set, and the teacher-forced NLL ran after it. The same day it stopped TensorFold and booted the vLLM sibling recipe with its canonical command for the same cells. s5 is the source of the c=1 L.A.I.L, `bench_openai`, quality and NLL rows below. Earlier sessions: [`s2-run-sh-ec28f35`](evidence/README.md#s2-run-sh-ec28f35-2026-10-04) (first `run.sh` boots, needles to 1,039,528 tokens, quality), [`s3-default-1m`](evidence/README.md#s3-default-1m-2026-10-04) (`--top-p 1.0`, two boots at 1048576), [`s4-device-nucleus`](evidence/README.md#s4-device-nucleus-2026-10-05) (the device top_p cut). The window default is the native 1048576. The snapshot is the pinned `snapshots/982b704…` folder on both nodes; `run.sh`'s check passed on both at every boot.

- **Engine.** TensorFold `19f5478` from [sfxnz/TensorFold `dsv41-recipe-engine5`](https://github.com/sfxnz/TensorFold/tree/dsv41-recipe-engine5): release v0.6.4 (`6ea5ade`); the `deepseek_v41` family of upstream PRs [#390](https://github.com/ashhart/TensorFold/pull/390) and [#391](https://github.com/ashhart/TensorFold/pull/391) (`e174ee0` has the tree of #391's tip `0d91389`); the device keyed draw for top-k-off rows (`0582d5f`, `3ab53da`; `db7f53e` has the tree of s4's pin `41306d5`); then seven speed units, each merged after its own receipts: C1 skips the decoder rows a prompt's result does not read, C3 a prompt-window EXL3 expert kernel, C2 decode fusion, C7 packed caches and a split decode index selection, C4 Engram I/O off the decode path with resident scale rows, C5 drafts drawn on the device and a confidence draft policy, C8 prompt chunks in two row halves with the kept snapshot taken inside its chunk; then concurrent requests (units N0-N10): up to 4 requests share one verify forward a round, each in its own lane with a whole window, prompts fill between rounds, and every reply equals its solo run (`b86514a`); then upstream's final Python line, `python-0.6` at `ed78d6f` (v0.6.6: API keys, `--name-priority`, the tool-JSON rewrite), merged as `dc971a2`; then five decode units, each kept on its own ABBA: P4 one grouped `wo_a` launch, P10 greedy target draws on the device, P2 a paced L2 warm of the next dense weights, P1 shape-picked MXFP8 decode tiles, P6b DSpark proposals batched across lanes. Each unit is bit-exact, so replies equal `b86514a`'s, and each is the engine's default. s2 and s3 ran `ec28f35`, s4 `41306d5`, s5 `903a1e8`, s6 `b86514a`; the numbers below are s6's, at `b86514a`, until a session measures `19f5478`. Apache-2.0. The image builds from [`docker/Dockerfile`](docker/Dockerfile) on `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned).
- **Weights.** Revision `982b70452f399814f56b46272fd30394ae10d58c` (branch `2.0bpw-mcg-viterbi-lmhead-mxfp8`), 48 shards. The routed experts are EXL3 at 2 bits (`mcg` codebook). Everything else is DeepSeek's own bytes: FP8 with 32x32 block scales, an MXFP8 LM head, MXFP4 DSpark draft experts, and the two Engram tables in shards 47 and 48.
- **Engram.** Each rank reads its half of every Engram row from the pack on its own local disk. No repacking, no second copy. Both nodes need the whole revision.
- **Ranks.** Rank 1 runs on `spark2`, rank 0 serves HTTP on `spark1`. NCCL over the QSFP RoCE link.

## Upstream status

TensorFold's Python engine ends at 0.6.6, kept on its [`python-0.6`](https://github.com/ashhart/TensorFold/tree/python-0.6) branch ([ashhart/TensorFold#286](https://github.com/ashhart/TensorFold/issues/286)); new work goes to its Zig engine. For that reason the maintainer closed [#390](https://github.com/ashhart/TensorFold/pull/390), [#391](https://github.com/ashhart/TensorFold/pull/391) and [#408](https://github.com/ashhart/TensorFold/pull/408), and will rewrite the useful parts in Zig, with credit. Upstream `main` is now the Zig engine 1.0.x. It has no DeepSeek-V4 family: its [CHANGELOG](https://github.com/ashhart/TensorFold/blob/main/CHANGELOG.md) (1.0.0) lists DeepSeek-V4 among the families that "remain ports for 1.0.x". On CUDA the native binary registers only Nemotron, on one GPU (GB10 or an Ampere 8.6 card; [docs/recipes/cuda.md](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/cuda.md)), so it has no CUDA tensor parallel either. So this recipe pins the fork: `sfxnz/TensorFold` branch `dsv41-recipe-engine5` at `19f5478`, which `docker/Dockerfile` installs from git at the full commit. It carries upstream's final Python line (`python-0.6` at `ed78d6f`). A later bump goes to another fork commit, or to an upstream release once one serves this family, and needs new evidence.

## Measured on 2× DGX Spark (L.A.I.L lab)

`python3 bench_decode.py`, byte-identical to the vLLM sibling's frozen ruler (sha256 `3172cbc4…`). It sends `ignore_eos` and `chat_template_kwargs {"thinking": false, "reasoning_effort": "low"}`; the server log shows `thinking=False` for every gate request (322 of 322 and 321 of 321 `done` lines in boots 2B and 3B, [`2B`](evidence/s6-concurrent/2B/gate/docker-head.log), [`3B`](evidence/s6-concurrent/3B/gate/docker-head.log)). Do not copy community tok/s into this table.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: frozen bench_decode.py (sha256 3172cbc4…), streamed, thinking off, ignore_eos, max_tokens 200, 9 runs, two boots at the shipped defaults (s6 boots 2B and 3B, 2026-10-08: TensorFold `b86514a`, `--context 1048576 --parallel 4 --decode-share 0.5`, `--top-p 1.0`, `--mtp-drafts 5 --mtp-confidence 0.15`; the only env override, EXTRA_ENV, loaded a probe that logs memory around graph warm-up); each row is the median of the two per-boot medians (evidence/s6-concurrent/decode.txt, which gives each boot's); TP=2 through `./run.sh`, `tools/session_gate.sh` with RUNS=9 RUNS_LONG=9. Concurrency N sends N streams at once, 9 waves: decode is the median stream's rate after its first token, aggregate the median wave's tokens over its wall time, TTFT p50 over every stream (runs 2 to 9 resume the kept prompt). prose, structured and prose_long are the ruler's greedy cells. The sampled rows send the ruler's prose prompt with no sampling field (the server's temperature 1.0, top_p 1.0, top-k off) or top_p 0.95, no seed, so every stream draws the same reply (evidence/s6-concurrent/scripts/sampled_conc.py). Before (s5 bootE and bootF, TensorFold `903a1e8`, one request at a time): prose 55.8, structured 97.3, prose_long 40.7 (evidence/s5-final/decode.txt).

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose (note 1) | 1 | 56.3 | 56.3 | 0.07 s |
| prose | 2 | 41.3 | 82.2 | 0.13 s |
| prose | 4 | 29.1 | 115.7 | 0.28 s |
| structured | 1 | 97.7 | 97.7 | 0.07 s |
| structured | 2 | 72.5 | 144.0 | 0.13 s |
| structured | 4 | 50.6 | 200.7 | 0.26 s |
| prose_long | 1 | 41.2 | 41.2 | 0.07 s |
| prose_long | 2 | 30.6 | 61.0 | 0.13 s |
| prose_long | 4 | 21.8 | 87.1 | 0.27 s |
| sampled (no field) | 1 | 43.0 | 43.0 | 0.07 s |
| sampled (no field) | 2 | 31.6 | 63.3 | 0.13 s |
| sampled (no field) | 4 | 22.4 | 89.8 | 0.27 s |
| sampled (top_p 0.95) | 1 | 37.9 | 37.9 | 0.07 s |
| sampled (top_p 0.95) | 2 | 28.6 | 57.2 | 0.13 s |
| sampled (top_p 0.95) | 4 | 20.5 | 82.4 | 0.29 s |

1. The prose reply ends at its end token after 74 tokens; ignore_eos decodes the other 126 (post_eos_fraction 0.63).
<!-- END generated measured -->

`run.sh` serves `--parallel 4`: up to 4 requests decode together, each in its own lane holding a whole 1,048,576-token window, and each reply equals the same request served alone (Concurrent requests, below). A fifth request waits for a free lane.

- Per boot (2B / 3B), c=1: prose 56.24 / 56.34, structured 98.47 / 96.90, prose_long 41.24 / 41.24 tok/s; inter-chunk p50 51-60 ms ([`decode.txt`](evidence/s6-concurrent/decode.txt)). The ruler is greedy, so `--top-p` does not touch it.
- c=1 against `--parallel 1` in the same session (boots 1A and 4A: 56.51 / 98.00 / 41.42): -0.39%, -0.33%, -0.43%; cold prefill at 2k to 64k within -1.31% to +0.66%. The gate was 3% ([`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 1). A lone request on 4 lanes runs at the one-lane speed.
- Against s5 (`903a1e8`, one request at a time, two boots: 55.8 / 97.3 / 40.7, [`s5 decode.txt`](evidence/s5-final/decode.txt)): +0.9%, +0.4%, +1.3%.
- Earlier tables: s3 at `ec28f35` 46.0 / 64.1 / 34.6 ([`s3 decode.txt`](evidence/s3-default-1m/decode.txt)); s2 at `--context 65538` 45.7 / 64.2 / 34.4 ([`s2 decode.txt`](evidence/s2-run-sh-ec28f35/decode.txt)); the engine's receipt run at `0d91389` 45.5 / 63.5 / 34.0 ([`s0`](evidence/s0-engine-receipts/s65538/bench_decode.log)). s5 against s3: prose x1.21, structured x1.52, prose_long x1.18, the speed units and the draft policy together ([`s5 derived.txt`](evidence/s5-final/derived.txt); Draft policy, below).

### Concurrent requests (s6, 2026-10-08)

Lanes: `PARALLEL=4` lanes, each holding a whole `CONTEXT` window (4 x 1,048,576 tokens), allocated at startup. A request takes a free lane and keeps it to its end; while all 4 are busy, further requests wait in the queue. Each round runs one shared verify forward over every decoding lane. A new prompt fills between those rounds, and the rounds take `DECODE_SHARE` (0.5) of each prompt span's time. Rows: the decode table above (boots 2B and 3B, the median of the per-boot medians).

**Against vLLM, same session.** The vLLM sibling booted after the TensorFold boots with its canonical command (`env AUDIT=strict ./run.sh` at `3de9146`, image `dsv41-flash-exl3-sm121:canonical-e14`, `audit ok` on both ranks, `MAX_NUM_SEQS=2`), one boot, the same clients ([`7V/`](evidence/s6-concurrent/7V/), [`derived.txt`](evidence/s6-concurrent/derived.txt)). vLLM decodes 2 sequences at a time, so in its c=4 waves 2 streams wait for the first 2.

| Cell | c | TF aggregate | vLLM aggregate | TF / vLLM | TF per stream | vLLM per stream |
|---|---:|---:|---:|---:|---:|---:|
| prose (greedy, 200 tokens) | 1 | 56.28 | 49.28 | 1.142 | 56.29 | 49.29 |
| prose | 2 | 82.19 | 81.29 | 1.011 | 41.34 | 41.14 |
| prose | 4 | 115.65 | 102.10 | 1.133 | 29.11 | 39.91 |
| structured | 1 | 97.66 | 83.97 | 1.163 | 97.68 | 84.00 |
| structured | 2 | 144.02 | 155.85 | 0.924 | 72.51 | 77.95 |
| structured | 4 | 200.65 | 201.80 | 0.994 | 50.60 | 78.23 |
| prose_long | 1 | 41.24 | 46.23 | 0.892 | 41.24 | 46.24 |
| prose_long | 2 | 60.96 | 69.42 | 0.878 | 30.60 | 35.28 |
| prose_long | 4 | 87.09 | 87.61 | 0.994 | 21.80 | 35.00 |
| sampled, no sampling field | 2 | 63.31 | 65.42 | 0.968 | 31.65 | 34.66 |
| sampled, no sampling field | 4 | 89.84 | 82.27 | 1.092 | 22.40 | 34.44 |
| sampled, top_p 0.95 | 2 | 57.16 | 68.17 | 0.838 | 28.62 | 36.30 |
| sampled, top_p 0.95 | 4 | 82.41 | 82.95 | 0.993 | 20.54 | 34.15 |

- Aggregate over c=1: x1.46-1.48 at c=2 and x2.05-2.11 at c=4 on the ruler's cells.
- TensorFold leads at c=4 prose (+13%) and no-field sampling (+9%), and ties vLLM within 1% at c=2 prose and at c=4 structured, prose_long and top_p 0.95. It trails at c=2: structured -7.6%, prose_long -12.2%, top_p 0.95 -16.2%, no-field sampling -3.2%.
- Per stream, a TensorFold request at c=4 decodes at 52-54% of its c=1 rate: the 4 lanes share each round. vLLM's per-stream rate at c=4 counts only the 2 streams decoding at a time; the waiting 2 show up in its TTFT.
- vLLM's c=2 prose and structured match its round-36 run (81.02 / 156.29). Its c=1 in s6: 49.29 / 84.00 / 46.24 (s5: 47.88 / 82.57 / 43.32).
- vLLM draws a new reply each sampled run; TensorFold seeds an unseeded request from its prompt, so every sampled stream draws one reply. The sampled cells time different text.
- At `--parallel 1` (boots 1A, 4A) the same waves queue: prose c=2 74.55, c=4 88.95 tok/s aggregate ([`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 2).

**TTFT and inter-token time under load** ([`derived.txt`](evidence/s6-concurrent/derived.txt)):

- Short prompts sent together: TTFT p50 0.13 s at c=2 and 0.26-0.29 s at c=4 (0.07 s at c=1), each prompt filling between the others' rounds. vLLM: 0.23-0.31 s at c=2, 1.6-3.1 s at c=4 (queued).
- Inter-chunk time, p50 / p90 over the ruler's cells: 51-60 / 59-63 ms at c=1, 69-80 / 79-86 ms at c=2, 97-114 / 113-123 ms at c=4. vLLM: 47-48 / 50-54 ms at c=1, 51-61 / 53-71 ms at c=2 and c=4. A TensorFold chunk carries a round's accepted tokens, so a longer gap still carries more tokens.
- A 16,386-token prompt arriving while 3 lanes decode ([`ttft_cells.jsonl`](evidence/s6-concurrent/2B/ttft_cells.jsonl) in 2B, 3B, 5C, 6D; [`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 4). Alone it reaches its first token in 9.58 s. Over each whole reply, fill included, the lanes decode at about 17.7 tok/s each.

| `--decode-share` | Fill TTFT (median) | Each live lane during the fill | Largest gap between tokens |
|---|---:|---:|---:|
| 0.25 (boot 5C) | 14.98 s | about 4.2 tok/s | 0.79 s |
| 0.5, shipped (2B / 3B) | 17.98 / 18.66 s | about 7.2 tok/s | 0.73-0.77 s |
| 1 (6D) | 24.04 s | about 11.3 tok/s | 0.75 s |

- A request sent with `"priority": "background"` (or a session-title request) yields its lane when all 4 are busy and a prompt waits. It replays its whole reply later and sends only the tokens it had not sent: its largest gap was 18.7-27.7 s, and its reply equals its solo run.
- A 1,039,528-token needle beside 3 decoding lanes (boot 2B, [`needle_lanes.jsonl`](evidence/s6-concurrent/2B/needle_lanes.jsonl)): found, the same token hash as s5's needle alone, answered in 1370.6 s against 756.6 s alone. The 24 live replies overlapping it decoded at 7.9-10.6 tok/s, with a largest gap of 2.13 s, and all drew the same hash.

**Exactness under load.** Boots 1A, 2B, 3B and 4A: `tools/bench_concurrent.py`, plain and `--mixed --stagger-ms 2000`, 102 of 102 replies equal to the same request alone ([`conc-check.out`](evidence/s6-concurrent/2B/gate/conc-check.out) per boot). `tools/pairs_concurrent.py` (4 staggered prompts of 18 to 25,061 tokens, greedy and keyed, no top_k or top_p field): boots 2B, 3B, 5C and 6D equal both `--parallel 1` boots on 4 of 4 token hashes. The 68 streams of the TTFT cells equal their solo runs. The sampled cells drew the same two hashes at c=1, 2 and 4 as s5 at c=1. In both arms the 12 nucleus pairs equal s5 bootF's ([`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 3).

**Memory.** `startup estimate 82.55 GiB` at 4 x 1,048,576 against 81.32 GiB at one lane, on both ranks ([`2B startup.txt`](evidence/s6-concurrent/2B/startup.txt), [`1A startup.txt`](evidence/s6-concurrent/1A/startup.txt)). MemAvailable, every 2 s on both nodes ([`memwatch.tsv`](evidence/s6-concurrent/2B/memwatch.tsv) per boot): at least 17.42 GiB on the head (during the needle) and 21.90 GiB on the worker over the warm-cache 4-lane boots (2B, 3B, 5C, 6D); 13.01 / 14.74 GiB during the first start on an empty kernel cache, while the kernels compiled (warm-up boot W1, [`W1/memwatch.tsv`](evidence/s6-concurrent/W1/memwatch.tsv)); 23.1 / 25.2 GiB at one lane. Four lanes cost about 3.7 GiB on the head and 3.0 GiB on the worker (MemAvailable after warm-up, `startup.txt`, boots 2B and 3B against 1A and 4A). Capturing the 88 lane graphs took 2.8-4.6 s at startup.

**Limits.**

- At most 4 requests decode at once (the family's limit). More wait for a free lane.
- A client that disconnects stops its request after the next round. One that disconnects while its prompt fills is noticed only at its first token: the fill runs to its end and holds its lane until then.
- `thinking_budget` stays refused (HTTP 400) at every `--parallel`, 2 to 4 included.
- Under `--parallel` 2 or more, a reply's `prefill_s` (its `tensorfold` stats and rank 0's `done … prefill=` field) counts only the time spent launching prompt spans: the needle above read 48.4 s. Time prompts from the client, or read `ttft=`.
- A routed-expert fp16 overflow in one lane, while the first forwards still check for it, fails that round for every live request (the family page). A failure on one rank in the middle of a round is not recovered: restart both with `./stop.sh && ./run.sh`.

### Against vLLM at c=1, s5 (2026-10-06, `903a1e8`)

Same pack, same pair, same clients, one day. TensorFold: boots E and F at the shipped defaults, each cell the median of the two per-boot medians. vLLM: the sibling recipe booted after them with its canonical command (`env AUDIT=strict ./run.sh` in its clean checkout at `3de9146`, image `dsv41-flash-exl3-sm121:canonical-e14` on both nodes, `audit ok` on both ranks, DSpark k=3), one boot, the same commands, then stopped gracefully ([`vllm/`](evidence/s5-final/vllm/), driver [`vllm.sh`](evidence/s5-final/scripts/vllm.sh)). Ratios in [`derived.txt`](evidence/s5-final/derived.txt).

| Cell (c=1, decode tok/s) | TensorFold | vLLM | TF / vLLM |
|---|---:|---:|---:|
| bench_decode prose (greedy, 200 tokens, 9 runs) | 55.81 | 47.88 | 1.165 |
| bench_decode structured | 97.25 | 82.57 | 1.178 |
| bench_decode prose_long | 40.68 | 43.32 | 0.939 |
| L.A.I.L prose (`measure_lail_prose.py`: 512 tokens, t=0.2, 10 runs) | 41.23 | 40.19 | 1.026 |
| sampled, no sampling field (200 tokens, 9 runs) | 42.77 | 42.06 | 1.017 |
| sampled, temperature 1.0, top_p 0.95 | 37.70 | 42.19 | 0.894 |
| sampled, temperature 1.0, top_k 20 | 42.55 | 43.45 | 0.979 |
| bench_openai fibonacci-raw, t=1 (64 tokens, 5 reps), see note | 45.76 | 47.05 | 0.973 |
| bench_openai gpu-chat-no-think, t=1 | 41.47 | 43.12 | 0.962 |
| gpu-chat-no-think, t=0 (`bench_t0_chat.py`, 64 tokens, 5 reps) | 41.22 | 47.06 | 0.876 |

- Files: ruler [`bootE/gate`](evidence/s5-final/bootE/gate/), [`bootF/gate`](evidence/s5-final/bootF/gate/), [`vllm/bench-*.out`](evidence/s5-final/vllm/); L.A.I.L `lail.log`, sampled `sampled-*.txt`, `bench_openai.json`, `bench_t0_chat.json` in [`bootE/`](evidence/s5-final/bootE/), [`bootF/`](evidence/s5-final/bootF/) and [`vllm/`](evidence/s5-final/vllm/).
- TTFT p50 of the ruler cells: TensorFold 0.068-0.071 s (runs 2-9 resume the kept prompt), vLLM 0.214-0.271 s. L.A.I.L TTFT 0.075 s against 0.325 s.
- A request with no sampling field gets each server's own defaults (TensorFold: temperature 1.0, top_p 1.0, top-k off). TensorFold seeds an unseeded request from its prompt, so its 9 runs draw one reply; vLLM draws a new reply each run. TensorFold's top_p 0.95 reply on this prompt takes more rounds than its top_p 1.0 reply (99 against 86 in s4), which is most of that cell's gap.
- `bench_openai`'s t=0 pass fails on vLLM (greedy emits only end tokens on the untemplated fibonacci-raw prompt), so vLLM ran it at t=1 only, plus [`bench_t0_chat.py`](evidence/s5-final/scripts/bench_t0_chat.py) (the oracle session's t=0 chat twin) on both. TensorFold's own t=0 pass: fibonacci-raw 64.25, gpu-chat-no-think 41.16.
- The fibonacci-raw t=1 cells time different text. The recorded sample (first rep, first 160 characters) of TensorFold's reply is end tokens only in boots E and F and in s2: with `ignore_eos` it keeps drawing the end token. vLLM's reply has one end token and then continues with text. Read that row as a rough figure, not a like-for-like comparison.
- Prose ends at its end token after 74 tokens on TensorFold and 84 on vLLM; both cells decode 200 with ignore_eos.

**Prefill.** The engine's `tools/prefill_cold.py` at `903a1e8`, quiet (only the lab app's `/v1/models` pollers held connections, [`prefill-clients.txt`](evidence/s5-final/bootE/prefill-clients.txt)), median of 3 prompts a length, prompt tok/s:

| Prompt tokens | TensorFold (E / F) | TF TTFT | vLLM | vLLM TTFT | TF / vLLM | s3 at `ec28f35` |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1146.5 (1139.7 / 1153.2) | 1.79 s | 792.9 | 2.58 s | 1.446 | 526.0 |
| 8,192 | 1542.5 (1573.6 / 1511.4) | 5.31 s | 744.8 | 11.00 s | 2.071 | 551.7 |
| 16,384 | 1684.8 (1704.3 / 1665.4) | 9.73 s | 739.9 | 22.14 s | 2.277 | 561.3 |
| 32,768 | 1709.0 (1655.6 / 1762.4) | 19.19 s | 777.1 | 42.16 s | 2.199 | 563.5 |
| 65,536 | 1790.1 (1800.2 / 1779.9) | 36.61 s | 760.7 | 86.15 s | 2.353 | 570.6 |

Both servers got the same prompts: s2's `prompts.json` for TensorFold and the oracle session's `G7_prefill_prompts.json` for vLLM hold the same 16 items (identical messages and token counts). The files differ only in an extra top-level `corpus_python` key in the G7 copy, so their sha256 values in `prefill_cold.log` differ. s3's column is s3 boot D ([`prefill_cold.json`](evidence/s3-default-1m/bootD/prefill_cold.json)). Against s3 the prompt rate is x2.2 at 2k to x3.1 at 64k; C1, C3 and C8 are the units on the prompt path.

**Where TensorFold trails.**

- prose_long c=1: 40.68 against 43.32 tok/s (-6.1%). Its inter-chunk p50 is 52 ms against vLLM's 49 ms. In s6: 41.24 against 46.24 (-10.8%).
- Short sampled and short greedy chat replies: top_k 20 -2.1%, `bench_openai` t=1 -2.7% and -3.8%, the 64-token greedy chat reply -12.4%, top_p 0.95 -10.6% (a reply with more rounds, above).
- Two requests at once (s6): structured -7.6%, prose_long -12.2%, top_p 0.95 -16.2% aggregate against vLLM's 2 sequences. At c=4 each TensorFold stream decodes slower than vLLM's 2 running streams; vLLM queues the other 2 (Concurrent requests, above).
- No images, structured outputs, `logprobs` or named tool choice (Not supported, below). vLLM serves images on the same pack.

### Draft policy

`run.sh` passes `--mtp-drafts 5 --mtp-confidence 0.15`: the engine's default at `903a1e8`, written out so a bump cannot change it silently. Each round DSpark drafts up to 5 tokens, and the chain stops where the drafts' confidence product falls under 0.15. Until s5 `run.sh` passed `--mtp-drafts 3`: 3 drafts every round, the default before C5. Decision `DRAFTS` ([`decision.tsv`](evidence/decision.tsv)), from the C5 unit receipts ([`c5-draft-policy/abba_table.txt`](evidence/s5-final/c5-draft-policy/abba_table.txt); dev launcher, `--context 65538`, not `run.sh`). Within the two C5 boots, each case run with d=3 and with the default, forward then reversed, decode tok/s (mean over requests):

| Case | d=3 | default | ratio |
|---|---:|---:|---:|
| prose, greedy 200 | 50.2 | 55.1 | 1.098 |
| structured (count to 200), greedy | 70.4 | 98.2 | 1.394 |
| L.A.I.L prose, t=0.2, 512 | 35.6 | 39.0 | 1.095 |
| code, greedy 400 | 49.2 | 54.6 | 1.110 |
| Swedish / German / Chinese prose, t=0.2, 512 | 27.0 / 36.3 / 42.6 | 30.9 / 40.5 / 46.1 | 1.142 / 1.116 / 1.081 |
| Swedish / German / Chinese, greedy 300 | 28.4 / 41.8 / 40.2 | 32.1 / 46.4 / 43.1 | 1.130 / 1.110 / 1.072 |

Every case drew the same reply under both policies (14 of 14 cells). Across the four C5 boots (A B B A, boot medians B/A) the one cell that lost beyond noise is `bench_openai` gpu-chat-no-think at t=0: 0.926 (prefill 2k and 8k sit at 0.999 and 0.995, within noise and off the draft path), a fixed 64-token greedy chat reply. At `903a1e8` that cell decodes at 41.16 tok/s, where s2 measured 41.2 at `ec28f35` with 3 drafts ([`s2 bench_openai.log`](evidence/s2-run-sh-ec28f35/bootA/bench_openai.log)). The policy changes speed, not tokens: every sampled reply and every pair in s5 has the token hash s4 or s2 drew with 3 drafts (below).

### Sampled decode (top-k off)

The ruler is greedy. [`sampled_cell.py`](evidence/s5-final/scripts/sampled_cell.py) (s4's) sends the ruler's prose prompt with sampling: streamed, 200 tokens, ignore_eos, no seed (the server seeds from the prompt, so each run draws the same reply), 9 runs a boot. Median decode tok/s at the shipped defaults (`TOP_P=1.0`):

| Request sends | s5, `903a1e8` (E / F) | s4, `41306d5` | Reply equal to s4 |
|---|---:|---:|:---:|
| no sampling field (server default: temperature 1.0, top_p 1.0, top-k off) | 42.65 / 42.88 | 36.59 | yes |
| temperature 1.0, top_p 0.95 (top-k off) | 37.54 / 37.87 | 31.76 | yes |
| temperature 1.0, top_k 20 | 42.53 / 42.56 | 37.04 | yes |

([`bootE/sampled-*.txt`](evidence/s5-final/bootE/), [`bootF/sampled-*.txt`](evidence/s5-final/bootF/), [`s4 sampled-*.txt`](evidence/s4-device-nucleus/), [`derived.txt`](evidence/s5-final/derived.txt)). Each reply has the same token hash in both s5 boots and in s4. top_p under 1 draws on the device since `41306d5`: at `ec28f35` a row whose nucleus ran past each rank's 1,024 best candidates read the whole vocabulary half to the host (15.5 tok/s on this prompt, s3). The engine still takes the host path past the cut's `TMAX` bound and wherever the device draw cannot decide (its slack check); the logs do not count those rows ([`s4 derived.txt`](evidence/s4-device-nucleus/derived.txt)).

### Exactness

At `903a1e8` (s5, each boot): temperature 1.0 with top_k omitted, seed 1234, 160 tokens, on three prompts, at top_p 0.95 and 1.0: drafted equals `"draft": false` in 6 of 6, and each drafted reply's token hash equals s2's reply to the same request at `ec28f35` (3 drafts a round) in 6 of 6 ([`bootE/pairs.jsonl`](evidence/s5-final/bootE/pairs.jsonl), [`bootF/pairs.jsonl`](evidence/s5-final/bootF/pairs.jsonl), [`pairs_nucleus.py`](evidence/s5-final/scripts/pairs_nucleus.py)). The engine's `tools/bench_concurrent.py --alone --serial` (4 requests queued at once, code and chat prompts, t=1 and 0, seeded): each boot 48 of 48 replies equal the same request sent alone and 10 of 10 equal `"draft": false` ([`bench_concurrent.json`](evidence/s5-final/bootE/bench_concurrent.json), [`derived.txt`](evidence/s5-final/derived.txt)).

Earlier: s2 checked drafted against `"draft": false` in 13 of 13 cases at `ec28f35`, with every reply the s0 receipts also sent equal to `0d91389`'s ([`pairs.jsonl`](evidence/s2-run-sh-ec28f35/bootA/pairs.jsonl), [`pairs_vs_s0.txt`](evidence/s2-run-sh-ec28f35/bootA/pairs_vs_s0.txt)); s3 the default request at top_p 1.0 ([`pairs.jsonl`](evidence/s3-default-1m/bootD/pairs.jsonl)); s4 the device top_p cut against the host path ([`pairs.jsonl`](evidence/s4-device-nucleus/pairs.jsonl)).

### Window and long prompts

`CONTEXT=1048576` boots through `run.sh` at `903a1e8`: `startup estimate 81.32 GiB within 99.54 GiB` on rank 0 and `within 102.12 GiB` on rank 1 (boot E, [`startup.txt`](evidence/s5-final/bootE/gate/startup.txt)). On boot F, after the other cells, one needle from the vLLM sibling's `quality_eval.py` builder, greedy ([`long_needle.py`](evidence/s5-final/scripts/long_needle.py), s2's, the same length, depth and seed as s2's longest): 1,039,528 prompt tokens, prefilled in 754.3 s (12.6 min, 1378 tok/s), found ([`needle.jsonl`](evidence/s5-final/bootF/needle.jsonl)). MemAvailable, sampled every 5 s on both nodes ([`memwatch.tsv`](evidence/s5-final/bootF/memwatch.tsv)), was at least 18.1 GiB on the head and 25.2 GiB on the worker during that prompt, and at least 15.0 / 23.7 GiB over the whole boot (the head's low came during `prefill_cold`). The same prompt took 2077.8 s at `ec28f35` (s2): x2.75.

s2's needles at `ec28f35`, one at a time ([`needles.jsonl`](evidence/s2-run-sh-ec28f35/ctx1m/needles.jsonl), [`memwatch.tsv`](evidence/s2-run-sh-ec28f35/ctx1m/memwatch.tsv), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)):

| Prompt tokens | Depth | Prefill s | Prompt tok/s | Found | Min MemAvailable head / worker |
|---:|---:|---:|---:|---|---:|
| 130,171 | 0.1 | 214.3 | 607 | yes | 24.8 / 26.7 GiB |
| 130,541 | 0.5 | 215.1 | 607 | yes | 24.6 / 27.4 GiB |
| 129,076 | 0.9 | 212.8 | 607 | yes | 24.8 / 27.4 GiB |
| 257,955 | 0.5 | 439.8 | 587 | yes | 24.6 / 27.0 GiB |
| 523,681 | 0.5 | 946.9 | 553 | yes | 23.4 / 26.7 GiB |
| 1,039,528 | 0.5 | 2077.8 | 500 | yes | 22.1 / 26.7 GiB |

At `--parallel 1` the server serves nothing else while it prefills. At `--parallel 4` the other lanes keep decoding between the prompt's spans (the needle beside 3 lanes, Concurrent requests, above). Memory stays flat after admission: every lane's window is allocated at startup. Tested windows: 65538 (two boots, s2) and 1048576 (s2's boot with the needles, s3's two, s4's one, s5's two, s6's six). The default is 1048576.

### Quality

On boot F after the needle ([`quality_full.txt`](evidence/s5-final/bootF/quality_full.txt), [`quality_full.json`](evidence/s5-final/bootF/quality_full.json)): the vLLM sibling's `quality_eval.py --full --only selfcons,gsm8k,gsm8k_think,mmlu,tools,needle` through the lab's `qe_tf.py`, gated against vLLM's round-36 full baseline on this pack: pass. selfcons 12 of 12 identical, golden hazard 0.01307 (s2: 0.01307); GSM8K 97/100 and with thinking 39/40 (baseline 97/100, 39/40); MMLU 204/228 (baseline 205/228, inside the gate); tools exact_args 22/22, json_valid 22/22, no_call 8/8; needles 9/9 at 8k, 32k and 128k. The `nll`, `decode`, `c2` and `vision` components need `prompt_logprobs`, two streams or images, which this engine does not serve.

Teacher-forced NLL on the 40 G1 passages (19,828 positions, both ranks, the recipe image at `903a1e8`, [`nll_pair.sh`](evidence/s5-final/scripts/nll_pair.sh)): 0.137069, the repeat equal, and all 80 score files bit-identical to s2's at `ec28f35`, which were bit-identical to `0d91389`'s ([`nll_compare.txt`](evidence/s5-final/nll/nll_compare.txt)). Nothing in the units changed the prompt path's arithmetic on these passages: C3's kernel, C4's resident scale rows and C7's packed caches are in the code it runs. It does not reach C8's halves (a chunk runs in halves from 1,280 rows; the passages are 511-513 tokens) or C1's skip (the scorer reads every row's logits), and C2 and C5 change decode only. The scorer is the lab's `dev/final/nll_tf.py`; the lab's `dev/nll_tf.py` is the same file before the score module moved to `tests/cuda/` (`7032f26`), and does not import at `903a1e8`.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (`10.100.8.1` / `10.100.8.2` in this lab), both RoCE ports `ACTIVE`.
- Docker + NVIDIA Container Toolkit on both nodes. SSH from the head to the worker (`spark2` here).
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up; `run.sh` refuses to.
- Disk on each node: the whole revision, 333 GiB ([`pins.json`](evidence/s1-snapshot-pins/pins.json), [family page](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/docs/recipes/deepseek-v4.1-flash.md#download-on-both-machines)), in the HF cache (`HF_CACHE`, default `~/.cache/huggingface`) on local NVMe, never NFS. Other revisions already in the same cache entry add to that. Plus the kernel cache under `TF_CACHE`.

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

`run.sh` builds `tf-dsv41-flash:0.6.6-19f5478` from [`docker/Dockerfile`](docker/Dockerfile) on the head when it is missing. It copies the image to the worker over the link when the worker's image ID differs (`docker save | ssh docker load`). Do both before the downtime, while another serve still holds the pair:

```bash
IMAGE_ONLY=1 ./run.sh   # build on the head, copy to the worker; no GPU, no weights, no memory gate
```

- The build runs `pip install "tensorfold @ git+TF_REPO@TF_SHA"` under [`docker/constraints.txt`](docker/constraints.txt). It then checks the commit pip recorded (`direct_url.json`) against `TF_SHA` and `TF_REPO`.
- The constraints pin TensorFold's runtime deps at the versions expected in the base image. The build fails if pip changes any installed package: a wrong pin fails the build, not a serve, and torch and triton stay the base's. s2's build passed that check: pip changed nothing ([`image-only.txt`](evidence/s2-run-sh-ec28f35/image-only.txt)). `IMAGE_ONLY=1 ./run.sh` took 3 min 34 s, most of it the copy to the worker ([`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). s4's build at `41306d5` took 208 s ([`image-only.txt`](evidence/s4-device-nucleus/image-only.txt)); s5's at `903a1e8` passed the same checks and took 214 s, the same image ID on both nodes ([`image-only.txt`](evidence/s5-final/image-only.txt), [`image-ids.txt`](evidence/s5-final/image-ids.txt), [`derived.txt`](evidence/s5-final/derived.txt)).
- No extras: the family serves no grammars and no images.
- Label `tensorfold.sha`. `run.sh` refuses an image whose label differs from `TF_SHA`.
- The four CUDA extensions compile at the first start, into `TF_CACHE/<TF_SHA>` on each node.
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

The serve flags match the receipt run's (`--tp 2 --master … --name deepseek-ai/DeepSeek-V4.1-Flash --no-thinking --no-update-check` on both ranks, and `--context`, 65538 there, [`environment.txt`](evidence/s0-engine-receipts/environment.txt)). `run.sh` also passes `--parallel 4 --decode-share 0.5` and `--mtp-drafts 5 --mtp-confidence 0.15` to both ranks, and rank 0 also gets `--max-tokens 4096`. The drafting values and the reply cap are the engine's and the CLI's defaults, written out so a bump cannot change them silently ([`serve-argv.txt`](evidence/s6-concurrent/2B/serve-argv.txt); the receipts ran 3 drafts a round, the default before C5, see Draft policy). Both ranks also get `--top-p 1.0`, which the receipts did not pass (their server default was 0.95; see Sampled decode). Rank 0 prints it: `serving … (sampling: top_p 1.0; …)` ([`startup.txt`](evidence/s6-concurrent/2B/gate/startup.txt)). A request that omits `temperature` gets the engine's 1.0 ([`cuda/server.py`](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/src/tensorfold/cuda/server.py#L92) at the pin; the family sets top-k off, [`app.py`](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/src/tensorfold/families/deepseek_v41/cuda/app.py#L36)). The container env adds `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, as the receipt run had: it keeps fp32 matmuls in fp32 whatever TF32 default the NGC image sets.

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

From the family's [recipe page](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/docs/recipes/deepseek-v4.1-flash.md) at the pinned commit:

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

- **Admission.** At startup each rank sizes its caches for the window and allocates everything once; serving allocates no device memory. The engine grants MemAvailable less a reserve of max(4 GiB, a tenth of MemTotal) ([`capacity.py`](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/src/tensorfold/cuda/capacity.py#L163-L168)): 12.1 GiB of 121 GiB on these Sparks ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). The page cache counts as available. An explicit `--context` that does not fit is refused on both ranks.
- **Receipts.** At `--context 65538`: `startup estimate 78.46 GiB within 99.76 GiB` on rank 0 and `within 101.46 GiB` on rank 1, with 73.23 GiB of weights ([`serve_s65538_rank0.log`](evidence/s0-engine-receipts/serve_s65538_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_s65538_rank1.log)). MemAvailable was 115 / 116 GiB before that boot ([`free_before_s65538.txt`](evidence/s0-engine-receipts/free_before_s65538.txt)) and 28 / 30 GiB after the receipt session ([`free_after.txt`](evidence/s0-engine-receipts/s65538/free_after.txt)).
- **Native window.** Without `--context`, both ranks admitted 1,048,576 tokens: `startup estimate 79.09 GiB within 98.95 GiB` / `101.55 GiB` ([`serve_default_rank0.log`](evidence/s0-engine-receipts/serve_default_rank0.log), [`rank1`](evidence/s0-engine-receipts/serve_default_rank1.log)). One smoke ran; 23 / 26 GiB were available after it ([`free_after_default.txt`](evidence/s0-engine-receipts/free_after_default.txt)). Through `run.sh` (s2): the same admission, then the bench and needles up to 1,039,528 tokens; the lowest MemAvailable was 22.1 GiB on the head and 26.7 GiB on the worker ([`memwatch.tsv`](evidence/s2-run-sh-ec28f35/ctx1m/memwatch.tsv)). At `903a1e8` (s5): `startup estimate 81.32 GiB within 99.54 GiB` / `102.12 GiB`, with 76.07 GiB of weights (C4 keeps the Engram scale rows resident); over boot F, the bench cells, the 1,039,528-token needle and the full quality set, the lowest MemAvailable was 15.0 GiB on the head and 23.7 GiB on the worker ([`startup.txt`](evidence/s5-final/bootE/gate/startup.txt), [`memwatch.tsv`](evidence/s5-final/bootF/memwatch.tsv), [`derived.txt`](evidence/s5-final/derived.txt)). At `b86514a` with `--parallel 4` (s6): `startup estimate 82.55 GiB` for 4 windows of 1,048,576, 81.32 GiB at one lane; the lowest MemAvailable over the 4-lane boots was 17.42 GiB on the head, during the needle beside 3 lanes, and 21.90 GiB on the worker ([`abba_table.txt`](evidence/s6-concurrent/abba_table.txt) section 5).
- **Before a load.** `run.sh` waits until MemAvailable ≥ `MEM_GATE_GIB=100` on each node, up to `MEM_GATE_TIMEOUT` (600 s). It refuses a gate below 95 GiB at the defaults: the startup estimate plus the 12.1 GiB reserve, rounded up, where the estimate is resident 75.99 + max(5.33, 3.71 + 0.95 GiB a further lane per 1048576-token window). That gives 82.55 GiB at 4 x 1048576 and 81.32 GiB at one lane (94 at `--parallel 1`), the estimates s6 measured ([`2B startup.txt`](evidence/s6-concurrent/2B/startup.txt), [`1A startup.txt`](evidence/s6-concurrent/1A/startup.txt)). It does not evict the page cache: the engine counts it as available, and a warm cache loads faster.
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

- **HCAs.** Pin `NCCL_IB_HCA`. Two Sparks on their direct cable expose two RoCE devices for the one port: list both ([TensorFold `RUNBOOK.md`](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/RUNBOOK.md) at the pin). `run.sh` checks that each listed device's port 1 is `ACTIVE`.
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

## Gotchas

- **Name the revision.** The repository's `main` holds only the model card. `run.sh` downloads by revision sha.
- **Both nodes hold the whole pack.** Each rank reads its own Engram half from shards 47 and 48 on its own disk.
- **The first start compiles.** Four CUDA extensions build at the first start on each node, into `TF_CACHE/<TF_SHA>`. s2's first boot with an empty cache took 3 min 1 s from `./run.sh` to ready (rank 0 `loaded in 167.9s`); warm boots took about a minute ([`bootA/run.txt`](evidence/s2-run-sh-ec28f35/bootA/run.txt), [`bootB/run.txt`](evidence/s2-run-sh-ec28f35/bootB/run.txt), [`derived.txt`](evidence/s2-run-sh-ec28f35/derived.txt)). s4's first boot at `41306d5` took 182 s ([`s4 run.txt`](evidence/s4-device-nucleus/run.txt)); s5's first at `903a1e8` 208 s (rank 0 `loaded in 194.7s`), its second, warm, 66 s ([`bootE/run.txt`](evidence/s5-final/bootE/run.txt), [`bootF/run.txt`](evidence/s5-final/bootF/run.txt), [`derived.txt`](evidence/s5-final/derived.txt)). Rank 1 waits up to 600 s for rank 0's rendezvous ([`comm.py`](https://github.com/sfxnz/TensorFold/blob/b86514a5ac8700b32e7b24f1095349df4ce2b922/src/tensorfold/cuda/comm.py#L54)).
- **The kernel cache is root-owned.** The containers run as root, so files under `TF_CACHE/<TF_SHA>` belong to root. Remove an old one through the image, not with sudo: `docker run --rm -v "$TF_CACHE:/c" "$IMAGE" rm -rf /c/<old TF_SHA>`.
- TensorFold opens HTTP only after the model is loaded: "connection refused" means still loading.

## Credits

- Engine: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) by its maintainer, with the `deepseek_v41` family of PRs #390 and #391, the device keyed draw, the speed units C1-C5, C7 and C8 and concurrent requests from the fork `sfxnz/TensorFold` (Upstream status, above).
- Weights: [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3), quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT per its model card).
- Harness: `bench_decode.py` and `smoke_chat.py` come from the vLLM sibling recipe, unchanged. The guards, `kit/render.py`, `stop.sh` and the gate follow the lab's Qwen3.8-Flash-Next TensorFold recipe.

## License

The recipe's own files are MIT ([`LICENSE`](LICENSE)). [`NOTICE`](NOTICE) lists the files that come from the lab's other recipes. TensorFold is not included; the image build fetches it at the pinned commit. Model weights follow their own licenses on Hugging Face.
