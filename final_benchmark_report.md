# Final Benchmark Report

This report merges the throughput, memory, TP/node scaling, sequence-length,
and backend sweeps for Giulio's `1.5b` benchmark track. Failed runs are excluded
from performance comparisons and called out separately where relevant.

## Best Configuration

The best measured configuration is:

| Setting | Value |
|---|---:|
| Model | 1.5b |
| Nodes | 4 |
| GPUs per node | 4 |
| Total GPUs | 16 |
| Tensor parallelism | 2 |
| Pipeline parallelism | 1 |
| Transformer implementation | transformer_engine |
| Attention backend | auto |
| Sequence length | 6144 |
| Micro batch size | 8 |
| Global batch size | 64 |
| Steps | 100 |

Run command:

```bash
NODES=4 GPUS_PER_NODE=4 TP=2 MBS=8 GBS=64 SEQ_LEN=6144 BACKENDS=auto bash benchmark_backend_best_tp.sh
```

Best observed throughput for this final config:

| Metric | Value |
|---|---:|
| Avg tokens/s/GPU | 21,355 |
| Aggregate tokens/s | 341,676 |
| Avg iter time | 1,166.7 ms |
| Avg TFLOP/s/GPU | 237.1 |
| Peak allocated memory | 81.4 GB |
| Peak reserved memory | 85.6 GB |

This is the best config because it combines the best completed TP/node setup
(`nodes=4`, `TP=2`), the best completed sequence length (`6144`), and the best
backend choice (`auto`).

## Main Findings

### 1. Micro Batch Size Controls Memory

The memory sweep showed that peak memory is driven by `MBS`, not `GBS`.

| MBS | Peak allocated GB | Peak reserved GB |
|---:|---:|---:|
| 4 | 22.7 | 23.5 |
| 8 | 40.3 | 41.8 |
| 16 | 75.7 | 78.4 |

Increasing `GBS` mainly increases gradient accumulation work. It does not keep
all microbatches resident at once, so it barely changes peak memory.

For the early one-node `TP=4`, `SEQ_LEN=4096` batch sweep, `MBS=8` was the best
practical choice. `MBS=16` did not improve throughput enough to justify the much
higher memory use.

### 2. Best TP/Node Scaling Is 4 Nodes with TP=2

The TP/node sweep used `MBS=8`, `GBS=64`, `SEQ_LEN=4096`, and `backend=auto`.

| Nodes | GPUs | TP | Aggregate tok/s | Peak alloc GB |
|---:|---:|---:|---:|---:|
| 1 | 4 | 2 | 83,768 | 58.0 |
| 1 | 4 | 4 | 43,070 | 40.3 |
| 2 | 8 | 2 | 84,185 | 56.2 |
| 2 | 8 | 4 | 84,051 | 38.6 |
| 4 | 16 | 2 | 161,474 | 55.4 |
| 4 | 16 | 4 | 159,122 | 37.8 |

Best throughput: `nodes=4`, `TP=2`.

`TP=4` is still useful when memory is tighter. At 4 nodes it is only about 1.5%
slower than `TP=2`, while saving about 17.6 GB allocated memory per GPU.

### 3. Longer Sequences Improve Throughput Until Memory Gets Tight

The sequence-length sweep used `nodes=4`, `TP=2`, `MBS=8`, `GBS=64`, and
`backend=auto`.

| Seq len | Aggregate tok/s | TFLOP/s/GPU | Peak alloc GB |
|---:|---:|---:|---:|
| 1024 | 166,352 | 90.9 | 16.9 |
| 2048 | 205,839 | 118.5 | 29.6 |
| 4096 | 247,612 | 157.2 | 55.4 |
| 6144 | 266,360 | 184.8 | 81.4 |

`SEQ_LEN=6144` was the best completed sequence-length run. It improves hardware
utilization substantially, but it is close to the memory limit at `MBS=8`.

`SEQ_LEN=4096` is the safer setting if memory headroom matters. It is slower,
but much less memory constrained.

### 4. Backend Auto Is the Best Choice

The final backend sweep used the best TP/node/sequence setup:
`nodes=4`, `TP=2`, `MBS=8`, `GBS=64`, `SEQ_LEN=6144`.

| Backend | Runs | Aggregate tok/s | TFLOP/s/GPU | Peak alloc GB | Diff vs auto |
|---|---:|---:|---:|---:|---:|
| auto | 2 | 341,676 | 237.1 | 81.4 | baseline |
| flash | 2 | 337,569 | 234.2 | 84.9 | -1.2% |
| fused | 1 | 275,687 | 191.3 | 81.4 | -19.3% |

`auto` is the best backend for the final config. `flash` is close, but slightly
slower and uses more memory. `fused` is clearly worse for this setup.

`unfused` OOMed at `SEQ_LEN=6144`, `MBS=8`. It tried to allocate another
5.62 GiB while PyTorch already had about 87.8 GiB allocated. It is not viable
for the final config.

`local` was not included in the final comparison because the GPT local-spec path
is currently not wired correctly: `--attention-backend local` requires
`--spec local`, but GPT then tries to import `local` as a two-part module spec.

## Final Recommendation

Use this for maximum measured throughput:

```bash
NODES=4 GPUS_PER_NODE=4 TP=2 MBS=8 GBS=64 SEQ_LEN=6144 BACKENDS=auto bash benchmark_backend_best_tp.sh
```

Use this if you want a safer memory-headroom configuration:

```bash
NODES=4 GPUS_PER_NODE=4 TP=2 MBS=8 GBS=64 SEQ_LEN=4096 BACKENDS=auto bash benchmark_backend_best_tp.sh
```

Use this if memory is the hard constraint and a small throughput loss is fine:

```bash
NODES=4 GPUS_PER_NODE=4 TP=4 MBS=8 GBS=64 SEQ_LEN=4096 BACKENDS=auto bash benchmark_backend_best_tp.sh
```

## What Not To Use

Avoid `MBS=16` unless the goal is specifically to test memory limits. It uses
far more memory and did not show a meaningful throughput win over `MBS=8` in the
batch sweep.

Avoid `ATTENTION_BACKEND=fused` for the final config. It was about 19% slower
than `auto`.

Avoid `ATTENTION_BACKEND=unfused` at `SEQ_LEN=6144`, `MBS=8`; it OOMs.

Avoid `ATTENTION_BACKEND=local` until the GPT local-spec wiring is fixed.

## Final Note

Keep `GBS=64` for the final configuration. It is the value used in the TP/node,
sequence-length, and backend sweeps, so the final recommendation is based on
directly measured runs rather than mixing settings from different experiments.
