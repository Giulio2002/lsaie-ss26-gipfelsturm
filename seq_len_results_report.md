# Sequence Length Throughput Results

## Setup

These results come from `seq_len_logs.tar.gz` and cover the `125m` model with:

- `backend=auto`
- `MBS=8`
- `GBS=64`
- `steps=100`
- `nodes=1`
- `gpus_per_node=4`
- `TP=1`, `PP=1`

All runs reached 100 iterations. Throughput numbers below are averaged over
post-warmup iterations, using iterations 11-100.

## Results

| Seq len | Last iter | Avg tok/s/GPU | Median tok/s/GPU | Aggregate avg tok/s | Avg iter ms | Avg TFLOP/s/GPU | Last loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 100 | 29,877 | 30,554 | 119,506 | 591.9 | 22.1 | 5.6552 |
| 2048 | 100 | 39,259 | 39,252 | 157,035 | 903.5 | 31.3 | 5.5095 |
| 4096 | 100 | 48,203 | 47,936 | 192,813 | 1,422.0 | 43.9 | 5.4685 |
| 8192 | 100 | 52,716 | 50,967 | 210,863 | 2,551.6 | 60.0 | 5.4381 |
| 12288 | 100 | 51,082 | 50,982 | 204,329 | 3,916.4 | 69.7 | 5.2900 |

## Findings

Token throughput improves strongly as sequence length increases from 1024 to
8192:

```text
1024 -> 8192: 29,877 to 52,716 tokens/s/GPU
```

That is a 76% improvement in tokens/s/GPU. The likely reason is that longer
sequences give the attention and matrix multiplication kernels more work per
launch and improve GPU utilization. Short sequence lengths are dominated more by
fixed overheads.

The best token throughput is:

```text
seq_len=8192
52,716 tokens/s/GPU
210,863 aggregate tokens/s
```

At `seq_len=12288`, token throughput drops slightly to 51,082 tokens/s/GPU, even
though TFLOP/s/GPU continues to increase. This distinction matters:

- `seq_len=8192` is best for the project metric, tokens/s/GPU.
- `seq_len=12288` is best for hardware utilization, TFLOP/s/GPU.

The `12288` run does more attention work per token, so the model achieves higher
TFLOP/s while processing slightly fewer tokens per second.

Average iteration time grows with sequence length:

```text
1024:    592 ms
2048:    903 ms
4096:  1,422 ms
8192:  2,552 ms
12288: 3,916 ms
```

This is expected because each step processes `GBS * seq_len` tokens. With
`GBS=64`, the tokens per step are:

| Seq len | Tokens per step |
|---:|---:|
| 1024 | 65,536 |
| 2048 | 131,072 |
| 4096 | 262,144 |
| 8192 | 524,288 |
| 12288 | 786,432 |

The larger sequence lengths process far more tokens per step, and the per-token
throughput improves up to 8192.

## Recommendation

For the Challenge 2 metric, use:

```bash
SEQ_LEN=8192
MBS=8
GBS=64
```

This is the best observed sequence length for tokens/s/GPU in this sweep.

If the goal is to demonstrate kernel efficiency rather than maximize raw
tokens/s/GPU, also report `seq_len=12288`, because it reaches the highest
TFLOP/s/GPU:

```text
69.7 TFLOP/s/GPU
```

For the next experiment, test around the peak:

```bash
SEQ_LENS="6144 8192 10240 12288"
MBS=8
GBS=64
```

This would clarify whether the true token-throughput optimum is exactly 8192 or
somewhere between 8192 and 12288.
