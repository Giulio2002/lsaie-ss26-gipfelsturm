# GBS x MBS Throughput Results

## Setup

These results come from `logs.tar.gz` and cover the `125m` model with:

- `backend=auto`
- `seq_len=4096`
- `steps=100`
- `nodes=1`
- `gpus_per_node=4`
- `TP=1`, `PP=1`, so data-parallel size is 4

The comparison is over micro-batch size (`MBS`) and global batch size (`GBS`).
Throughput numbers below are averaged over post-warmup iterations, using
iterations 11 onward where available.

## Important Caveats

`GBS=196` is invalid for every tested `MBS` value. Megatron requires:

```text
GBS % (MBS * data_parallel_size) == 0
```

With 4 data-parallel ranks:

```text
GBS % (MBS * 4) == 0
```

So:

- `MBS=4` requires GBS multiple of 16
- `MBS=8` requires GBS multiple of 32
- `MBS=16` requires GBS multiple of 64

`196` is not divisible by 16, 32, or 64, so it is omitted from the results.
Use `192` instead if you want a nearby valid value.

Also, only the `GBS=64` runs reached iteration 100 in the archived logs. The
`GBS=128` and `GBS=256` runs are partial, so their loss values are not directly
comparable to the completed `GBS=64` runs.

## Results

| MBS | GBS | Last Iter | Avg tok/s/GPU | Median tok/s/GPU | Aggregate avg tok/s | Avg iter ms | Avg TFLOP/s/GPU | Last loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 100 | 58,714 | 57,704 | 234,854 | 1,170.6 | 53.5 | 5.4028 |
| 4 | 128 | 86 | 58,867 | 58,481 | 235,469 | 2,261.7 | 53.6 | 5.4267 |
| 4 | 256 | 44 | 58,644 | 58,085 | 234,578 | 4,492.8 | 53.4 | 6.1153 |
| 8 | 64 | 100 | 59,055 | 57,716 | 236,218 | 1,166.6 | 53.8 | 5.4764 |
| 8 | 128 | 71 | 51,335 | 49,677 | 205,339 | 2,593.2 | 46.8 | 5.7549 |
| 8 | 256 | 43 | 58,645 | 57,676 | 234,581 | 4,491.7 | 53.4 | 6.3568 |
| 16 | 64 | 100 | 58,828 | 58,738 | 235,313 | 1,170.5 | 53.6 | 5.4269 |
| 16 | 128 | 70 | 51,432 | 49,781 | 205,728 | 2,589.7 | 46.9 | 5.7508 |
| 16 | 256 | 43 | 58,676 | 58,019 | 234,706 | 4,489.9 | 53.5 | 6.3329 |

## Findings

The best completed run is:

```text
MBS=8, GBS=64
```

It reaches about:

```text
59,055 tokens/s/GPU
236,218 aggregate tokens/s
53.8 TFLOP/s/GPU
```

However, the three completed `GBS=64` runs are very close:

| MBS | GBS | Avg tok/s/GPU |
|---:|---:|---:|
| 4 | 64 | 58,714 |
| 8 | 64 | 59,055 |
| 16 | 64 | 58,828 |

The spread is only about 0.6%, so there is no strong evidence that `MBS` is a
major throughput lever in this range for the 125m model. `MBS=8` is the best
observed value, but `MBS=4`, `8`, and `16` are effectively tied for throughput.

Increasing `GBS` mostly increases iteration time because each optimizer step
contains more tokens. For valid runs, aggregate throughput stays near
235k tokens/s in most cases, especially for `GBS=64` and `GBS=256`. That means
`GBS` is not improving raw token throughput here; it mainly changes how many
tokens are processed per optimizer step.

The suspicious cases are:

```text
MBS=8,  GBS=128
MBS=16, GBS=128
```

These report about 51k tokens/s/GPU, noticeably lower than the rest. Since both
logs are partial and stop around iteration 70, rerunning them would be useful
before treating that slowdown as real.

## Recommendation

For this setup, use:

```bash
MBS=8
GBS=64
```

as the current best configuration for `125m`, `seq_len=4096`, 4 GPUs.

For the next sweep, avoid invalid batch sizes and use a clean grid:

```bash
MBS_VALUES="4 8 16 24 32"
GBS_VALUES="64 128 192 256"
```

or, if you want divisibility across all of those MBS values, use multiples of
384:

```bash
GBS_VALUES="384 768"
```

For a practical next run, I would do:

```bash
MBS_VALUES="4 8 16"
GBS_VALUES="64 128 192 256"
```

and increase the SLURM time limit if you need all `GBS=128/192/256` jobs to
reach 100 iterations.
