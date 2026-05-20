#!/usr/bin/env python3
"""Generate throughput charts for the final Giulio benchmark report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2f6fbb"
GREEN = "#2f9e75"
ORANGE = "#d8892f"
GRAY = "#6b7280"


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(path)


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax, values, fmt="{:,.0f}") -> None:
    ymax = max(values) if values else 0
    for patch, value in zip(ax.patches, values):
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax.set_ylim(top=ymax * 1.14 if ymax else 1)


def plot_batch_grid() -> None:
    rows = [
        (4, 64, 10896),
        (4, 128, 11129),
        (4, 256, 10345),
        (8, 64, 10714),
        (8, 128, 11858),
        (8, 256, 11855),
        (16, 64, 11668),
        (16, 128, 11858),
        (16, 256, 11855),
    ]
    gbs_values = [64, 128, 256]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for mbs, color in [(4, BLUE), (8, GREEN), (16, ORANGE)]:
        values = [next(tok for m, g, tok in rows if m == mbs and g == gbs) for gbs in gbs_values]
        ax.plot(gbs_values, values, marker="o", linewidth=2.4, color=color, label=f"MBS={mbs}")
    style_axes(ax, "Batch Grid Throughput", "tokens/s/GPU")
    ax.set_xlabel("Global batch size")
    ax.set_xticks(gbs_values)
    ax.legend(frameon=False)
    savefig("batch_grid_throughput.png")


def plot_tp_nodes() -> None:
    rows = [
        ("1n TP2", 83768, 58.0),
        ("1n TP4", 43070, 40.3),
        ("2n TP2", 84185, 56.2),
        ("2n TP4", 84051, 38.6),
        ("4n TP2", 161474, 55.4),
        ("4n TP4", 159122, 37.8),
    ]
    labels = [r[0] for r in rows]
    throughput = [r[1] for r in rows]
    colors = [BLUE if "TP2" in label else GREEN for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, throughput, color=colors, width=0.66)
    style_axes(ax, "TP and Node Scaling Throughput", "Aggregate tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    annotate_bars(ax, throughput)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="TP=2"),
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="TP=4"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    savefig("tp_node_scaling.png")


def plot_seq_len() -> None:
    seq = np.array([1024, 2048, 4096, 6144])
    throughput = np.array([166352, 205839, 247612, 266360])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(seq, throughput, color=BLUE, marker="o", linewidth=2.5)
    style_axes(ax, "Sequence Length Throughput", "Aggregate tokens/s")
    ax.set_xlabel("Sequence length")
    ax.set_xticks(seq)
    ax.ticklabel_format(axis="y", style="plain")
    savefig("seq_len_sweep.png")


def plot_backend() -> None:
    labels = ["auto", "flash", "fused"]
    throughput = [341676, 337569, 275687]
    colors = [BLUE, GREEN, ORANGE]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(labels))
    ax.bar(x, throughput, color=colors, width=0.62)
    style_axes(ax, "Backend Throughput at Best Config", "Aggregate tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    annotate_bars(ax, throughput)
    savefig("backend_sweep.png")


def plot_final_summary() -> None:
    labels = ["4096 seq\n(auto)", "6144 seq\n(auto)", "6144 seq\n(flash)", "6144 seq\n(fused)"]
    throughput = [247612, 341676, 337569, 275687]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, throughput, color=[GRAY, BLUE, GREEN, ORANGE], width=0.62)
    style_axes(ax, "Final Throughput Comparison", "Aggregate tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    annotate_bars(ax, throughput)
    savefig("final_tradeoff.png")


def main() -> None:
    plot_batch_grid()
    plot_tp_nodes()
    plot_seq_len()
    plot_backend()
    plot_final_summary()


if __name__ == "__main__":
    main()
