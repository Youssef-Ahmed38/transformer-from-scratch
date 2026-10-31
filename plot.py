"""Draw the README charts from the measured results."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def loss_curve(log_path, meta, out):
    rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["step"] >= 500]  # the first points would flatten the rest
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.plot(steps, [r["train_loss"] for r in rows], color=BLUE, lw=2, label="Train")
    ax.plot(steps, [r["val_loss"] for r in rows], color=ORANGE, lw=2, label="Validation (unseen files)")
    ax.axhline(meta["bigram_val_loss"], color=MUTED, lw=1.2, ls="--", label="Bigram baseline")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (lower is better)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out)


def ablation(results, out):
    full = results["training"][-1]
    rows = [{"stage": "All optimizations on", **{k: full[k] for k in ("steps_per_sec", "min", "max")}}]
    rows += sorted(results["ablation"], key=lambda r: -r["steps_per_sec"])
    rows.append({"stage": "Baseline, all off", **{k: results["training"][0][k] for k in ("steps_per_sec", "min", "max")}})
    labels = [r["stage"].replace("without", "Without") for r in rows]
    values = [r["steps_per_sec"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    bars = ax.barh(labels, values, color=BLUE, height=0.55)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f", padding=4, color=INK)
    ax.set_xlabel("Training steps per second (median of 3 runs)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(right=max(values) * 1.12)
    fig.tight_layout()
    fig.savefig(out)


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    meta = json.loads(Path("data/meta.json").read_text(encoding="utf-8"))
    loss_curve("checkpoints/run/log.jsonl", meta, out / "loss_curve.png")
    ablation(json.loads((out / "benchmark.json").read_text(encoding="utf-8")), out / "ablation.png")


if __name__ == "__main__":
    main()
