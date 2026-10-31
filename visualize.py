"""Architecture diagram, result charts and attention views, drawn from real outputs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot import BLUE, GRID, INK, MUTED, ORANGE, SURFACE
from tfs.attention import attention_maps
from tfs.model import Config, Transformer
from tfs.tokenizer import BPETokenizer

AQUA = "#1baf7a"
OUT = Path("results")
SNIPPET = """def add_item(items, value):
    if value is None:
        return items
    items.append(value)
    return items
"""
TEAL_FILL, TEAL_EDGE = "#e1f5ee", "#0f6e56"
PURPLE_FILL, PURPLE_EDGE = "#eeedfe", "#534ab7"
CORAL_FILL, CORAL_EDGE = "#faece7", "#993c1d"
GRAY_FILL, GRAY_EDGE = "#f1efe8", "#5f5e5a"
WEIGHTS = LinearSegmentedColormap.from_list("weight", [SURFACE, "#b5d4f4", BLUE, "#0c447c"])


def box(ax, x, y, w, h, title, subtitle, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec=edge, lw=0.9))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=10.5, color=edge, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8.5, color=edge)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10.5, color=edge, weight="bold")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=11, color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=170)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    box(ax, 3, 9.15, 4, 0.7, "Input token IDs", "Up to 256 tokens", GRAY_FILL, GRAY_EDGE)
    arrow(ax, (5, 9.15), (5, 8.75))
    box(ax, 2.6, 8.05, 4.8, 0.7, "Token + position embeddings", "Each token becomes 384 numbers", TEAL_FILL, TEAL_EDGE)
    arrow(ax, (5, 8.05), (5, 7.6))
    ax.add_patch(FancyBboxPatch((0.5, 3.75), 9, 3.85, boxstyle="round,pad=0,rounding_size=0.14", fc="none", ec=MUTED, lw=0.8, ls="--"))
    ax.text(5, 7.28, "Transformer block, repeated 6 times", ha="center", fontsize=10.5, color=INK, weight="bold")
    rows = [
        (5.62, ("Layer norm", "Written by hand"), ("Masked attention", "6 heads, size 64"), PURPLE_FILL, PURPLE_EDGE),
        (4.05, ("Layer norm", "Written by hand"), ("Feed-forward", "384 to 1,536 to 384"), CORAL_FILL, CORAL_EDGE),
    ]
    for y, first, middle, fill, edge in rows:
        box(ax, 0.8, y, 2.6, 1.05, *first, fill, edge)
        box(ax, 3.7, y, 2.6, 1.05, *middle, fill, edge)
        box(ax, 6.6, y, 2.6, 1.05, "Residual add", "Dropout, then add input", fill, edge)
        arrow(ax, (3.4, y + 0.52), (3.7, y + 0.52))
        arrow(ax, (6.3, y + 0.52), (6.6, y + 0.52))
    ax.plot([7.9, 7.9, 2.1], [5.62, 5.36, 5.36], color=MUTED, lw=0.9)
    arrow(ax, (2.1, 5.36), (2.1, 5.1))
    arrow(ax, (5, 3.75), (5, 3.3))
    box(ax, 3, 2.75, 4, 0.55, "Final layer norm", "", GRAY_FILL, GRAY_EDGE)
    arrow(ax, (5, 2.75), (5, 2.35))
    box(ax, 2.6, 1.65, 4.8, 0.7, "Output projection", "Reuses token embedding weights", TEAL_FILL, TEAL_EDGE)
    arrow(ax, (5, 1.65), (5, 1.25))
    box(ax, 3, 0.7, 4, 0.55, "4,096 next-token scores", "", GRAY_FILL, GRAY_EDGE)
    ax.text(5, 0.28, "12.31M parameters", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "architecture.png", facecolor=SURFACE)
    plt.close(fig)


def tradeoff():
    runs = json.loads((OUT / "early_exit.json").read_text(encoding="utf-8"))["runs"]
    series = [("learned/optimistic", "Optimistic bound", GRID, "o"), ("learned/frozen", "Frozen", ORANGE, "s"), ("learned/gated", "Gated (real)", BLUE, "o")]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=160)
    ax.axhline(1, color=MUTED, lw=1.2, ls="--", label="1% quality budget")
    for key, label, color, marker in series:
        color = "#8a8983" if color == GRID else color
        points = sorted(runs[key]["curve"], key=lambda p: p["saved_pct"])
        ax.plot([p["saved_pct"] for p in points], [p["loss_increase_pct"] for p in points], color=color, lw=2, marker=marker, ms=5, label=label)
    ax.set_xlabel("Compute saved (%)")
    ax.set_ylabel("Loss increase (%, lower is better)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "early_exit_tradeoff.png")
    plt.close(fig)


def token_types():
    first = json.loads((OUT / "early_exit.json").read_text(encoding="utf-8"))["baseline"]["by_token_type"]
    second = json.loads((OUT / "early_exit_swapped.json").read_text(encoding="utf-8"))["baseline"]["by_token_type"]
    names = ["identifier", "number", "keyword", "whitespace", "punctuation"]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.7), dpi=160)
    a = ax.barh(y - 0.19, [first[n]["loss"] for n in names], 0.34, color=BLUE, label="Second half of validation data")
    b = ax.barh(y + 0.19, [second[n]["loss"] for n in names], 0.34, color=ORANGE, label="First half of validation data")
    ax.bar_label(a, fmt="%.2f", padding=4, color=INK, fontsize=9)
    ax.bar_label(b, fmt="%.2f", padding=4, color=INK, fontsize=9)
    ax.set_yticks(y, [n.capitalize() for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-entropy loss by token type (lower is easier)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(right=4.7)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "token_types.png")
    plt.close(fig)


def step_time():
    share = json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))["breakdown_baseline"]["share_pct"]
    names = sorted(share, key=share.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 2.7), dpi=160)
    bars = ax.barh([n.capitalize() for n in names], [share[n] for n in names], color=BLUE, height=0.55)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK)
    ax.set_xlabel("Share of one training step, before optimization")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 66)
    fig.tight_layout()
    fig.savefig(OUT / "step_time.png")
    plt.close(fig)


def show(token):
    return token.replace("\n", "↵").replace("\t", "→").replace(" ", "·")


def attention_demo():
    state = torch.load("checkpoints/run/best.pt", map_location="cpu", weights_only=False)
    model = Transformer(Config(**state["config"]))
    model.load_state_dict(state["model"])
    tokenizer = BPETokenizer.load("data/tokenizer.json")
    ids = tokenizer.encode(SNIPPET)
    tokens = [tokenizer.decode([i]) for i in ids]
    maps = attention_maps(model, torch.tensor([ids]))
    layers = maps.mean(1).numpy()

    demo = {
        "snippet": SNIPPET,
        "tokens": tokens,
        "layers": [np.round(layer, 3).tolist() for layer in layers],
        "note": "Attention weights from the trained checkpoint, averaged over 6 heads.",
    }
    (OUT / "attention_demo.json").write_text(json.dumps(demo), encoding="utf-8")

    labels = [show(t) for t in tokens]
    fig, ax = plt.subplots(figsize=(8.6, 8), dpi=160)
    image = ax.imshow(layers[-1], cmap=WEIGHTS, vmin=0, vmax=np.quantile(layers[-1], 0.995))
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7.5, fontfamily="monospace")
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5, fontfamily="monospace")
    ax.set_xlabel("Token being looked at")
    ax.set_ylabel("Token doing the looking")
    ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.75, label="Attention weight, layer 6")
    fig.tight_layout()
    fig.savefig(OUT / "attention_heatmap.png")
    plt.close(fig)

    page = Path("docs/attention_explorer.template.html").read_text(encoding="utf-8")
    Path("docs/attention_explorer.html").write_text(page.replace("__DATA__", json.dumps(demo)), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    architecture()
    tradeoff()
    token_types()
    step_time()
    attention_demo()


if __name__ == "__main__":
    main()
