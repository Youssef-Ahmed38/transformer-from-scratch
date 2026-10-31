"""Go/no-go test: can predictable tokens leave the network early at equal quality?

The rules are in docs/PREREGISTRATION.md. Heads and thresholds are fit on one half of the
validation data, and every reported number comes from the other half. The primary result
is the learned gate in "gated" mode. Other signals and modes are secondary analyses.
"""
import argparse
import json
import keyword
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tfs.model import Config, LayerNorm, Transformer
from tfs.provenance import provenance
from tfs.tokenizer import BPETokenizer

MAX_LOSS_INCREASE = 0.01
NEVER = float("inf")
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
SIGNALS = ("learned", "confidence", "convergence")
MODES = {
    "optimistic": "exited tokens cost nothing, but deeper layers still see them (upper bound)",
    "gated": "exited tokens disappear from deeper context",
    "frozen": "exited tokens stop updating, but stay visible as keys and values",
}
TOKEN_CLASSES = ("whitespace", "punctuation", "keyword", "identifier", "number", "fragment")


class ExitHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.gate = nn.Linear(dim, 1)

    def logits(self, x, embedding):
        h = self.norm(x)
        return (h + self.proj(h)) @ embedding.T


def token_class(text):
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if keyword.iskeyword(stripped):
        return "keyword"
    if stripped.isidentifier():
        return "identifier"
    if stripped.isdigit():
        return "number"
    if not any(ch.isalnum() for ch in stripped):
        return "punctuation"
    return "fragment"


def class_table(tokenizer, device):
    ids = [
        TOKEN_CLASSES.index(token_class(tokenizer.vocab[i].decode("utf-8", errors="replace")))
        for i in range(tokenizer.eot)
    ]
    return torch.tensor([*ids, TOKEN_CLASSES.index("whitespace")], device=device)


def exit_score(signal, logits, gate, h, prev):
    if signal == "learned":
        return gate.sigmoid()
    if signal == "confidence":
        return logits.float().softmax(-1).amax(-1)
    change = (h - prev).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
    return 1 / (1 + change)


def check_cost(signal, d, vocab_size):
    """Multiply-accumulates to run one exit check on one token."""
    return {"learned": d, "convergence": 2 * d, "confidence": d * d + d * vocab_size}[signal]


def total_macs(cfg, queries, pairs, frozen, checks, exited, tokens, signal):
    """Everything the run computed, per token. Attention here is the q-k dot product and k-v sum."""
    d = cfg.d_model
    blocks = sum(queries) * 12 * d * d  # qkv 3, proj 1, feed-forward 8
    attention = sum(pairs) * 2 * d
    kv_only = sum(frozen) * 2 * d * d  # frozen tokens still produce keys and values
    gates = checks * check_cost(signal, d, cfg.vocab_size)
    if signal == "confidence":
        output = (tokens - exited) * d * cfg.vocab_size  # exits reuse the check's projection
    else:
        output = tokens * d * cfg.vocab_size + exited * d * d
    return (blocks + attention + kv_only + gates + output) / tokens


def windows(tokens, block_size, batch_size, device):
    n = (len(tokens) - 1) // block_size
    data = torch.from_numpy(tokens[: n * block_size + 1].astype(np.int64)).to(device)
    x = data[:-1].reshape(n, block_size)
    y = data[1:].reshape(n, block_size)
    return list(zip(x.split(batch_size), y.split(batch_size)))


def embed(model, x):
    return model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=x.device))


def layer_states(model, x):
    h = embed(model, x)
    states = [h]
    for block in model.blocks:
        h = block(h)[0]
        states.append(h)
    return states


def fit_heads(model, heads, batches, steps, positions=48, lr=1e-3):
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=0.01)
    embedding = model.tok_emb.weight.detach()
    for step in range(steps):
        x, y = batches[step % len(batches)]
        pick = torch.randperm(x.size(1), device=x.device)[:positions]  # full logits would not fit in memory
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            states = layer_states(model, x)
            final = (model.norm(states[-1][:, pick]) @ embedding.T).argmax(-1)
        loss = 0.0
        target = y[:, pick]
        for head, state in zip(heads, states[1:-1]):
            state = state[:, pick].detach()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head.logits(state, embedding)
                gate = head.gate(state).squeeze(-1)
            agree = (logits.argmax(-1) == final).float()
            loss = loss + F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))
            loss = loss + F.binary_cross_entropy_with_logits(gate.float(), agree)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return heads


@torch.no_grad()
def thresholds_for(model, heads, batches, signal, max_scores=200_000):
    """Quantiles of each signal, so every signal gets an equally fine grid."""
    embedding = model.tok_emb.weight
    collected = []
    for x, _ in batches[:4]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = layer_states(model, x)
            for head, prev, h in zip(heads, states[:-2], states[1:-1]):
                logits = head.logits(h, embedding)
                collected.append(exit_score(signal, logits, head.gate(h).squeeze(-1), h, prev).float().cpu().view(-1))
    scores = torch.cat(collected)
    scores = scores[torch.randperm(len(scores))[:max_scores]]
    return [*torch.quantile(scores, torch.tensor(QUANTILES)).tolist(), NEVER]


@torch.no_grad()
def evaluate(model, heads, batches, threshold, mode, signal, classes=None):
    cfg = model.cfg
    embedding = model.tok_emb.weight
    n_layer = cfg.n_layer
    total_loss = correct = tokens = checks = exited = 0
    queries, pairs, frozen = [0] * n_layer, [0] * n_layer, [0] * n_layer
    exit_share = [0] * n_layer
    class_count = torch.zeros(len(TOKEN_CLASSES), device="cuda")
    class_depth = torch.zeros(len(TOKEN_CLASSES), device="cuda")
    class_loss = torch.zeros(len(TOKEN_CLASSES), device="cuda")

    for x, y in batches:
        position = torch.arange(1, x.size(1) + 1, device=x.device)
        per_token_loss = torch.zeros_like(y, dtype=torch.float)
        depth = torch.full_like(y, n_layer)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = embed(model, x)
            active = torch.ones_like(x, dtype=torch.bool)

            def settle(logits, leaving):
                logits = logits[leaving].float()
                per_token_loss[leaving] = F.cross_entropy(logits, y[leaving], reduction="none")
                return (logits.argmax(-1) == y[leaving]).sum().item()

            for layer, block in enumerate(model.blocks):
                queries[layer] += active.sum().item()
                if mode == "frozen":
                    pairs[layer] += (active * position).sum().item()
                    frozen[layer] += (~active).sum().item()
                else:
                    pairs[layer] += (active.cumsum(1) * active).sum().item()

                previous = h
                updated = block(h, key_mask=active if mode == "gated" else None)[0]
                h = torch.where(active[..., None], updated, h) if mode == "frozen" else updated
                if layer == n_layer - 1:
                    break

                head = heads[layer]
                logits = head.logits(h, embedding)
                score = exit_score(signal, logits, head.gate(h).squeeze(-1), h, previous)
                checks += active.sum().item()
                leaving = active & (score >= threshold)
                correct += settle(logits, leaving)
                depth[leaving] = layer + 1
                exit_share[layer] += leaving.sum().item()
                exited += leaving.sum().item()
                active = active & ~leaving

            correct += settle(model.norm(h) @ embedding.T, active)
        total_loss += per_token_loss.sum().item()
        tokens += y.numel()
        if classes is not None:
            token_classes = classes[y].view(-1)
            class_count.index_add_(0, token_classes, torch.ones_like(token_classes, dtype=torch.float))
            class_depth.index_add_(0, token_classes, depth.view(-1).float())
            class_loss.index_add_(0, token_classes, per_token_loss.view(-1))

    row = {
        "loss": total_loss / tokens,
        "accuracy": correct / tokens,
        "macs_per_token": total_macs(cfg, queries, pairs, frozen, checks, exited, tokens, signal),
        "exit_share_by_layer": [round(e / tokens, 4) for e in exit_share],
        "mean_depth": sum(queries) / tokens,
    }
    if classes is not None:
        count = class_count.clamp_min(1)
        row["by_token_type"] = {
            name: {
                "share_pct": round(100 * class_count[i].item() / tokens, 1),
                "mean_depth": round((class_depth[i] / count[i]).item(), 2),
                "loss": round((class_loss[i] / count[i]).item(), 3),
            }
            for i, name in enumerate(TOKEN_CLASSES)
        }
    return row


def choose(results):
    """Most compute saved while the loss stays inside the quality budget."""
    limit = results[NEVER]["loss"] * (1 + MAX_LOSS_INCREASE)
    return min((t for t, r in results.items() if r["loss"] <= limit), key=lambda t: results[t]["macs_per_token"])


def summarize(threshold, tuned, tested, reference):
    row = tested[threshold]
    return {
        "threshold": None if threshold == NEVER else round(threshold, 5),
        "saved_pct_tuning": round(100 * (1 - tuned[threshold]["macs_per_token"] / reference["tuning_macs"]), 1),
        "saved_pct_held_out": round(100 * (1 - row["macs_per_token"] / reference["macs"]), 1),
        "loss_increase_pct_held_out": round(100 * (row["loss"] / reference["loss"] - 1), 2),
        "loss": round(row["loss"], 4),
        "accuracy": round(row["accuracy"], 4),
        "mean_depth": round(row["mean_depth"], 2),
        "exit_share_by_layer": row["exit_share_by_layer"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/early_exit.json")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--swap", action="store_true", help="tune on the second half, test on the first")
    args = parser.parse_args()

    torch.manual_seed(0)
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = Transformer(Config(**state["config"])).to("cuda").eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    cfg = model.cfg

    data = Path(args.data)
    tokens = np.fromfile(data / "val.bin", dtype=np.uint16)
    middle = len(tokens) // 2
    tuning = windows(tokens[:middle], cfg.block_size, args.batch_size, "cuda")
    held_out = windows(tokens[middle:], cfg.block_size, args.batch_size, "cuda")
    if args.swap:
        tuning, held_out = held_out, tuning
    classes = class_table(BPETokenizer.load(data / "tokenizer.json"), "cuda")

    heads = nn.ModuleList(ExitHead(cfg.d_model) for _ in range(cfg.n_layer - 1)).to("cuda")
    fit_heads(model, heads, tuning, args.steps)
    heads.eval()

    results = {
        "provenance": provenance(checkpoint=args.checkpoint, checkpoint_step=state["step"], head_steps=args.steps),
        "max_loss_increase_pct": 100 * MAX_LOSS_INCREASE,
        "modes": MODES,
        "runs": {},
    }
    tuning_base = evaluate(model, heads, tuning, NEVER, "gated", "learned")
    base = evaluate(model, heads, held_out, NEVER, "gated", "learned", classes)
    reference = {"loss": base["loss"], "macs": base["macs_per_token"], "tuning_macs": tuning_base["macs_per_token"]}
    results["baseline"] = {
        "loss": round(base["loss"], 4),
        "tuning_loss": round(tuning_base["loss"], 4),
        "by_token_type": base["by_token_type"],
    }

    for signal in SIGNALS:
        thresholds = thresholds_for(model, heads, tuning, signal)
        for mode in MODES:
            tuned = {t: evaluate(model, heads, tuning, t, mode, signal) for t in thresholds}
            tested = {t: evaluate(model, heads, held_out, t, mode, signal) for t in thresholds}
            threshold = choose(tuned)
            run = summarize(threshold, tuned, tested, reference)
            run["by_token_type"] = evaluate(model, heads, held_out, threshold, mode, signal, classes)["by_token_type"]
            run["curve"] = [
                {
                    "saved_pct": round(100 * (1 - r["macs_per_token"] / reference["macs"]), 1),
                    "loss_increase_pct": round(100 * (r["loss"] / reference["loss"] - 1), 2),
                }
                for r in tested.values()
            ]
            results["runs"][f"{signal}/{mode}"] = run
            print(signal, mode, {k: v for k, v in run.items() if k not in ("curve", "by_token_type")}, flush=True)

    saved = results["runs"]["learned/gated"]["saved_pct_held_out"]
    results["decision"] = "go" if saved > 70 else "write up" if saved >= 30 else "stop"
    print("primary result:", saved, "% saved, decision:", results["decision"], flush=True)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
