"""Does the model write valid Python? Compare it with a bigram model, and check for copying."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from tfs.model import Config, Transformer
from tfs.provenance import provenance
from tfs.tokenizer import BPETokenizer


def first_function(text):
    """The first complete top-level function, or None if the sample ended inside it."""
    lines = text.split("\n")
    for i, line in enumerate(lines[1:], 1):
        if line and not line[0].isspace() and not line.startswith(")"):
            return "\n".join(lines[:i]).rstrip()
    return None


def parses(source):
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def score(samples, train_text):
    functions = [f for f in map(first_function, samples) if f is not None]
    valid = [f for f in functions if parses(f)]
    # Short functions match by chance, so only multi-line ones count as copying.
    longer = [f for f in valid if f.count("\n") >= 3]
    copied = sum(f in train_text for f in longer)
    return {
        "samples": len(samples),
        "complete_pct": round(100 * len(functions) / len(samples), 1),
        "valid_pct": round(100 * len(valid) / len(samples), 1),
        "valid_of_complete_pct": round(100 * len(valid) / max(1, len(functions)), 1),
        "valid_multiline": len(longer),
        "copied_from_train": copied,
    }


def bigram_samples(train, prompt_ids, vocab_size, tokens, n, temperature, top_k, seed=0):
    counts = np.bincount(train[:-1].astype(np.int64) * vocab_size + train[1:], minlength=vocab_size**2)
    counts = counts.reshape(vocab_size, vocab_size).astype(np.float64) + 0.1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ids = list(prompt_ids)
        for _ in range(tokens):
            row = counts[ids[-1]]
            top = np.argpartition(row, -top_k)[-top_k:]
            probs = (row[top] / row[top].sum()) ** (1 / temperature)
            ids.append(int(rng.choice(top, p=probs / probs.sum())))
        out.append(ids)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/code_eval.json")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    tokenizer = BPETokenizer.load(data / "tokenizer.json")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Transformer(Config(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])

    train = np.fromfile(data / "train.bin", dtype=np.uint16)
    train_text = tokenizer.decode(train.tolist())
    prompt = tokenizer.encode("def ")

    results = {
        "provenance": provenance(checkpoint=args.checkpoint, checkpoint_step=state["step"], tokens=args.tokens),
        "prompt": "def ",
        "runs": [],
    }
    for temperature in (0.5, 0.8, 1.0):
        torch.manual_seed(0)
        batches = []
        for _ in range(args.samples // 100):
            start = torch.tensor([prompt] * 100, device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                out = model.generate(start, args.tokens, temperature, args.top_k)
            batches.extend(out.tolist())
        row = {"model": "transformer", "temperature": temperature, **score(map_decode(tokenizer, batches), train_text)}
        results["runs"].append(row)
        print(row, flush=True)

    ids = bigram_samples(train, prompt, tokenizer.vocab_size, args.tokens, args.samples, 0.8, args.top_k)
    row = {"model": "bigram", "temperature": 0.8, **score(map_decode(tokenizer, ids), train_text)}
    results["runs"].append(row)
    print(row, flush=True)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def map_decode(tokenizer, batches):
    return [tokenizer.decode(ids) for ids in batches]


if __name__ == "__main__":
    main()
