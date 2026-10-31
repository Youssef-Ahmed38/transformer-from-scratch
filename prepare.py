"""Build the dataset from the Python standard library already on this machine."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tfs.tokenizer import BPETokenizer

SKIP_DIRS = {"site-packages", "__pycache__", "test", "tests", "idlelib", "turtledemo", "lib2to3"}


def collect(root):
    files = []
    for path in sorted(Path(root).rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def is_val(relpath):
    # Split by file, so no file appears on both sides. A hash keeps it stable across runs.
    return hashlib.sha1(relpath.encode()).digest()[0] < 26  # about 10%


def bigram_baseline(train, val, vocab_size, smoothing=0.1):
    """Cross-entropy of a smoothed bigram model. The Transformer has to beat this."""
    counts = np.bincount(train[:-1].astype(np.int64) * vocab_size + train[1:], minlength=vocab_size**2)
    counts = counts.reshape(vocab_size, vocab_size) + smoothing
    probs = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.log(probs[val[:-1], val[1:]]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(os.__file__).parent))
    parser.add_argument("--out", default="data")
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    texts = {"train": [], "val": []}
    for path in collect(args.root):
        text = path.read_text(encoding="utf-8", errors="replace")
        texts["val" if is_val(path.relative_to(args.root).as_posix()) else "train"].append(text)

    start = time.perf_counter()
    tokenizer = BPETokenizer.train("".join(texts["train"]), args.vocab_size)
    tokenizer.save(out / "tokenizer.json")
    train_seconds = time.perf_counter() - start

    meta = {
        "source": f"Python {sys.version.split()[0]} standard library",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_train_seconds": round(train_seconds, 1),
    }
    arrays = {}
    for split, docs in texts.items():
        ids = []
        for doc in docs:
            ids.extend(tokenizer.encode(doc))
            ids.append(tokenizer.eot)
        arrays[split] = np.array(ids, dtype=np.uint16)
        arrays[split].tofile(out / f"{split}.bin")
        chars = sum(len(d) for d in docs)
        meta[split] = {"files": len(docs), "chars": chars, "tokens": len(ids)}
    meta["chars_per_token"] = round(meta["train"]["chars"] / meta["train"]["tokens"], 2)
    meta["uniform_loss"] = round(float(np.log(tokenizer.vocab_size)), 3)
    meta["bigram_val_loss"] = round(bigram_baseline(arrays["train"], arrays["val"], tokenizer.vocab_size), 3)

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
