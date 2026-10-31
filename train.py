import argparse
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from tfs.data import Batches
from tfs.model import Config, Transformer


def learning_rate(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, batches, n_batches, autocast):
    model.eval()
    batches.generator.manual_seed(1234)  # same batches every time, so curves are comparable
    total = 0.0
    for _ in range(n_batches):
        x, y = batches.get()
        with autocast():
            total += model(x, targets=y)[1].item()
    model.train()
    return total / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="checkpoints/run")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))

    cfg = Config(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        attn="sdpa",
    )
    model = Transformer(cfg).to(device)
    optimizer = model.optimizer(args.lr, args.weight_decay, fused=device == "cuda")
    autocast = partial(torch.autocast, device, dtype=torch.bfloat16)
    train = Batches(data / "train.bin", cfg.block_size, args.batch_size, device, seed=args.seed)
    val = Batches(data / "val.bin", cfg.block_size, args.batch_size, device, seed=args.seed + 1)

    step = 0
    best_val = float("inf")
    if args.resume:
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best_val = state["step"], state["best_val"]

    print(f"{model.num_params() / 1e6:.2f}M parameters, {meta['train']['tokens'] / 1e6:.2f}M train tokens")

    def save(name):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "step": step,
                "best_val": best_val,
            },
            out / name,
        )

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    started = time.perf_counter()
    last_step, last_time = step, started
    while True:
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(model, train, args.eval_batches, autocast)
            val_loss = estimate_loss(model, val, args.eval_batches, autocast)
            now = time.perf_counter()
            rate = (step - last_step) / (now - last_time) if step > last_step else 0.0
            last_step, last_time = step, now
            row = {
                "step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "steps_per_sec": round(rate, 1),
                "elapsed_sec": round(now - started, 1),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            if val_loss < best_val and step > 0:
                best_val = val_loss
                save("best.pt")
            save("last.pt")
        if step == args.steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)
        x, y = train.get()
        with autocast():
            loss = model(x, targets=y)[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1
    log.close()


if __name__ == "__main__":
    main()
