"""Measure training and generation throughput, one optimization at a time."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs.data import Batches
from tfs.model import Config, Transformer
from tfs.provenance import provenance

DEVICE = "cuda"
BATCH_SIZE = 64
GENERATION_BATCH = 16
SKIP_STEPS = 20  # allocation and compilation noise


def build(meta, attn):
    torch.manual_seed(0)
    return Transformer(Config(vocab_size=meta["vocab_size"], attn=attn)).to(DEVICE)


def run_training(meta, data, stages, steps, repeats):
    """Each stage keeps the previous stage's settings and changes one thing."""
    results = []
    for stage in stages:
        rates, peak = [], 0
        for repeat in range(repeats):
            model = build(meta, stage["attn"])
            optimizer = model.optimizer(3e-4, 0.1, fused=stage["fused"])
            batches = Batches(data / "train.bin", 256, BATCH_SIZE, DEVICE, stage["data"], seed=repeat)
            torch.cuda.reset_peak_memory_stats()

            for step in range(SKIP_STEPS + steps):
                if step == SKIP_STEPS:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                x, y = batches.get()
                with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=stage["amp"]):
                    loss = model(x, targets=y)[1]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            rates.append(steps / (time.perf_counter() - start))
            peak = max(peak, torch.cuda.max_memory_allocated())
            del model, optimizer, batches
            torch.cuda.empty_cache()

        row = {
            "stage": stage["name"],
            "steps_per_sec": round(statistics.median(rates), 2),
            "min": round(min(rates), 2),
            "max": round(max(rates), 2),
            "tokens_per_sec": round(statistics.median(rates) * BATCH_SIZE * 256),
            "peak_gpu_mb": round(peak / 2**20),
        }
        results.append(row)
        print(row, flush=True)
    return results


def breakdown(meta, data, mode, attn, amp, steps=200):
    """Where one step's time goes. Syncing adds overhead, so read this as shares, not speed."""
    model = build(meta, attn)
    optimizer = model.optimizer(3e-4, 0.1, fused=False)
    batches = Batches(data / "train.bin", 256, BATCH_SIZE, DEVICE, mode)
    totals = dict.fromkeys(["data", "forward", "backward", "optimizer"], 0.0)

    def mark(name, since):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if step >= SKIP_STEPS:
            totals[name] += now - since
        return now

    for step in range(SKIP_STEPS + steps):
        t = time.perf_counter()
        x, y = batches.get()
        t = mark("data", t)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=amp):
            loss = model(x, targets=y)[1]
        t = mark("forward", t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t = mark("backward", t)
        optimizer.step()
        mark("optimizer", t)

    whole = sum(totals.values())
    return {
        "ms_per_step": round(whole / steps * 1000, 2),
        "share_pct": {k: round(v / whole * 100, 1) for k, v in totals.items()},
    }


def run_generation(meta, lengths, repeats):
    model = build(meta, "sdpa").eval()
    # Steps often take about 30 ms here whatever the workload, so read these numbers with care.
    start = torch.zeros(GENERATION_BATCH, 1, dtype=torch.long, device=DEVICE)
    model.generate(start, 32, top_k=50)
    rows = []
    for length in lengths:
        for use_cache in (False, True):
            rates = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    model.generate(start, length, top_k=50, use_cache=use_cache)
                torch.cuda.synchronize()
                rates.append(GENERATION_BATCH * length / (time.perf_counter() - began))
            row = {"new_tokens": length, "kv_cache": use_cache, "tokens_per_sec": round(statistics.median(rates), 1)}
            rows.append(row)
            print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/benchmark.json")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))

    base = {"data": "python", "attn": "manual", "amp": False, "fused": False}
    changes = [
        ("baseline", {}),
        ("+ data on GPU", {"data": "gpu"}),
        ("+ SDPA attention", {"attn": "sdpa"}),
        ("+ bfloat16 autocast", {"amp": True}),
        ("+ fused AdamW", {"fused": True}),
    ]
    stages = []
    for name, change in changes:
        base = {**base, **change}
        stages.append({"name": name, **base})

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)

    def save():
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results = {
        "provenance": provenance(steps=args.steps, repeats=args.repeats),
        "batch_size": BATCH_SIZE,
        "block_size": 256,
        "breakdown_baseline": breakdown(meta, data, "python", "manual", False),
        "breakdown_after_data_fix": breakdown(meta, data, "gpu", "manual", False),
    }
    print(results, flush=True)
    results["training"] = run_training(meta, data, stages, args.steps, args.repeats)
    save()

    # Cumulative stages depend on their order. Removing one change at a time does not.
    full = stages[-1]
    removals = [
        ("without data on GPU", {"data": "python"}),
        ("without SDPA attention", {"attn": "manual"}),
        ("without bfloat16 autocast", {"amp": False}),
        ("without fused AdamW", {"fused": False}),
    ]
    ablation = [{**full, "name": name, **change} for name, change in removals]
    results["ablation"] = run_training(meta, data, ablation, args.steps, args.repeats)
    save()
    results["generation"] = run_generation(meta, [64, 128, 255], args.repeats)
    save()


if __name__ == "__main__":
    main()
