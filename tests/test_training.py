import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train import learning_rate

ROOT = Path(__file__).parent.parent


def test_learning_rate_warms_up_then_decays_to_minimum():
    args = SimpleNamespace(lr=1.0, min_lr=0.1, warmup=10, steps=100)
    assert learning_rate(0, args) == 0.1
    assert learning_rate(9, args) == 1.0
    assert learning_rate(10, args) == 1.0
    assert 0.1 < learning_rate(55, args) < 1.0
    assert abs(learning_rate(100, args) - 0.1) < 1e-9


def run_train(data, out, steps, *extra):
    command = [
        sys.executable, "train.py", "--data", str(data), "--out", str(out), "--steps", str(steps),
        "--warmup", "1", "--eval-every", "2", "--eval-batches", "1", "--batch-size", "2",
        "--block-size", "16", "--n-layer", "1", "--n-head", "2", "--d-model", "16", *extra,
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)


def test_training_runs_end_to_end_and_resumes_exactly(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 50, 2000, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 50, "train": {"tokens": 2000}}))

    out = tmp_path / "run"
    run_train(tmp_path, out, 2)
    run_train(tmp_path, out, 4, "--resume")

    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 2, 2, 4]
    # Reloading the checkpoint must reproduce the loss measured just before saving.
    assert rows[1]["val_loss"] == rows[2]["val_loss"]
