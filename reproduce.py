"""Rebuild the data, model and every published result with one command."""
import argparse
import subprocess
import sys

STEPS = {
    "test": ["-m", "pytest", "-q"],
    "prepare": ["prepare.py"],
    "train": ["train.py", "--steps", "8000", "--dropout", "0.2", "--out", "checkpoints/run"],
    "benchmark": ["benchmark.py"],
    "eval": ["eval_code.py"],
    "early-exit": ["early_exit.py"],
    "plot": ["plot.py"],
    "visualize": ["visualize.py"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=STEPS, default="test", help="skip earlier steps")
    args = parser.parse_args()

    names = list(STEPS)
    for name in names[names.index(args.start) :]:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, *STEPS[name]], check=True)


if __name__ == "__main__":
    main()
