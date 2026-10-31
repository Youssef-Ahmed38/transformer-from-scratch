import argparse

import torch

from tfs.model import Config, Transformer
from tfs.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Transformer(Config(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])

    tokenizer = BPETokenizer.load(args.tokenizer)
    idx = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = model.generate(idx, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
