# Related work on per-token adaptive depth

Checked before running `early_exit.py`. This is a focused search, not a systematic review.
Entries marked "abstract" were read at abstract level, so details may be incomplete.

## Conclusion

Per-token early exit is an established and active field, so this project does not claim a new
architecture. In particular, using a hidden state that has stopped changing as the exit signal
already exists in TIDE, and that is very close to the surprise signal proposed here.

| Work | Year | Approach | Reported result |
|---|---|---|---|
| [CALM](https://arxiv.org/abs/2207.07061), Schuster et al. | 2022 | Exit per token on confidence, with calibrated quality guarantees | Up to about 3x speedup (search summary, not checked in the paper) |
| [EE-LLM](https://arxiv.org/abs/2312.04916) | 2023 | Large-scale early-exit training and inference | Not reviewed in detail |
| [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | 2024 | Per-token gating at every layer | Not reviewed in detail |
| [Dynamic layer selection](https://arxiv.org/abs/2410.20022) | 2024 | Compares layer skipping with early exit | Controllers must stay light, because overhead accumulates |
| [DASH](https://arxiv.org/abs/2505.17420) | 2025 | Learned skip, compute or exit decisions | Not reviewed in detail |
| [ADEPT](https://arxiv.org/abs/2601.03700), Yoo et al. | 2026 | Token-level exit, decouples skipped-layer dependencies | Up to 25% in language generation (abstract) |
| [TIDE](https://arxiv.org/abs/2603.21365), Jaber and Jaber | 2026 | Routers find the layer where a token's hidden state converged, no retraining | 6.6% to 8.1% throughput, 7.2% prefill latency (abstract) |
| [Token-Selective Attention](https://arxiv.org/abs/2605.05222), Mohammed | 2026 | Learned differentiable per-token gates | 14% to 23% of token-layer operations at under 0.5% quality loss (abstract) |

## Newest work checked, as of 30 October 2026

| Work | Date | Contribution |
|---|---|---|
| [Probe and Skip](https://arxiv.org/abs/2601.13155), Wu et al. | January 2026, v3 on 13 October 2026 | Training-free token skipping for long context, up to 2.46x prefill and 2.29x end-to-end (abstract) |
| [Continuous Depth Batching](https://arxiv.org/abs/2608.09444), Schwethelm et al. | August 2026, v2 on 25 September 2026 | Serves depth-adaptive looped models efficiently, up to 99% of the estimated maximum speedup. Finds that large non-looped parts such as the LM head complicate scheduling (abstract) |
| [FlexEE](https://arxiv.org/abs/2609.17008) | About September 2026, from its ID | KV-compatible early exit that avoids updating later-layer caches (search summary) |
| [GateSkip](https://proceedings.iclr.cc/paper_files/paper/2026/file/7e3491e922bfd199ea34ecafeb7380f0-Paper-Conference.pdf) | ICLR 2026 | Token-wise residual gates with a per-layer budget (search summary) |

The field has moved on from "can tokens exit early?" Recent work focuses on serving: KV cache
consistency, batching, and looped models where the same layers repeat. A fuller date-filtered
arXiv search could find more.

## What is still worth testing here

These are possible gaps, not established ones. Each needs checking against the full papers.

1. **Routing by token type.** Token-Selective Attention
   [names quantitative token-type analysis as future work](https://arxiv.org/html/2605.05222).
   This project measures depth and loss per Python token type.
2. **Source code.** No search here found early-exit results on code. Code may be far more
   predictable, or its long-range references may break when context is removed.
3. **Exit signals on one dataset.** This project compares a learned gate, softmax confidence and
   hidden-state convergence under the same budget. The convergence signal here is a simple
   version inspired by TIDE, not TIDE itself.
4. **Three context policies.** Exited tokens are removed, frozen or kept (upper bound), and the
   quality cost of each is measured.
5. **Cost accounting.** Token-layer operations can hide the output projection. Continuous Depth
   Batching also flags the LM head as a problem. This project counts multiply-accumulates,
   including gates and output.

## How this changes the claims

- The early-exit result can be reported as a measurement on code. It cannot be called a new
  architecture, whatever the number turns out to be.
- A negative or modest result is still useful, and it would agree with the modest gains above.
