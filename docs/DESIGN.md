# Design

## Goals

- Implement the parts that are usually imported: attention, normalization, tokenizer and KV cache.
- Prove correctness with tests that target known Transformer bugs.
- Measure speed and quality against explicit baselines.
- Keep every result reproducible on one laptop GPU.

## Non-goals

- Competing with production models. At 12.31M parameters, this model is not useful as a coding tool.
- Multi-GPU training. That is a separate project.
- Hiding framework primitives for its own sake. `nn.Linear`, `nn.Embedding` and `AdamW` stay.

## Decisions

| Decision | Why | Alternative rejected |
|---|---|---|
| Train on the Python standard library | Already on the machine, licensed for reuse, and relevant to coding agents | Tiny Shakespeare, which is common and has no structure to test |
| Split train and validation by file | A random split would put nearly identical code on both sides | Splitting by position within files |
| Custom BPE that keeps newline and indent together | Indentation is syntax in Python | Generic whitespace handling |
| Pre-norm blocks | Generally more stable as depth grows, and residual paths stay clean | Post-norm from the original paper |
| Weight tying | Saves about 1.57M parameters | A separate output matrix |
| Hand-written and SDPA attention side by side | The hand version documents the math, and the test proves both agree | One backend only |
| Keep the dataset on the GPU | No Python per-sample loop on the critical path | A memmap read per sample |
| Rebuild the KV cache from half a block | Positions are absolute, so a full cache cannot slide | Rebuilding on every token past the limit |

## What can go wrong, and what catches it

| Failure | Check |
|---|---|
| Future tokens leak through attention | Perturbation test and gradient test |
| KV cache drifts from the full forward pass | Exact comparison at every step |
| Hand-written attention differs from SDPA | Agreement test |
| Tokenizer loses text | Round-trip tests, including Unicode and unseen text |
| Train and validation overlap | Hash split by file |
| Training looks good but memorizes | Unseen-file loss and the copied-function check |
| A resumed run changes silently | End-to-end resume test |
| Benchmark timing is wrong on an asynchronous GPU | `torch.cuda.synchronize` around every timed section |

## Known limits

- Overfitting starts by about step 3,500, because 3.09M tokens is small for this model.
- Results come from one seed unless a result says otherwise.
- The code-quality eval checks syntax, not whether the code does anything useful.
