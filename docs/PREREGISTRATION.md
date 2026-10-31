# Early-exit go/no-go test: written before running

This file was written before `early_exit.py` was run for the first time. Any later change
to the rules below must be listed under Deviations, with the reason.

## Question

Can predictable tokens leave a Transformer early without losing quality, and how much
compute would that save?

## Hypothesis

Source code contains many predictable tokens, such as closing brackets and repeated names,
so a gate on each layer can send those tokens out early. Compute should then depend more on
how hard the text is than on how long it is.

## Setup

- Model: the 6-layer checkpoint with the lowest validation loss, frozen.
- Data: the validation files only, split in half by position.
- First half: fits the exit heads and gates, and picks the gate threshold.
- Second half: produces every reported number. It is not used for any choice.
- Quality budget: loss on the tuning half may rise by at most 1%.

## Primary metric

Compute saved in the real gated run, in multiply-accumulates per token, on the held-out half.
The count includes gates, exit heads and the output projection.

Exited tokens are removed from deeper context in this run, because a real system would not
compute their deeper layers. The optimistic run keeps them in context and is reported only
as an upper bound.

## Decision rule

| Compute saved | Decision |
|---|---|
| Above 70% | Go: build the full architecture |
| 30% to 70% | Write it up, not a breakthrough |
| Below 30% | Stop this direction |

## Known limits

- One checkpoint and one seed, so this is a screening test and not a final result.
- Deeper layers are not retrained to work with sparser context. A retrained model could do better.
- Multiply-accumulates estimate compute. They are not measured latency.
- Python source is unusually repetitive, so results may not carry over to normal text.
- The output projection alone costs about 1.57M of a total of about 12.2M per token, and every
  token pays it, so the theoretical maximum saving is below 100%.

## Deviations

The primary metric and decision rule are unchanged: the learned gate, real gated run, on held-out data.

Related work was checked after this file was written and before the test ran. It showed the idea
is not new, so a "go" now means the result is worth pursuing on code, not that it is a new
architecture. See [RELATED_WORK.md](RELATED_WORK.md).

These changes were also made before the first run:

1. **Gate thresholds:** a fixed list was replaced by quantiles of each signal on the tuning half,
   so all signals get a comparable grid.
2. **Batch size and head training:** batch size dropped to 16, and exit heads train on 48 sampled
   positions per window, because full output logits would not fit in 8 GB of GPU memory.
3. **Secondary analyses, which cannot change the decision:**
   - Compare the learned gate with softmax confidence and hidden-state convergence.
   - Add a "frozen" mode, where exited tokens stop updating but stay visible as keys and values.
   - Report depth and loss by Python token type.

One analysis was added after seeing the first result, so it is exploratory and cannot change the
decision: swapping the two halves with `--swap`. Compute saved fell from 9.6% on the tuning half to
1.2% on the held-out half, and held-out loss (2.566) is well above overall validation loss (2.205).
That suggests the two halves differ, which the swap can test.

## Outcome

Primary result: 0.0% compute saved, so the decision is **stop**. The exploratory swap saved 1.8%,
which is also a stop. The direction ends here unless a new test is preregistered, for example with
retrained layers or a larger model.

## Secondary predictions, written before the first run

- Frozen mode will keep quality better than gated mode, but save less compute.
- Softmax confidence will lose on compute, because each check needs a full output projection.
- Whitespace and punctuation will exit earlier than identifiers.
