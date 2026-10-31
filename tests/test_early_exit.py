import numpy as np
import pytest
import torch

from early_exit import NEVER, choose, exit_score, token_class, total_macs, windows
from tfs.model import Config


def test_full_model_macs_match_hand_count():
    cfg = Config(vocab_size=4096)
    tokens = 256
    pairs = tokens * (tokens + 1) // 2
    macs = total_macs(cfg, [tokens] * 6, [pairs] * 6, [0] * 6, 0, 0, tokens, "learned")

    d = cfg.d_model
    assert macs == 6 * 12 * d * d + 6 * 2 * d * (tokens + 1) / 2 + d * 4096


def test_exits_reduce_cost_but_frozen_tokens_still_pay_for_keys():
    cfg = Config(vocab_size=4096)
    gated = total_macs(cfg, [100, *[50] * 5], [5050, *[1275] * 5], [0] * 6, 500, 50, 100, "learned")
    frozen = total_macs(cfg, [100, *[50] * 5], [5050, *[1275] * 5], [0, *[50] * 5], 500, 50, 100, "learned")
    full = total_macs(cfg, [100] * 6, [5050] * 6, [0] * 6, 0, 0, 100, "learned")
    assert gated < frozen < full


def test_confidence_check_counts_a_full_output_projection():
    cfg = Config(vocab_size=4096)
    learned = total_macs(cfg, [100] * 6, [5050] * 6, [0] * 6, 500, 0, 100, "learned")
    confidence = total_macs(cfg, [100] * 6, [5050] * 6, [0] * 6, 500, 0, 100, "confidence")
    assert confidence > learned + 4 * cfg.d_model * cfg.vocab_size


def test_choose_takes_cheapest_threshold_inside_budget():
    results = {
        0.5: {"loss": 2.10, "macs_per_token": 100},
        0.9: {"loss": 2.015, "macs_per_token": 200},
        NEVER: {"loss": 2.00, "macs_per_token": 300},
    }
    assert choose(results) == 0.9


def test_choose_falls_back_to_never_exiting():
    results = {0.9: {"loss": 3.0, "macs_per_token": 100}, NEVER: {"loss": 2.0, "macs_per_token": 300}}
    assert choose(results) == NEVER


def test_windows_cover_data_without_overlap_and_targets_shift_by_one():
    tokens = np.arange(21, dtype=np.uint16)
    batches = windows(tokens, 5, 2, "cpu")
    x = torch.cat([b[0] for b in batches])
    y = torch.cat([b[1] for b in batches])
    assert x.shape == (4, 5)
    assert x.reshape(-1).tolist() == list(range(20))
    assert y.reshape(-1).tolist() == list(range(1, 21))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\n    ", "whitespace"),
        (" def", "keyword"),
        (" path", "identifier"),
        ("42", "number"),
        ("):", "punctuation"),
        ("a.b", "fragment"),
    ],
)
def test_token_classes(text, expected):
    assert token_class(text) == expected


def test_convergence_score_is_higher_when_the_state_stops_changing():
    h = torch.ones(1, 1, 8)
    steady = exit_score("convergence", None, None, h, h * 0.99)
    moving = exit_score("convergence", None, None, h, h * 0.2)
    assert steady > moving
