from eval_code import first_function, parses, score


def test_first_function_stops_at_next_top_level_line():
    text = "def a(x):\n    return x\n\ndef b():\n    pass\n"
    assert first_function(text) == "def a(x):\n    return x"


def test_first_function_keeps_multiline_signature():
    text = "def a(\n    x,\n):\n    return x\nprint(1)\n"
    assert first_function(text) == "def a(\n    x,\n):\n    return x"


def test_sample_that_ends_inside_a_function_is_incomplete():
    assert first_function("def a():\n    return 1") is None


def test_parses_rejects_broken_code():
    assert parses("def a():\n    return 1")
    assert not parses("def a(:\n    return 1")


def test_score_counts_incomplete_samples_as_invalid():
    samples = ["def a():\n    return 1\nx", "def b(:\n    pass\nx", "def c():\n    pa"]
    result = score(samples, "")
    assert result["complete_pct"] == 66.7
    assert result["valid_pct"] == 33.3


def test_score_detects_copied_functions():
    function = "def a():\n    x = 1\n    y = 2\n    return x + y"
    result = score([function + "\nz"], train_text=function)
    assert result["copied_from_train"] == 1
