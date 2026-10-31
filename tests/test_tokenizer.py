from tfs.tokenizer import BPETokenizer, SPLIT

SAMPLE = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n" * 20


def test_split_is_lossless():
    text = "class A:\n\tx = 10  # café\r\n\n  \n"
    assert "".join(SPLIT.findall(text)) == text


def test_round_trip_seen_and_unseen_text():
    tokenizer = BPETokenizer.train(SAMPLE, 300)
    for text in [SAMPLE, "lambda x: x ** 2\n", "naïve 日本語 🙂\n", ""]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_merges_compress():
    tokenizer = BPETokenizer.train(SAMPLE, 300)
    # Tokens never cross chunk boundaries, so the chunk count is the floor here.
    assert len(tokenizer.encode(SAMPLE)) == len(SPLIT.findall(SAMPLE))


def test_eot_is_outside_merge_range_and_dropped_on_decode():
    tokenizer = BPETokenizer.train(SAMPLE, 300)
    assert tokenizer.eot == tokenizer.vocab_size - 1
    assert tokenizer.decode([*tokenizer.encode("x"), tokenizer.eot]) == "x"


def test_save_and_load_match(tmp_path):
    tokenizer = BPETokenizer.train(SAMPLE, 300)
    tokenizer.save(tmp_path / "t.json")
    loaded = BPETokenizer.load(tmp_path / "t.json")
    assert loaded.encode(SAMPLE) == tokenizer.encode(SAMPLE)
