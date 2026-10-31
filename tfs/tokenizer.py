"""Byte-level BPE tokenizer, trained from scratch on code."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Lossless by construction: "." with DOTALL matches anything the other branches miss.
# Newline plus indent is one chunk, because indentation is syntax in Python.
SPLIT = re.compile(r"[ \t]*[A-Za-z_]+|[ \t]*\d+|\n[ \t]*|[ \t]+|.", re.S)


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]
        self.eot = 256 + len(self.merges)
        self.vocab_size = self.eot + 1
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size):
        counts = Counter(SPLIT.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in counts]
        freqs = list(counts.values())

        pairs = Counter()
        where = defaultdict(set)
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pairs[pair] += freqs[i]
                where[pair].add(i)

        merges = []
        for new_id in range(256, vocab_size - 1):
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            if pairs[best] < 2:
                break
            merges.append(best)
            # Only words containing `best` change, so update their counts in place.
            for i in where.pop(best):
                word, freq = words[i], freqs[i]
                for pair in zip(word, word[1:]):
                    pairs[pair] -= freq
                    if pairs[pair] <= 0:
                        del pairs[pair]
                word = words[i] = merge(word, best, new_id)
                for pair in zip(word, word[1:]):
                    pairs[pair] += freq
                    where[pair].add(i)
        return cls(merges)

    def _encode_chunk(self, chunk):
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in SPLIT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids if i != self.eot)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])
