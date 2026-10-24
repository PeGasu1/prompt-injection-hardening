"""Hardening method 1: a text normaliser applied before any detector.

It needs no training and can be placed in front of any detector. Each step targets one kind
of obfuscation; steps can be switched off individually for the ablation in notebook 06.

Step order matters:

1. ``nfkc``         Unicode NFKC (full-width and "mathematical" letters → ASCII)
2. ``format``       delete invisible format characters (Unicode category Cf: zero-width, BOM…)
3. ``marks``        delete combining accent marks (NFKD, drop category Mn, recompose)
4. ``confusables``  map look-alike characters to ASCII using the Unicode confusables list
                    (UTS #39, shipped with the ``confusable-homoglyphs`` package). This list is
                    independent of the hand-made table in perturb.py.
5. ``deleet``       inside tokens that mix letters and digits, map digits back to letters
6. ``punct``        delete . - _ * · | between two letters ("ig.no-re" → "ignore")
7. ``spacing``      join runs of ≥ 3 single letters separated by spaces ("i g n o r e" → "ignore")
8. ``whitespace``   collapse repeated whitespace

Steps 5–7 can damage benign text ("COVID-19", "e.g.", "well-known"); the over-defence and
false-positive measurements in the experiments quantify that cost.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache

STEPS = ["nfkc", "format", "marks", "confusables", "deleet", "punct", "spacing", "whitespace"]

LEET_REVERSE = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "9": "g", "8": "b",
                "@": "a", "$": "s"}

_PUNCT_IN_WORD = re.compile(r"(?<=[A-Za-z])[.\-_*·•|](?=[A-Za-z])")
_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){2,}[A-Za-z](?![A-Za-z])")
_TOKEN = re.compile(r"\S+")


@lru_cache(maxsize=1)
def confusable_map() -> dict[str, str]:
    """Non-ASCII character → single ASCII letter/digit it is visually confusable with."""
    import confusable_homoglyphs
    path = os.path.join(os.path.dirname(confusable_homoglyphs.__file__), "confusables.json")
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    out = {}
    for ch, entries in table.items():
        if len(ch) != 1 or ord(ch) < 128:
            continue
        for e in entries:
            c = e["c"]
            if len(c) == 1 and c.isascii() and c.isalnum():
                out.setdefault(ch, c)
                break
    return out


def _deleet_token(tok: str) -> str:
    letters = sum(ch.isalpha() for ch in tok)
    leet = sum(ch in LEET_REVERSE for ch in tok)
    if letters == 0 or leet == 0 or letters / (letters + leet) < 0.4:
        return tok
    return "".join(LEET_REVERSE.get(ch, ch) for ch in tok)


class Normalizer:
    def __init__(self, steps: list[str] | None = None):
        self.steps = list(STEPS if steps is None else steps)
        unknown = set(self.steps) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown steps {unknown}")
        self._cmap = confusable_map() if "confusables" in self.steps else {}

    def __call__(self, text: str) -> str:
        s = text
        for step in STEPS:            # always apply in canonical order
            if step in self.steps:
                s = getattr(self, f"_{step}")(s)
        return s

    def batch(self, texts):
        return [self(t) for t in texts]

    # individual steps -------------------------------------------------------------
    @staticmethod
    def _nfkc(s):
        return unicodedata.normalize("NFKC", s)

    @staticmethod
    def _format(s):
        return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")

    @staticmethod
    def _marks(s):
        d = unicodedata.normalize("NFKD", s)
        return unicodedata.normalize("NFC", "".join(ch for ch in d if unicodedata.category(ch) != "Mn"))

    def _confusables(self, s):
        return "".join(self._cmap.get(ch, ch) for ch in s)

    @staticmethod
    def _deleet(s):
        return _TOKEN.sub(lambda m: _deleet_token(m.group()), s)

    @staticmethod
    def _punct(s):
        return _PUNCT_IN_WORD.sub("", s)

    @staticmethod
    def _spacing(s):
        return _SPACED.sub(lambda m: m.group().replace(" ", ""), s)

    @staticmethod
    def _whitespace(s):
        return re.sub(r"[ \t]+", " ", s).strip()
