"""Character-level obfuscation families used to test detector robustness.

These are the generic text transformations studied in the guardrail-evasion literature
(Hackett et al., 2025). They are applied only to *existing* dataset samples, of both classes,
so that we can measure (a) how often a detector misses obfuscated attacks and (b) whether
obfuscation alone makes it flag benign text.

Each family obfuscates a share ``rate`` of the eligible words (alphabetic, ≥ 3 letters).
Every call is deterministic given ``seed``.
"""
from __future__ import annotations

import hashlib
import random
import re

ZERO_WIDTH = ["​", "‌", "‍", "⁠", "﻿"]

# Hand-picked look-alikes (Cyrillic, Greek, Latin extensions). The normaliser does NOT use this
# table: it uses the independent Unicode confusables list (UTS #39), see normalize.py.
HOMOGLYPHS = {
    "a": ["а", "ɑ", "α"], "c": ["с", "ϲ"], "d": ["ԁ"], "e": ["е", "ҽ"], "g": ["ɡ"], "h": ["һ"],
    "i": ["і", "ı", "ɩ"], "j": ["ј"], "k": ["κ"], "l": ["ӏ"], "n": ["ո"], "o": ["о", "ο", "σ"],
    "p": ["р", "ρ"], "q": ["ԛ"], "r": ["г"], "s": ["ѕ"], "u": ["υ", "ս"], "v": ["ν"],
    "w": ["ԝ"], "x": ["х"], "y": ["у"],
    "A": ["А", "Α"], "B": ["В", "Β"], "C": ["С"], "E": ["Е", "Ε"], "H": ["Н", "Η"], "I": ["І", "Ι"],
    "K": ["К", "Κ"], "M": ["М", "Μ"], "N": ["Ν"], "O": ["О", "Ο"], "P": ["Р", "Ρ"], "S": ["Ѕ"],
    "T": ["Т", "Τ"], "X": ["Х", "Χ"], "Y": ["Υ"],
}

COMBINING = ["́", "̀", "̈", "̂", "̃", "̧", "̄"]

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "g": "9", "b": "8"}

PUNCT = [".", "-", "_", "*"]

WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _rng(seed, *parts) -> random.Random:
    h = hashlib.sha256(("|".join(map(str, (seed,) + parts))).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _insert_between(word: str, chars: list[str], rng: random.Random, k: int) -> str:
    """Insert ``k`` characters (drawn from ``chars``) at distinct interior positions."""
    positions = sorted(rng.sample(range(1, len(word)), min(k, len(word) - 1)))
    out, last = [], 0
    for p in positions:
        out.append(word[last:p])
        out.append(rng.choice(chars))
        last = p
    out.append(word[last:])
    return "".join(out)


def _zero_width(word, rng):
    return _insert_between(word, ZERO_WIDTH, rng, max(1, len(word) // 3))


def _homoglyph(word, rng):
    idx = [i for i, ch in enumerate(word) if ch in HOMOGLYPHS]
    if not idx:
        return word
    chosen = set(rng.sample(idx, max(1, len(idx) // 2)))
    return "".join(rng.choice(HOMOGLYPHS[ch]) if i in chosen else ch for i, ch in enumerate(word))


def _diacritics(word, rng):
    chosen = set(rng.sample(range(len(word)), max(1, len(word) // 2)))
    return "".join(ch + (rng.choice(COMBINING) if i in chosen else "") for i, ch in enumerate(word))


def _leetspeak(word, rng):
    idx = [i for i, ch in enumerate(word) if ch.lower() in LEET]
    if not idx:
        return word
    chosen = set(rng.sample(idx, max(1, (len(idx) * 2 + 2) // 3)))
    return "".join(LEET[ch.lower()] if i in chosen else ch for i, ch in enumerate(word))


def _char_spacing(word, rng):
    return " ".join(word)


def _punct_split(word, rng):
    sep = rng.choice(PUNCT)
    return _insert_between(word, [sep], rng, max(1, len(word) // 3))


TRANSFORMS = {
    "zero_width": _zero_width,
    "homoglyph": _homoglyph,
    "diacritics": _diacritics,
    "leetspeak": _leetspeak,
    "char_spacing": _char_spacing,
    "punct_split": _punct_split,
}

DESCRIPTIONS = {
    "zero_width": "invisible zero-width characters inserted inside words",
    "homoglyph": "letters swapped for look-alike Cyrillic/Greek/Latin-extended characters",
    "diacritics": "combining accent marks added to letters",
    "leetspeak": "letters replaced by similar-looking digits (e→3, o→0, …)",
    "char_spacing": "spaces inserted between every letter of a word",
    "punct_split": "punctuation (. - _ *) inserted inside words",
}


def perturb(text: str, family: str, rate: float = 0.5, seed=0) -> str:
    """Obfuscate a share ``rate`` of eligible words in ``text`` with one family."""
    if family == "clean":
        return text
    fn = TRANSFORMS[family]
    rng = _rng(seed, family, text)
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return text
    chosen = [m for m in matches if rng.random() < rate] or [rng.choice(matches)]
    out, last = [], 0
    for m in chosen:
        out.append(text[last:m.start()])
        out.append(fn(m.group(), rng))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def perturb_random(text: str, families: list[str], rate_range=(0.2, 1.0), rng: random.Random = None) -> str:
    """Pick a family and rate at random (used for training-time augmentation)."""
    rng = rng or random.Random()
    fam = rng.choice(families)
    rate = rng.uniform(*rate_range)
    return perturb(text, fam, rate, seed=rng.random())
