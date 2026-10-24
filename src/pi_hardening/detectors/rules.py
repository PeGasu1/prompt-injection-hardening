"""Baseline 1: keyword / regular-expression rules, the simplest deployable filter.

Score = 1 − exp(−number of matched patterns), so one match already crosses the 0.5 threshold.
The patterns describe common *wording* of injection attempts (published in many guardrail
projects); they are the kind of filter that obfuscation is designed to defeat.
"""
from __future__ import annotations

import re

import numpy as np

from .base import Detector

PATTERNS = [
    r"\b(ignore|disregard|forget|skip|override)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines|directions)\b",
    r"\b(previous|prior|above|earlier|original)\s+(instructions?|rules?|prompts?|context)\b",
    r"\bsystem\s+prompt\b",
    r"\b(reveal|show|print|output|repeat|leak)\b.{0,40}\b(prompt|instructions?|password|secret|rules)\b",
    r"\byou\s+are\s+(now|no\s+longer)\b",
    r"\b(act|behave)\s+as\b",
    r"\bpretend\s+(to\s+be|you)\b",
    r"\b(developer|god|jailbreak|dan)\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"\b(no|without)\s+(restrictions|limits|limitations|rules|filters|censorship)\b",
    r"\bnew\s+instructions?\b",
    r"\bfrom\s+now\s+on\b",
    r"\bjailbreak\w*\b",
]


class RuleDetector(Detector):
    name = "rules"

    def __init__(self, patterns=PATTERNS):
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]

    def n_matches(self, text: str) -> int:
        return sum(bool(p.search(text)) for p in self.patterns)

    def predict_proba(self, texts):
        return np.array([1.0 - np.exp(-self.n_matches(t)) for t in texts])
