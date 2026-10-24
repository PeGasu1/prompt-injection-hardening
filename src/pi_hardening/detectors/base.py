"""Common interface: every detector maps a list of texts to P(injection) in [0, 1]."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "detector"
    threshold: float = 0.5

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an array of P(injection), one per text."""

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def unload(self) -> None:
        """Free GPU memory (overridden by neural detectors)."""
