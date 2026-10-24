"""Hardening method 3 (and baseline 2): a byte-level convolutional classifier (course week 4).

Unlike a sub-word Transformer, it reads raw UTF-8 bytes, so it has no tokeniser that an
invisible character or a look-alike letter can break. Architecture (Kim-style multi-kernel CNN
over a character sequence, Zhang et al. 2015):

    bytes → embedding(257, 64) → [Conv1d(k=3,5,7) × 128 filters → ReLU → global max-pool]
          → concat(384) → dropout(0.3) → linear → logit
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .. import config
from ..perturb import perturb_random
from ..utils import get_device, set_seed
from .base import Detector


def encode_bytes(texts, max_len):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        b = list(t.encode("utf-8", errors="ignore"))[:max_len]
        if b:
            out[i, :len(b)] = torch.tensor(b) + 1   # 0 is padding
    return out


class CharCNN(nn.Module):
    def __init__(self, cfg: config.CharCNNConfig):
        super().__init__()
        self.emb = nn.Embedding(257, cfg.emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.emb_dim, cfg.n_filters, k, padding=k // 2)
                                    for k in cfg.kernel_sizes])
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.n_filters * len(cfg.kernel_sizes), 1)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)                      # (B, E, L)
        mask = (x != 0).unsqueeze(1)                         # ignore padding in the max-pool
        pooled = [torch.relu(c(h)).masked_fill(~mask, 0).max(dim=2).values for c in self.convs]
        return self.out(self.drop(torch.cat(pooled, dim=1))).squeeze(1)


class CharCNNDetector(Detector):
    def __init__(self, model: CharCNN, cfg: config.CharCNNConfig, name="char_cnn", device=None):
        self.model, self.cfg, self.name = model, cfg, name
        self.device = device or get_device()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        probs = []
        for i in range(0, len(texts), 256):
            x = encode_bytes(texts[i:i + 256], self.cfg.max_len).to(self.device)
            probs.append(torch.sigmoid(self.model(x)).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def unload(self):
        self.model.cpu()
        torch.cuda.empty_cache()


def train_char_cnn(train_df, val_df, cfg: config.CharCNNConfig = None, seed: int = 42,
                   augment: config.AugmentConfig | None = None, name: str = "char_cnn", verbose=True):
    """Train with early model selection on validation F1. ``augment`` enables method 2."""
    cfg = cfg or config.CharCNNConfig()
    set_seed(seed)
    device = get_device()
    rng = random.Random(seed)
    model = CharCNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    items = list(zip(train_df["text"], train_df["label"]))

    def collate(batch):
        texts = [t if augment is None or rng.random() >= augment.p
                 else perturb_random(t, augment.families, augment.rate_range, rng) for t, _ in batch]
        return encode_bytes(texts, cfg.max_len), torch.tensor([y for _, y in batch], dtype=torch.float)

    loader = DataLoader(items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    best, best_f1, history = None, -1.0, []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        det = CharCNNDetector(model, cfg, name, device)
        f1 = f1_score(val_df["label"], det.predict(list(val_df["text"])), zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": total / len(items), "val_f1": f1})
        if verbose:
            print(f"  epoch {epoch + 1}: loss {total / len(items):.4f}  val F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return CharCNNDetector(model, cfg, name, device), history
