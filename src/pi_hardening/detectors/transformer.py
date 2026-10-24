"""Sub-word Transformer detectors (course week 3).

* :class:`HFClassifier` wraps any Hugging Face sequence classifier: our fine-tuned DeBERTa-v3
  and the two off-the-shelf detectors (ProtectAI v2, PIGuard).
* :func:`finetune` fine-tunes DeBERTa-v3-base, optionally with obfuscation augmentation
  (hardening method 2). Augmentation is applied on the fly to *both* classes, so the model
  cannot learn the shortcut "unusual characters ⇒ injection".
"""
from __future__ import annotations

import copy
import math
import random
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .. import config
from ..perturb import perturb_random
from ..utils import bf16_ok, get_device, set_seed
from .base import Detector

POSITIVE_WORDS = ("inject", "jailbreak", "unsafe", "malicious", "attack", "label_1")


def positive_index(id2label: dict) -> int:
    """Find which output index means 'injection' from the model's label names."""
    for idx, name in id2label.items():
        if any(w in str(name).lower() for w in POSITIVE_WORDS):
            return int(idx)
    if len(id2label) == 2:
        return 1
    raise ValueError(f"cannot tell which label is positive: {id2label}")


def _autocast(device, enabled=True):
    if enabled and bf16_ok(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class HFClassifier(Detector):
    def __init__(self, path_or_id: str, name: str, max_len: int = 512, batch_size: int = 16,
                 trust_remote_code: bool = False, model=None, tokenizer=None, device=None,
                 bf16: bool = True):   # bf16 autocast for inference (weights stay fp32)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.max_len, self.batch_size, self.bf16 = name, max_len, batch_size, bf16
        self.device = device or get_device()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code)
        # dtype=float32: transformers v5 otherwise keeps the checkpoint's own precision
        # (DeBERTa-v3 is stored in fp16), which gives NaN when training without loss scaling.
        self.model = model or AutoModelForSequenceClassification.from_pretrained(
            path_or_id, trust_remote_code=trust_remote_code, dtype=torch.float32)
        self.model.to(self.device).eval()
        self.pos = positive_index(self.model.config.id2label)

    @torch.no_grad()
    def predict_proba(self, texts):
        texts = list(texts)
        order = np.argsort([len(t) for t in texts])           # length-sorted batches are faster
        probs = np.empty(len(texts))
        for i in range(0, len(texts), self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self.tokenizer([texts[j] for j in idx], truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            with _autocast(self.device, self.bf16):
                logits = self.model(**enc).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1)[:, self.pos].cpu().numpy()
        if not np.isfinite(probs).all():
            raise FloatingPointError(f"{self.name}: non-finite scores; the model weights or inputs are broken")
        return probs

    def unload(self):
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()


def finetune(train_df, val_df, cfg: config.TransformerConfig = None, seed: int = 42,
             augment: config.AugmentConfig | None = None, name: str = "deberta",
             base_model: str | None = None, save_dir=None, verbose: bool = True):
    """Fine-tune a sequence classifier; select the epoch with the best validation F1."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cfg = cfg or config.TransformerConfig()
    base_model = base_model or config.model_id("deberta")
    set_seed(seed)
    device = get_device()
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label={0: "benign", 1: "injection"},
        label2id={"benign": 0, "injection": 1}, dtype=torch.float32).to(device)  # see HFClassifier
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True     # faster fp32 matmuls on Ampere/Ada, same stability
        torch.backends.cudnn.allow_tf32 = True

    items = list(zip(train_df["text"], train_df["label"]))

    def collate(batch):
        texts = [t if augment is None or rng.random() >= augment.p
                 else perturb_random(t, augment.families, augment.rate_range, rng) for t, _ in batch]
        enc = tok(texts, truncation=True, max_length=cfg.max_len_train, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    loader = DataLoader(items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm", "layernorm"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm", "layernorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    total_steps = cfg.epochs * math.ceil(len(items) / cfg.batch_size)
    warm = int(cfg.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))

    best_state, best_f1, history = None, -1.0, []
    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device, cfg.bf16):
                out = model(**batch)
            if not torch.isfinite(out.loss):
                raise FloatingPointError(f"{name} seed {seed}: loss became {out.loss.item()} at epoch {epoch + 1}. "
                                         "Training stopped; check precision (bf16) and learning rate.")
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += out.loss.item() * len(batch["labels"])
            n += len(batch["labels"])
        det = HFClassifier(None, name, cfg.max_len_eval, cfg.eval_batch_size, model=model, tokenizer=tok,
                           device=device, bf16=cfg.bf16_eval)
        f1 = f1_score(val_df["label"], det.predict(list(val_df["text"])), zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": running / n, "val_f1": f1})
        if verbose:
            print(f"  epoch {epoch + 1}: loss {running / n:.4f}  val F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    return HFClassifier(None, name, cfg.max_len_eval, cfg.eval_batch_size, model=model, tokenizer=tok,
                        device=device, bf16=cfg.bf16_eval), history
