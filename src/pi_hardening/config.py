"""Central configuration: paths, model ids, perturbation settings and training hyper-parameters.

Everything that changes between a full run and a quick smoke test lives here, so the
notebooks never hard-code a path or a magic number.

Smoke mode
----------
Set the environment variable ``PI_SMOKE=1`` to replace the real Hugging Face datasets and
models with small synthetic stand-ins. This is only used to check that the whole pipeline
runs end to end; its numbers are meaningless.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root (…/src/pi_hardening/config.py)
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"

SMOKE = os.environ.get("PI_SMOKE", "0") == "1"
LOG_DIR = ROOT / ("smoke/logs" if SMOKE else "logs")
if SMOKE:  # keep smoke artefacts away from real results
    DATA_DIR = ROOT / "smoke" / "data"
    RESULTS_DIR = ROOT / "smoke" / "results"
    PRED_DIR = RESULTS_DIR / "predictions"
    FIG_DIR = ROOT / "smoke" / "figures"
    MODEL_DIR = ROOT / "smoke" / "models"

for _d in (DATA_DIR, RESULTS_DIR, PRED_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------------------
# Datasets (Hugging Face Hub ids) and their licences, for the report's data section
# --------------------------------------------------------------------------------------
DATASETS = {
    "deepset": "deepset/prompt-injections",                          # Apache-2.0, both classes
    "gandalf": "Lakera/gandalf_ignore_instructions",                 # MIT, attacks only
    "benign_chat": "alespalla/chatbot_instruction_prompts",          # Apache-2.0, benign only
    "jackhhao": "jackhhao/jailbreak-classification",                 # Apache-2.0, both classes
    "notinject": "leolee99/NotInject",                               # MIT, benign only
}
# Single-class sources: every row gets this label (they have no label column).
FIXED_LABEL = {"gandalf": 1, "benign_chat": 0}
ONLY_SPLITS = {"benign_chat": "test"}      # load one split only (the full set has 320k rows)
SAMPLE_N = {"benign_chat": 1000}           # random sample, roughly matching the 1,000 Gandalf attacks
EXTRA_TEST_FRAC = 0.15     # share of Gandalf + benign_chat held out for the in-domain test set
VAL_FRAC = 0.15            # share of the training pool used for validation / model selection
OOD_MAX = 400              # cap on the out-of-distribution jailbreak set (balanced), for speed

# --------------------------------------------------------------------------------------
# Obfuscation families (see perturb.py)
# --------------------------------------------------------------------------------------
FAMILIES = ["zero_width", "homoglyph", "diacritics", "leetspeak", "char_spacing", "punct_split"]
PERTURB_RATE = 0.5         # share of eligible words obfuscated in the evaluation sets

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
MODELS = {
    "deberta": "microsoft/deberta-v3-base",
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "piguard": "leolee99/PIGuard",
    "llm_judge": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Datasets each off-the-shelf detector is documented to have been trained on.
# Used to flag (and exclude) evaluation sets where its scores would be inflated.
# ProtectAI v2's model card lists alespalla/chatbot_instruction_prompts and jackhhao among its
# training data and eight unnamed MIT datasets, so Gandalf is treated as seen too (conservative).
SEEN_BY = {
    "protectai": {"benign_chat", "gandalf", "jackhhao"},
    "piguard": set(),   # check the PIGuard paper's training list before reporting
}


def model_id(key: str) -> str:
    """Return the Hub id for a model key, or the local stand-in path in smoke mode."""
    if SMOKE:
        return str(MODEL_DIR / f"smoke_{key}")
    return MODELS[key]


# --------------------------------------------------------------------------------------
# Training hyper-parameters (tuned for one 12 GB GPU, e.g. RTX 4070 Super)
# --------------------------------------------------------------------------------------
# In smoke mode (S) every setting shrinks so the smoke test finishes in minutes on a CPU.
S = SMOKE


@dataclass
class TransformerConfig:
    max_len_train: int = 64 if S else 256   # covers most prompts; long jailbreaks are truncated
    max_len_eval: int = 64 if S else 512    # DeBERTa's usual limit, same as the off-the-shelf detectors
    batch_size: int = 16
    eval_batch_size: int = 16               # small batches: long 512-token inputs otherwise overflow VRAM on Windows
    lr: float = 5e-4 if S else 2e-5        # the tiny smoke model needs a large step to learn at all
    epochs: int = 3 if S else 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = False                      # training precision: fp32 (stable for DeBERTa-v3)
    bf16_eval: bool = True                  # inference only: bf16 autocast is ~2-3x faster, scores barely change
    seeds: tuple = (42,) if S else (13, 42, 77)


@dataclass
class CharCNNConfig:
    max_len: int = 512                      # UTF-8 bytes
    emb_dim: int = 64
    n_filters: int = 128
    kernel_sizes: tuple = (3, 5, 7)
    dropout: float = 0.3
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 2 if S else 15
    seeds: tuple = (42,) if S else (13, 42, 77)


@dataclass
class AugmentConfig:
    p: float = 0.5                          # chance a training example is obfuscated
    rate_range: tuple = (0.2, 1.0)          # share of words obfuscated, sampled uniformly
    families: list = field(default_factory=lambda: list(FAMILIES))


@dataclass
class LLMJudgeConfig:
    max_input_tokens: int = 64 if S else 1024
    batch_size: int = 8
