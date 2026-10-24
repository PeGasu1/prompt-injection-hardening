"""Loading, cleaning, de-duplicating and splitting the four public datasets.

Output of :func:`build_splits` (all ``pandas.DataFrame`` with columns
``uid, text, label, source`` and, for NotInject, ``n_triggers``):

=============  =====================================================================
``train``      deepset (official train, English) + 85 % of Gandalf attacks and benign chat prompts
``val``        15 % of that pool, stratified by source and label (model selection)
``test_in``    deepset official test (English) + the held-out 15 % of Gandalf and benign chat
``test_ood``   jackhhao jailbreak-classification, balanced, capped at ``OOD_MAX``
``notinject``  NotInject: benign prompts containing 1–3 trigger words (over-defence)
=============  =====================================================================

Label convention everywhere: ``1`` = injection / jailbreak, ``0`` = benign.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

TEXT_CANDIDATES = ("text", "prompt", "Prompt", "input", "query", "sentence", "content")
LABEL_CANDIDATES = ("label", "Label", "labels", "type", "class", "is_injection", "injection")
POSITIVE = {"1", "injection", "jailbreak", "malicious", "prompt_injection", "prompt injection",
            "unsafe", "attack", "true", "yes", "harmful"}
NEGATIVE = {"0", "benign", "safe", "legit", "legitimate", "valid", "false", "no", "normal", "harmless"}


# --------------------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------------------
def _hf_load(repo: str, only_split: str | None = None) -> dict:
    """Return ``{split_name: Dataset}`` from the Hub, or the synthetic stand-in in smoke mode."""
    if config.SMOKE:
        from .smoke import synthetic_dataset
        ds = dict(synthetic_dataset(repo))
    else:
        from datasets import load_dataset
        ds = {only_split: load_dataset(repo, split=only_split)} if only_split else dict(load_dataset(repo))
    return {only_split: ds[only_split]} if only_split else ds


def _key_of(repo: str) -> str:
    return next(k for k, v in config.DATASETS.items() if v == repo)


def describe(repo: str) -> pd.DataFrame:
    """One row per split: number of rows and column names. Run this first on a new machine."""
    ds = _hf_load(repo, config.ONLY_SPLITS.get(_key_of(repo)))
    return pd.DataFrame([{"dataset": repo, "split": name, "rows": len(split),
                          "columns": ", ".join(split.column_names)} for name, split in ds.items()])


def _pick(columns: Iterable[str], candidates: tuple, what: str, repo: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"{repo}: no {what} column among {list(columns)}; add it to the candidates in data.py")


def _to_binary(values, repo: str) -> list[int]:
    norm = [str(v).strip().lower() for v in values]
    unknown = sorted(set(norm) - POSITIVE - NEGATIVE)
    if unknown:
        counts = pd.Series(norm).value_counts().to_dict()
        raise ValueError(f"{repo}: label values {unknown} are not mapped. All values and counts: {counts}. "
                         "Add each to POSITIVE (attack) or NEGATIVE (benign) at the top of data.py.")
    return [1 if v in POSITIVE else 0 for v in norm]


def label_values(repo: str) -> pd.DataFrame:
    """Distinct raw label values and counts per split, to check the mapping before building splits."""
    rows = []
    for split_name, split in _hf_load(repo, config.ONLY_SPLITS.get(_key_of(repo))).items():
        df = split.to_pandas()
        lcol = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        if lcol is None:
            continue
        for v, n in df[lcol].astype(str).value_counts().items():
            k = v.strip().lower()
            rows.append({"dataset": repo, "split": split_name, "column": lcol, "value": v, "count": n,
                         "maps_to": "attack (1)" if k in POSITIVE else "benign (0)" if k in NEGATIVE else "UNMAPPED"})
    return pd.DataFrame(rows)


def load_labelled(key: str, seed: int = config.SEED) -> pd.DataFrame:
    """Load one dataset as ``text, label, source, orig_split``.

    Single-class sources (``config.FIXED_LABEL``) get a constant label; ``config.SAMPLE_N``
    draws a seeded random sample from very large sources.
    """
    repo = config.DATASETS[key]
    frames = []
    for split_name, split in _hf_load(repo, config.ONLY_SPLITS.get(key)).items():
        df = split.to_pandas()
        tcol = _pick(df.columns, TEXT_CANDIDATES, "text", repo)
        if key in config.FIXED_LABEL:
            labels = [config.FIXED_LABEL[key]] * len(df)
        else:
            labels = _to_binary(df[_pick(df.columns, LABEL_CANDIDATES, "label", repo)], repo)
        frames.append(pd.DataFrame({"text": df[tcol].astype(str), "label": labels,
                                    "source": key, "orig_split": split_name}))
    out = pd.concat(frames, ignore_index=True)
    if key in config.SAMPLE_N and len(out) > config.SAMPLE_N[key]:
        out = out.sample(config.SAMPLE_N[key], random_state=seed).reset_index(drop=True)
    return out


def load_notinject() -> pd.DataFrame:
    """NotInject is benign only; keep the number of trigger words (1, 2 or 3) for analysis."""
    repo = config.DATASETS["notinject"]
    frames = []
    for split_name, split in _hf_load(repo).items():
        df = split.to_pandas()
        tcol = _pick(df.columns, TEXT_CANDIDATES, "text", repo)
        if "word_list" in df.columns:
            n_trig = df["word_list"].apply(lambda w: len(list(w)) if w is not None else np.nan)
        else:  # fall back on the split name, e.g. "NotInject_two"
            n = {"one": 1, "two": 2, "three": 3}
            n_trig = next((v for k, v in n.items() if k in split_name.lower()), np.nan)
        frames.append(pd.DataFrame({"text": df[tcol].astype(str), "label": 0, "source": "notinject",
                                    "orig_split": split_name, "n_triggers": n_trig,
                                    "category": df.get("category", pd.Series([None] * len(df)))}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
# German words that are not also common English words ("die", "was", "den", "war", "man" are excluded)
_GERMAN_WORDS = {"und", "der", "das", "ist", "nicht", "ich", "sie", "mit", "auf", "für", "ein", "eine",
                 "wie", "wer", "warum", "worum", "welche", "zu", "dem", "von", "bitte", "werden", "auch",
                 "wird", "sind", "oder", "aber", "noch", "nach", "über", "kann", "du", "mir", "mich",
                 "jetzt", "durch", "aus", "nur", "sehr", "nun", "alles", "vergiss", "schreibe", "sag",
                 "sage", "gibt", "geht", "sich", "zum", "zur", "bei", "heute", "schnell", "gut", "neue"}


def is_probably_german(text: str) -> bool:
    """Cheap language filter for deepset/prompt-injections, which mixes English and German.

    Flags a text as German when it contains umlauts/ß or when at least 12 % of its words are
    common German words. Good enough for a binary EN/DE split; removed rows are saved for checking.
    """
    if re.search(r"[äöüÄÖÜß]", text):
        return True
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    return sum(w in _GERMAN_WORDS for w in words) / len(words) >= 0.12


def dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clean(df: pd.DataFrame, english_only: bool, log: list, name: str) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    n_empty = n0 - len(df)
    n_de = 0
    if english_only:
        mask = df["text"].apply(is_probably_german)
        n_de = int(mask.sum())
        if n_de:  # keep the removed rows so the language filter can be checked by eye
            out = config.RESULTS_DIR / "tables"
            out.mkdir(exist_ok=True)
            df.loc[mask, ["text", "label"]].to_csv(out / f"removed_non_english_{name}.csv", index=False,
                                                   encoding="utf-8-sig")
        df = df[~mask]
    df = df.assign(_k=df["text"].map(dedup_key))
    n1 = len(df)
    df = df.drop_duplicates("_k")
    log.append({"dataset": name, "raw": n0, "empty": n_empty, "non_english": n_de,
                "duplicates": n1 - len(df), "kept": len(df)})
    return df


def _drop_overlap(df: pd.DataFrame, seen: set, log: list, name: str) -> pd.DataFrame:
    mask = df["_k"].isin(seen)
    log.append({"split": name, "removed_overlap_with_earlier_splits": int(mask.sum())})
    return df[~mask]


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def build_splits(seed: int = config.SEED) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Build all evaluation splits. Returns ``(splits, cleaning_log, overlap_log)``."""
    clean_log, overlap_log = [], []

    deepset = _clean(load_labelled("deepset", seed), True, clean_log, "deepset")
    gandalf = _clean(load_labelled("gandalf", seed), True, clean_log, "gandalf")
    benign = _clean(load_labelled("benign_chat", seed), True, clean_log, "benign_chat")
    jackhhao = _clean(load_labelled("jackhhao", seed), True, clean_log, "jackhhao")
    notinject = _clean(load_notinject(), False, clean_log, "notinject")

    # Gandalf + benign chat prompts: hold out a stratified share for the in-domain test set
    extra = pd.concat([gandalf, benign], ignore_index=True).drop_duplicates("_k")
    e_pool, e_test = train_test_split(extra, test_size=config.EXTRA_TEST_FRAC,
                                      stratify=extra["source"], random_state=seed)
    d_train = deepset[deepset["orig_split"] != "test"]
    d_test = deepset[deepset["orig_split"] == "test"]

    pool = pd.concat([d_train, e_pool], ignore_index=True).drop_duplicates("_k")
    strat = pool["source"] + "_" + pool["label"].astype(str)
    train, val = train_test_split(pool, test_size=config.VAL_FRAC, stratify=strat, random_state=seed)

    seen = set(train["_k"])
    val = _drop_overlap(val, seen, overlap_log, "val")
    seen |= set(val["_k"])
    test_in = _drop_overlap(pd.concat([d_test, e_test], ignore_index=True).drop_duplicates("_k"),
                            seen, overlap_log, "test_in")
    seen |= set(test_in["_k"])
    ood = _drop_overlap(jackhhao, seen, overlap_log, "test_ood")
    ood = _balanced_cap(ood, config.OOD_MAX, seed)
    seen |= set(ood["_k"])
    notinject = _drop_overlap(notinject, set(train["_k"]), overlap_log, "notinject")

    splits = {"train": train, "val": val, "test_in": test_in, "test_ood": ood, "notinject": notinject}
    for name, df in splits.items():
        df = df.drop(columns=["_k"]).reset_index(drop=True)
        df.insert(0, "uid", [f"{name}-{i:05d}" for i in range(len(df))])
        splits[name] = df
    return splits, pd.DataFrame(clean_log), pd.DataFrame(overlap_log)


def _balanced_cap(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    per_class = cap // 2
    parts = [g.sample(min(len(g), per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts, ignore_index=True)


def split_summary(splits: dict) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        for src, g in df.groupby("source"):
            words = g["text"].str.split().str.len()
            rows.append({"split": name, "source": src, "n": len(g), "injection": int(g["label"].sum()),
                         "benign": int((g["label"] == 0).sum()), "median_words": float(words.median()),
                         "p95_words": float(words.quantile(0.95))})
    return pd.DataFrame(rows)


def save_splits(splits: dict) -> None:
    for name, df in splits.items():
        df.to_parquet(config.DATA_DIR / f"{name}.parquet", index=False)


def load_splits() -> dict:
    names = ["train", "val", "test_in", "test_ood", "notinject"]
    missing = [n for n in names if not (config.DATA_DIR / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {config.DATA_DIR}; run notebook 01 first.")
    return {n: pd.read_parquet(config.DATA_DIR / f"{n}.parquet") for n in names}
