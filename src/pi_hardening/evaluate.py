"""Evaluation: one shared test bundle, one scoring function, one metrics table for every detector.

The bundle is built once (notebook 01) and saved, so all detectors see identical obfuscated
texts. Each row is one (evaluation set, obfuscation family, sample):

    eval_set  ∈ {test_in, test_ood, notinject}
    family    ∈ {clean, zero_width, homoglyph, diacritics, leetspeak, char_spacing, punct_split}
    text      the (possibly obfuscated) input
    text_norm the same input after the Normalizer (hardening method 1)

Predictions are stored in long format, one row per (run, condition, eval_set, family, uid):
``model, seed, condition ∈ {raw, norm}, score``. Every metric is computed from these files.
"""
from __future__ import annotations

import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .normalize import Normalizer
from .perturb import perturb

KEY = ["model", "condition", "eval_set", "family"]


# --------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------
def build_eval_bundle(splits: dict, families=None, rate=None, seed=config.SEED) -> pd.DataFrame:
    families = families or config.FAMILIES
    rate = config.PERTURB_RATE if rate is None else rate
    norm = Normalizer()
    rows = []
    for eval_set in ("test_in", "test_ood"):
        for r in splits[eval_set].itertuples(index=False):
            for fam in ["clean"] + list(families):
                t = perturb(r.text, fam, rate, seed=(seed, r.uid))
                rows.append({"eval_set": eval_set, "family": fam, "uid": r.uid, "label": r.label,
                             "source": r.source, "n_triggers": np.nan, "text": t})
    for r in splits["notinject"].itertuples(index=False):
        rows.append({"eval_set": "notinject", "family": "clean", "uid": r.uid, "label": 0,
                     "source": "notinject", "n_triggers": r.n_triggers, "text": r.text})
    b = pd.DataFrame(rows)
    b["text_norm"] = norm.batch(b["text"])
    clean = b[b["family"] == "clean"].set_index(["eval_set", "uid"])["text"]
    b["clean_text"] = [clean[(e, u)] for e, u in zip(b["eval_set"], b["uid"])]
    b["changed"] = b["text"] != b["clean_text"]
    return b


def save_bundle(b: pd.DataFrame):
    b.to_parquet(config.DATA_DIR / "eval_bundle.parquet", index=False)


def load_bundle() -> pd.DataFrame:
    p = config.DATA_DIR / "eval_bundle.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run notebook 01 first.")
    return pd.read_parquet(p)


def _word_recovery(a: str, b: str) -> float:
    """Share of the words of ``b`` that appear unchanged, in order, in ``a``."""
    wa, wb = a.split(), b.split()
    if not wb:
        return 1.0
    return sum(m.size for m in SequenceMatcher(None, wa, wb, autojunk=False).get_matching_blocks()) / len(wb)


def normaliser_report(b: pd.DataFrame) -> pd.DataFrame:
    """How well the normaliser undoes each family (in-domain test set).

    * ``restored_exact``: normalised obfuscated text == normalised clean text (strict)
    * ``words_restored``: share of words of the normalised clean text recovered exactly
    * ``words_intact_before``: the same measure *without* normalisation, for comparison
    """
    norm_clean = dict(zip(zip(b.loc[b.family == "clean", "eval_set"], b.loc[b.family == "clean", "uid"]),
                          b.loc[b.family == "clean", "text_norm"]))
    b = b.assign(norm_of_clean=[norm_clean[(e, u)] for e, u in zip(b["eval_set"], b["uid"])])
    rows = []
    for fam, g in b[b.eval_set == "test_in"].groupby("family"):
        rows.append({"family": fam, "n": len(g),
                     "obfuscated_share": g["changed"].mean(),
                     "restored_exact": (g["text_norm"] == g["norm_of_clean"]).mean(),
                     "words_restored": np.mean([_word_recovery(x, y) for x, y in zip(g["text_norm"], g["norm_of_clean"])]),
                     "words_intact_before": np.mean([_word_recovery(x, y) for x, y in zip(g["text"], g["clean_text"])])})
    order = ["clean"] + list(config.FAMILIES)
    out = pd.DataFrame(rows).set_index("family").loc[lambda d: [f for f in order if f in d.index]].reset_index()
    ni = b[b.eval_set == "notinject"]
    out = pd.concat([out, pd.DataFrame([{"family": "NotInject benign: left unchanged", "n": len(ni),
                                         "obfuscated_share": 0.0,
                                         "restored_exact": (ni["text_norm"] == ni["text"]).mean()}])],
                    ignore_index=True)
    return out


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _progress(msg: str):
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        (config.LOG_DIR / "progress.txt").write_text(f"{time.strftime('%H:%M:%S')} {msg}\n", encoding="utf-8")
    except OSError:
        pass


def score_bundle(detector, bundle: pd.DataFrame, model: str, seed: int = 0,
                 conditions=("raw", "norm")) -> pd.DataFrame:
    """Score every bundle row with ``detector``; 'norm' feeds the normalised text instead."""
    out = []
    for cond in conditions:
        col = "text" if cond == "raw" else "text_norm"
        texts = bundle[col].tolist()
        uniq = list(dict.fromkeys(texts))                  # many rows share a text (e.g. after norm)
        scores, t0 = [], time.perf_counter()
        for i in range(0, len(uniq), 256):                 # chunks, so progress can be followed in logs/progress.txt
            scores.extend(detector.predict_proba(uniq[i:i + 256]))
            _progress(f"{model} seed {seed} {cond}: {min(i + 256, len(uniq))}/{len(uniq)} texts, "
                      f"{time.perf_counter() - t0:.0f}s")
        s = dict(zip(uniq, scores))
        out.append(pd.DataFrame({"model": model, "seed": seed, "condition": cond,
                                 "eval_set": bundle["eval_set"].values, "family": bundle["family"].values,
                                 "uid": bundle["uid"].values, "label": bundle["label"].values,
                                 "n_triggers": bundle["n_triggers"].values,
                                 "score": [s[t] for t in texts]}))
    return pd.concat(out, ignore_index=True)


def save_predictions(df: pd.DataFrame, run_name: str):
    df.to_parquet(config.PRED_DIR / f"{run_name}.parquet", index=False)


def predictions_ok(run_name: str) -> bool:
    """True if a prediction file exists, is readable and has only finite scores (used to resume runs)."""
    p = config.PRED_DIR / f"{run_name}.parquet"
    if not p.exists():
        return False
    try:
        s = pd.read_parquet(p, columns=["score"])["score"]
    except Exception:
        return False
    return len(s) > 0 and bool(np.isfinite(s).all())


def load_predictions(pattern: str = "*") -> pd.DataFrame:
    files = sorted(config.PRED_DIR.glob(f"{pattern}.parquet"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {config.PRED_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _metrics(g: pd.DataFrame, threshold=0.5) -> pd.Series:
    y, s = g["label"].values, g["score"].values
    p = s >= threshold
    pos, neg = y == 1, y == 0
    tp, fp = (p & pos).sum(), (p & neg).sum()
    prec = tp / p.sum() if p.sum() else np.nan
    tpr = tp / pos.sum() if pos.sum() else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if pos.sum() and p.sum() and (prec + tpr) else (0.0 if pos.sum() else np.nan)
    return pd.Series({"n": len(y), "n_attack": int(pos.sum()), "n_benign": int(neg.sum()),
                      "tpr": tpr, "fpr": fp / neg.sum() if neg.sum() else np.nan,
                      "precision": prec, "f1": f1,
                      "auroc": roc_auc_score(y, s) if pos.any() and neg.any() else np.nan})


def metrics_table(preds: pd.DataFrame, by=None) -> pd.DataFrame:
    """Metrics per (model, seed, condition, eval_set, family) — then averaged over seeds."""
    by = by or KEY
    per_seed = (preds.groupby(by + ["seed"]).apply(_metrics, include_groups=False).reset_index())
    agg = per_seed.groupby(by).agg(
        **{c: (c, "mean") for c in ["tpr", "fpr", "precision", "f1", "auroc"]},
        **{f"{c}_sd": (c, "std") for c in ["tpr", "fpr", "f1"]},
        n=("n", "first"), n_attack=("n_attack", "first"), n_benign=("n_benign", "first"),
        seeds=("seed", "nunique")).reset_index()
    return agg


def bootstrap_ci(hits: np.ndarray, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of a mean of 0/1 outcomes (e.g. detection rate)."""
    rng = np.random.default_rng(seed)
    hits = np.asarray(hits, float)
    if len(hits) == 0:
        return np.nan, np.nan
    m = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_tpr_diff(preds, a: tuple, b: tuple, eval_set: str, family: str, n_boot=2000, seed=0):
    """Detection-rate difference A − B on the same attack samples, with a paired bootstrap CI.

    ``a`` and ``b`` are (model, condition). Scores are averaged over seeds first.
    """
    def hits(model, cond):
        g = preds[(preds.model == model) & (preds.condition == cond) & (preds.eval_set == eval_set)
                  & (preds.family == family) & (preds.label == 1)]
        return g.groupby("uid")["score"].mean().ge(0.5).astype(float)
    ha, hb = hits(*a), hits(*b)
    common = ha.index.intersection(hb.index)
    d = (ha[common] - hb[common]).values
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), len(d)


# --------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------
def measure_latency(detector, texts, model: str, n: int = 100, warmup: int = 5, normaliser=None) -> dict:
    """Median and p95 milliseconds per single input (batch size 1), optionally including normalisation."""
    texts = list(texts)[:n]
    for t in texts[:warmup]:
        detector.predict_proba([t])
    times = []
    for t in texts:
        t0 = time.perf_counter()
        x = normaliser(t) if normaliser else t
        detector.predict_proba([x])
        times.append((time.perf_counter() - t0) * 1000)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return {"model": model, "with_normaliser": normaliser is not None, "device": device,
            "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, 0.95)), "n": len(times)}


def save_latency(rows: list[dict]):
    p = config.RESULTS_DIR / "latency.csv"
    df = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        old = old[~old.set_index(["model", "with_normaliser"]).index.isin(df.set_index(["model", "with_normaliser"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)


def save_history(history: list[dict], model: str, seed: int):
    p = config.RESULTS_DIR / "training_history.csv"
    df = pd.DataFrame(history).assign(model=model, seed=seed)
    if p.exists():
        old = pd.read_csv(p)
        old = old[~((old.model == model) & (old.seed == seed))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)
