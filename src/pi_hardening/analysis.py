"""Analyses used in the results notebook: robustness matrix, recovery, tokenisation damage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def robustness_matrix(metrics: pd.DataFrame, eval_set: str, metric: str = "tpr", condition: str = "raw",
                      models: list[str] | None = None) -> pd.DataFrame:
    """Rows = models, columns = clean + families, values = metric (default: detection rate)."""
    m = metrics[(metrics.eval_set == eval_set) & (metrics.condition == condition)]
    if models:
        m = m[m.model.isin(models)]
    pivot = m.pivot(index="model", columns="family", values=metric)
    cols = ["clean"] + [f for f in config.FAMILIES if f in pivot.columns]
    pivot = pivot[cols]
    return pivot.loc[models] if models else pivot


def recovery_table(metrics: pd.DataFrame, eval_set: str, base: tuple, variants: dict) -> pd.DataFrame:
    """Share of the obfuscation-induced detection loss recovered by each hardening variant.

    recovery = (TPR_variant(family) − TPR_base(family)) / (TPR_base(clean) − TPR_base(family))
    ``base`` and each variant are (model, condition). Undefined when the base lost nothing.
    """
    def tpr(model, cond, fam):
        r = metrics[(metrics.model == model) & (metrics.condition == cond)
                    & (metrics.eval_set == eval_set) & (metrics.family == fam)]
        return float(r["tpr"].iloc[0]) if len(r) else np.nan
    rows = []
    clean = tpr(*base, "clean")
    for fam in config.FAMILIES:
        b = tpr(*base, fam)
        loss = clean - b
        row = {"family": fam, "base_clean": clean, "base_obfuscated": b, "loss": loss}
        for label, (m, c) in variants.items():
            v = tpr(m, c, fam)
            row[f"{label}_tpr"] = v
            row[f"{label}_recovery"] = (v - b) / loss if loss > 0.02 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def tokenisation_damage(tokenizer, bundle: pd.DataFrame, eval_set: str = "test_in", text_col: str = "text") -> pd.DataFrame:
    """Sub-word tokens per whitespace word, and share of unknown tokens, per family.

    Obfuscation that shatters words into many rare pieces explains why sub-word models fail.
    """
    unk = tokenizer.unk_token_id
    rows = []
    for fam, g in bundle[bundle.eval_set == eval_set].groupby("family"):
        n_tok, n_word, n_unk = 0, 0, 0
        for t in g[text_col]:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            n_tok += len(ids)
            n_word += max(1, len(t.split()))
            n_unk += sum(i == unk for i in ids) if unk is not None else 0
        rows.append({"family": fam, "tokens_per_word": n_tok / n_word, "unk_share": n_unk / max(1, n_tok)})
    df = pd.DataFrame(rows).set_index("family")
    order = ["clean"] + [f for f in config.FAMILIES if f in df.index]
    return df.loc[order]


def show_tokens(tokenizer, text: str) -> list[str]:
    return tokenizer.convert_ids_to_tokens(tokenizer(text, add_special_tokens=False)["input_ids"])


def error_examples(preds: pd.DataFrame, bundle: pd.DataFrame, model: str, condition: str, eval_set: str,
                   family: str, kind: str = "missed", n: int = 5) -> pd.DataFrame:
    """Missed attacks (kind='missed') or false alarms (kind='false_alarm') for qualitative analysis."""
    p = preds[(preds.model == model) & (preds.condition == condition) & (preds.eval_set == eval_set)
              & (preds.family == family)].groupby(["uid", "label"])["score"].mean().reset_index()
    p = p[(p.label == 1) & (p.score < 0.5)] if kind == "missed" else p[(p.label == 0) & (p.score >= 0.5)]
    b = bundle[(bundle.eval_set == eval_set) & (bundle.family == family)][["uid", "text", "clean_text"]]
    return p.merge(b, on="uid").sort_values("score").head(n)
