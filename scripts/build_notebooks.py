"""Regenerate the numbered notebooks from the cell definitions below.

Run from the repo root:  python scripts/build_notebooks.py
(Only needed if you edit this file; the notebooks are committed.)
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = '''import os, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# os.environ["PI_SMOKE"] = "1"   # uncomment for a 5-minute synthetic test of the whole pipeline

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

from pi_hardening import config
print("smoke mode:", config.SMOKE, "| data:", config.DATA_DIR)'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


NOTEBOOKS = {}

# ======================================================================================
NOTEBOOKS["01_data_and_obfuscation.ipynb"] = [
    md("""
# 01 · Data, splits and obfuscated test sets

**Runs on CPU in about a minute.** This notebook:

1. loads the four public datasets and checks their columns,
2. cleans them (English only, no empty rows, no duplicates) and builds leak-free splits,
3. shows each obfuscation family on a benign sentence,
4. builds the shared evaluation bundle: every test sample × {clean + 6 obfuscation families},
   plus its normalised version (hardening method 1),
5. measures how well the normaliser restores obfuscated text, and how much it alters clean text.

Every later notebook reads the files saved here, so all detectors see identical inputs.
"""),
    code(SETUP),
    code('''
from pi_hardening import data, evaluate
from pi_hardening.perturb import perturb, DESCRIPTIONS
from pi_hardening.normalize import Normalizer
TABLES = config.RESULTS_DIR / "tables"; TABLES.mkdir(exist_ok=True)
if config.SMOKE:
    from pi_hardening.smoke import prepare_smoke_models
    prepare_smoke_models()
'''),
    md("## 1. Raw datasets\nCheck that each dataset has a text column and a label column the loader recognises."),
    code('''
raw = pd.concat([data.describe(repo) for repo in config.DATASETS.values()], ignore_index=True)
raw
'''),
    md("How each raw label value is mapped to the binary label. Every row must say *attack* or *benign*; "
       "an `UNMAPPED` value must be added to `POSITIVE` or `NEGATIVE` in `data.py` before continuing."),
    code('''
labels = pd.concat([data.label_values(config.DATASETS[k]) for k in ["deepset", "jackhhao"]], ignore_index=True)
labels.to_csv(TABLES / "label_mapping.csv", index=False)
labels
'''),
    md("""
## 2. Cleaning and splits

* **deepset/prompt-injections** mixes English and German; German rows are removed so that all
  sets are English (the obfuscation families target Latin script).
* **Lakera/gandalf_ignore_instructions** contains only attacks and **alespalla/chatbot_instruction_prompts**
  only benign prompts (a seeded sample of 1,000 from its test split), so they get a fixed label.
* Duplicates are removed inside each dataset, and every later split drops texts already in an
  earlier split (train → val → in-domain test → OOD test), so no test text was seen in training.
* **Train / val** pool deepset's official train split with 85 % of Gandalf + benign chat;
  **in-domain test** is deepset's official test split plus the other 15 %; **OOD test** is the jackhhao
  jailbreak set (balanced, capped); **NotInject** is benign only and measures over-defence.
"""),
    code('''
splits, clean_log, overlap_log = data.build_splits()
data.save_splits(splits)
clean_log.to_csv(TABLES / "cleaning_log.csv", index=False)
clean_log
'''),
    code('''
overlap_log
'''),
    code('''
summary = data.split_summary(splits)
summary.to_csv(TABLES / "split_summary.csv", index=False)
summary
'''),
    md("## 3. The six obfuscation families on a benign sentence"),
    code('''
example = "Please summarise the attached quarterly report and translate it into French."
norm = Normalizer()
demo = pd.DataFrame([{"family": f, "what it does": DESCRIPTIONS[f],
                      "obfuscated": perturb(example, f, 0.5, seed=1),
                      "after normaliser": norm(perturb(example, f, 0.5, seed=1))}
                     for f in config.FAMILIES])
demo["visible as"] = demo["obfuscated"].map(ascii)   # makes invisible / look-alike characters visible
demo.to_csv(TABLES / "obfuscation_examples.csv", index=False)
demo[["family", "what it does", "obfuscated", "visible as", "after normaliser"]]
'''),
    md("""
## 4. Evaluation bundle

Each test sample (both classes) is obfuscated with every family at a rate of
`config.PERTURB_RATE` (share of eligible words changed). Obfuscating benign samples too lets us
check whether a detector reacts to the obfuscation itself rather than to the attack.
"""),
    code('''
bundle = evaluate.build_eval_bundle(splits)
evaluate.save_bundle(bundle)
bundle.groupby(["eval_set", "family"]).agg(n=("uid", "size"), attacks=("label", "sum"),
                                           obfuscated_share=("changed", "mean")).unstack(0)
'''),
    md("""
## 5. How well does the normaliser undo each family?

In-domain test set. The comparison target is the *normalised* clean text, i.e. what a detector
sees when the input was never obfuscated.

* `restored_exact`: the whole text matches (strict: one wrong character counts as a failure).
* `words_restored`: share of words recovered exactly; `words_intact_before` is the same measure
  before normalisation, so the gap between them is what the normaliser wins back.
* The NotInject row shows how often the normaliser leaves a benign prompt unchanged; the rest
  are prompts it alters (e.g. whitespace, "COVID-19" → "COVIDig"), which can cause false alarms.
"""),
    code('''
nrep = evaluate.normaliser_report(bundle)
nrep.to_csv(TABLES / "normaliser_report.csv", index=False)
nrep
'''),
    code('''
for t in ["COVID-19 cases rose in 2024.", "e.g. a well-known Zürich café", "Plan A or B: a b c"]:
    print(f"{t!r:40} -> {norm(t)!r}")
'''),
]

# ======================================================================================
NOTEBOOKS["02_rules_and_char_cnn.ipynb"] = [
    md("""
# 02 · Baselines: keyword rules and the byte-level CNN

**CPU is fine (≈ 10–20 min); a GPU makes it a few minutes.**

* **Rules**: regular expressions for common injection wording (baseline 1).
* **Char-CNN**: a byte-level CNN (course week 4). Because it has no sub-word tokeniser, it is
  hardening method 3: it may be naturally more robust to character tricks.
* **Char-CNN + augmentation**: the same model trained with random obfuscation (method 2).

Every detector is scored on the whole bundle twice: on the raw input and after the normaliser
(method 1). Three seeds per trained model.
"""),
    code(SETUP),
    code('''
from dataclasses import asdict
from pi_hardening import data, evaluate
from pi_hardening.detectors.rules import RuleDetector
from pi_hardening.detectors.char_cnn import train_char_cnn
from pi_hardening.normalize import Normalizer
from pi_hardening.utils import describe_hardware
print(describe_hardware())
splits = data.load_splits()
bundle = evaluate.load_bundle()
norm = Normalizer()
latency_texts = splits["test_in"]["text"].tolist()
latency = []
'''),
    md("## Rules"),
    code('''
rules = RuleDetector()
preds = evaluate.score_bundle(rules, bundle, model="rules")
evaluate.save_predictions(preds, "rules")
latency += [evaluate.measure_latency(rules, latency_texts, "rules"),
            evaluate.measure_latency(rules, latency_texts, "rules", normaliser=norm)]
evaluate.metrics_table(preds).query("eval_set == 'test_in'")[["condition", "family", "tpr", "fpr", "f1"]]
'''),
    md("## Char-CNN (plain and with obfuscation augmentation)"),
    code('''
cfg = config.CharCNNConfig()
aug = config.AugmentConfig()
print(asdict(cfg)); print(asdict(aug))
for model_name, augment in [("char_cnn", None), ("char_cnn_aug", aug)]:
    for seed in cfg.seeds:
        print(f"{model_name}  seed {seed}")
        det, hist = train_char_cnn(splits["train"], splits["val"], cfg, seed=seed, augment=augment, name=model_name)
        evaluate.save_history(hist, model_name, seed)
        evaluate.save_predictions(evaluate.score_bundle(det, bundle, model=model_name, seed=seed),
                                  f"{model_name}__s{seed}")
        if seed == cfg.seeds[0]:
            latency += [evaluate.measure_latency(det, latency_texts, model_name),
                        evaluate.measure_latency(det, latency_texts, model_name, normaliser=norm)]
        det.unload()
evaluate.save_latency(latency)
'''),
    code('''
m = evaluate.metrics_table(evaluate.load_predictions("char_cnn*"))
m.query("eval_set == 'test_in'").pivot_table(index=["model", "condition"], columns="family", values="tpr")
'''),
]

# ======================================================================================
NOTEBOOKS["03_deberta_finetune.ipynb"] = [
    md("""
# 03 · Fine-tuned DeBERTa-v3: plain, augmented, and leave-one-family-out

**Needs a GPU** (≈ 40–60 min on an RTX 4070 Super for all runs).

| Run | What it tests |
|---|---|
| `deberta` × 3 seeds | the main sub-word detector, trained on clean data |
| `deberta_aug` × 3 seeds | hardening method 2: random obfuscation of *both* classes during training |
| `deberta_aug_no_<family>` × 6 | trained with all families except one: does augmentation transfer to an obfuscation it never saw? |

Settings (`config.TransformerConfig`): fp32 (bf16 autocast produced NaN losses with DeBERTa-v3),
batch 16, max length 256 for training and 512 for evaluation, AdamW lr 2e-5, 4 epochs, best epoch
chosen on validation F1. Training stops with an error at the first non-finite loss.
Only the seed-42 checkpoints of `deberta` and `deberta_aug` are kept on disk (≈ 0.7 GB each);
all other runs are evaluated straight after training and then discarded.
"""),
    code(SETUP),
    code('''
from dataclasses import asdict, replace
from pi_hardening import data, evaluate
from pi_hardening.detectors.transformer import finetune
from pi_hardening.normalize import Normalizer
from pi_hardening.utils import describe_hardware
print(describe_hardware())
splits = data.load_splits()
bundle = evaluate.load_bundle()
norm = Normalizer()
cfg = config.TransformerConfig()
print(asdict(cfg))
latency_texts = splits["test_in"]["text"].tolist()
latency = []

def run(model_name, seed, augment=None, keep=False):
    """Train, evaluate and save predictions; returns None if this run already has predictions."""
    print(f"== {model_name}  seed {seed}")
    if evaluate.predictions_ok(f"{model_name}__s{seed}"):
        print("   valid predictions already exist - skipped (delete the file to redo)")
        return None
    save_dir = config.MODEL_DIR / f"{model_name}__s{seed}" if keep else None
    det, hist = finetune(splits["train"], splits["val"], cfg, seed=seed, augment=augment,
                         name=model_name, save_dir=save_dir)
    evaluate.save_history(hist, model_name, seed)
    evaluate.save_predictions(evaluate.score_bundle(det, bundle, model=model_name, seed=seed),
                              f"{model_name}__s{seed}")
    return det
'''),
    md("## Plain and augmented, three seeds each"),
    code('''
aug = config.AugmentConfig()
for model_name, augment in [("deberta", None), ("deberta_aug", aug)]:
    for seed in cfg.seeds:
        det = run(model_name, seed, augment, keep=(seed == 42))
        if det is None:
            continue
        if seed == cfg.seeds[0]:
            evaluate.save_latency([evaluate.measure_latency(det, latency_texts, model_name),
                                   evaluate.measure_latency(det, latency_texts, model_name, normaliser=norm)])
        det.unload()
'''),
    md("## Leave one family out (seed 42)\nThe held-out family is never used for augmentation; notebook 05 compares detection on exactly that family."),
    code('''
for held_out in config.FAMILIES:
    fams = [f for f in config.FAMILIES if f != held_out]
    det = run(f"deberta_aug_no_{held_out}", 42, replace(aug, families=fams))
    if det is not None:
        det.unload()
'''),
    code('''
m = evaluate.metrics_table(evaluate.load_predictions("deberta*"))
m.query("eval_set == 'test_in' and condition == 'raw'").pivot_table(index="model", columns="family", values="tpr")
'''),
]

# ======================================================================================
NOTEBOOKS["04_offtheshelf_and_llm_judge.ipynb"] = [
    md("""
# 04 · Off-the-shelf detectors and an LLM judge

**GPU recommended** (≈ 15–30 min on an RTX 4070 Super; most of it is the LLM judge).

* **ProtectAI deberta-v3-base-prompt-injection-v2**: a widely deployed open-source detector.
  Its model card lists the benign chat prompts and jackhhao among its training data (and may
  include Gandalf), so notebook 05 reports it on deepset and NotInject only.
* **PIGuard**: a DeBERTa detector trained to reduce over-defence. It needs
  `trust_remote_code=True`, which runs code from its Hugging Face repository.
* **LLM judge**: Qwen2.5-1.5B-Instruct, zero-shot, scored from P(Yes) vs P(No) in one forward pass.

Each model is scored raw and after the normaliser, so we can see whether normalisation alone
hardens a detector we cannot retrain.
"""),
    code(SETUP),
    code('''
from pi_hardening import data, evaluate
from pi_hardening.detectors.transformer import HFClassifier
from pi_hardening.detectors.llm_judge import LLMJudge
from pi_hardening.normalize import Normalizer
from pi_hardening.utils import describe_hardware
print(describe_hardware())
splits = data.load_splits()
bundle = evaluate.load_bundle()
norm = Normalizer()
latency_texts = splits["test_in"]["text"].tolist()
latency = []

def score(det, name):
    """Score the bundle (skipped if predictions already exist), then measure latency."""
    if evaluate.predictions_ok(name):
        print(f"{name}: valid predictions already exist - scoring skipped (delete the file to redo)")
    else:
        evaluate.save_predictions(evaluate.score_bundle(det, bundle, model=name), name)
    evaluate.save_latency([evaluate.measure_latency(det, latency_texts, name),
                           evaluate.measure_latency(det, latency_texts, name, normaliser=norm)])
    det.unload()
'''),
    md("## ProtectAI v2"),
    code('''
score(HFClassifier(config.model_id("protectai"), "protectai"), "protectai")
'''),
    md("## PIGuard\nIf loading fails with your `transformers` version, the cell reports it and the analysis simply leaves PIGuard out."),
    code('''
try:
    score(HFClassifier(config.model_id("piguard"), "piguard", trust_remote_code=True), "piguard")
except Exception as e:
    print("PIGuard skipped:", type(e).__name__, e)
'''),
    md("## LLM judge (Qwen2.5-1.5B-Instruct)"),
    code('''
judge = LLMJudge()
print("Yes ids", judge.yes_ids, "No ids", judge.no_ids)
print(judge._prompt("What is the capital of France?"))
score(judge, "llm_judge")
'''),
    code('''
m = evaluate.metrics_table(evaluate.load_predictions())
m.query("eval_set == 'test_in' and model in ['protectai', 'piguard', 'llm_judge']").pivot_table(
    index=["model", "condition"], columns="family", values="tpr")
'''),
]

# ======================================================================================
NOTEBOOKS["05_results_analysis.ipynb"] = [
    md("""
# 05 · Results and analysis

Reads every prediction file and produces the report's tables (`results/tables/`) and figures
(`figures/`). Runs on CPU in a minute. Sections follow the three research questions:

* **RQ1** How much do obfuscations reduce detection, per detector architecture?
* **RQ2** Which hardening method recovers it: normalisation, augmentation, or a character-level model?
* **RQ3** What does hardening cost: false alarms on benign prompts (over-defence) and latency?

All rates use the 0.5 threshold; trained models are averaged over seeds (± is the SD over seeds).
"""),
    code(SETUP),
    code('''
from pi_hardening import data, evaluate, analysis, plots
TABLES = config.RESULTS_DIR / "tables"; TABLES.mkdir(exist_ok=True)
bundle = evaluate.load_bundle()
preds = evaluate.load_predictions()
metrics = evaluate.metrics_table(preds)
metrics.to_csv(TABLES / "all_metrics.csv", index=False)

LABEL = {"rules": "Keyword rules", "char_cnn": "Char-CNN", "char_cnn_aug": "Char-CNN + aug",
         "deberta": "DeBERTa-v3", "deberta_aug": "DeBERTa-v3 + aug", "protectai": "ProtectAI v2",
         "piguard": "PIGuard", "llm_judge": "Qwen2.5-1.5B judge"}
available = set(metrics.model)
MAIN = [m for m in ["rules", "char_cnn", "deberta", "protectai", "piguard", "llm_judge"] if m in available]
print("models found:", sorted(available))
'''),
    md("""
## Leakage check for off-the-shelf detectors
ProtectAI v2 was trained on sources that are part of our in-domain and OOD test sets
(benign chat prompts, jackhhao and possibly Gandalf). For any model in `config.SEEN_BY`, rows from those sources are removed before computing
its metrics, so its numbers below come from deepset and NotInject only.
"""),
    code('''
src = bundle.drop_duplicates(["eval_set", "uid"])[["eval_set", "uid", "source"]]
preds = preds.merge(src, on=["eval_set", "uid"], how="left")
for model, seen in config.SEEN_BY.items():
    drop = (preds.model == model) & preds.source.isin(seen)
    print(f"{model}: dropping {drop.sum()} prediction rows from sources {sorted(seen)}")
    preds = preds[~drop]
metrics = evaluate.metrics_table(preds)
metrics.to_csv(TABLES / "all_metrics_leakage_filtered.csv", index=False)
'''),
    md("## Table 1 · Clean performance"),
    code('''
order = [m for m in ["rules", "char_cnn", "char_cnn_aug", "deberta", "deberta_aug", "protectai", "piguard", "llm_judge"] if m in available]
t1 = (metrics.query("family == 'clean' and condition == 'raw' and eval_set in ['test_in', 'test_ood']")
      .pivot_table(index="model", columns="eval_set", values=["f1", "precision", "tpr", "fpr", "auroc"]))
t1 = t1.reindex(order)
t1.to_csv(TABLES / "t1_clean_performance.csv")
t1
'''),
    md("""
### Per-source breakdown (in-domain test, clean)
Gandalf contributes only attacks and the chatbot prompts only benign examples, so a model could
score well by recognising the *source style*. deepset is the only source with both classes in
the same style: its row is the fairest test of whether a model detects injection intent.
"""),
    code('''
src_map = bundle.drop_duplicates(["eval_set", "uid"])[["eval_set", "uid", "source"]]
p_src = preds.drop(columns="source").merge(src_map, on=["eval_set", "uid"]) if "source" in preds else preds.merge(src_map, on=["eval_set", "uid"])
t1b = (evaluate.metrics_table(p_src.query("eval_set == 'test_in' and family == 'clean' and condition == 'raw'"),
                              by=["model", "source"])
       .pivot_table(index="model", columns="source", values=["tpr", "fpr", "f1"]).reindex(order))
t1b.to_csv(TABLES / "t1b_clean_by_source.csv")
t1b
'''),
    md("""
### Per-source check: is the detector learning the attack or the dataset?
Gandalf contains only attacks and benign_chat only benign prompts, so a detector could score well
by recognising each dataset's style. deepset contains both classes and is therefore the fairest
in-domain check; the OOD jailbreak set and NotInject are further checks. Clean, raw input.
"""),
    code('''
by_src = evaluate.metrics_table(preds.query("family == 'clean' and condition == 'raw' and eval_set == 'test_in'"),
                                by=["model", "source"])
by_src = (by_src[by_src.model.isin(order)]
          .pivot_table(index="model", columns="source", values=["tpr", "fpr", "f1"]).reindex(order))
by_src.to_csv(TABLES / "t1b_clean_by_source.csv")
by_src
'''),
    md("## RQ1 · Detection rate under each obfuscation (raw input, in-domain test)"),
    code('''
rob = analysis.robustness_matrix(metrics, "test_in", "tpr", "raw", MAIN)
rob.to_csv(TABLES / "t2_robustness_tpr_in_domain.csv")
plots.heatmap(rob, "Detection rate on obfuscated attacks (in-domain test, raw input)",
              "fig1_robustness_heatmap", row_labels=[LABEL[m] for m in rob.index],
              note="Share of attacks flagged at threshold 0.5. Trained models: mean of seeds.")
rob
'''),
    code('''
drop = rob.sub(rob["clean"], axis=0).drop(columns="clean")
drop.to_csv(TABLES / "t2b_tpr_drop_vs_clean.csv")
drop
'''),
    md("""
### Does obfuscation alone trigger an alarm?
False-positive rate on **obfuscated benign** prompts. A detector that flags these is reacting to
unusual characters, not to attack intent; that can look like robustness on attacks.
"""),
    code('''
fpr = analysis.robustness_matrix(metrics, "test_in", "fpr", "raw", MAIN)
fpr.to_csv(TABLES / "t3_fpr_obfuscated_benign.csv")
plots.heatmap(fpr, "False alarms on obfuscated benign prompts (in-domain test, raw input)",
              "fig2_fpr_heatmap", row_labels=[LABEL[m] for m in fpr.index])
fpr
'''),
    md("### Out-of-distribution jailbreaks (jackhhao)"),
    code('''
rob_ood = analysis.robustness_matrix(metrics, "test_ood", "tpr", "raw",
                                     [m for m in MAIN if m not in config.SEEN_BY or "jackhhao" not in config.SEEN_BY[m]])
rob_ood.to_csv(TABLES / "t4_robustness_tpr_ood.csv")
plots.heatmap(rob_ood, "Detection rate on obfuscated jailbreaks (OOD test, raw input)", "fig3_robustness_ood",
              row_labels=[LABEL[m] for m in rob_ood.index])
rob_ood
'''),
    md("""
### Why sub-word models break: tokenisation damage
Sub-word tokens per whitespace word on the in-domain test set, for DeBERTa-v3's tokeniser.
"""),
    code('''
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(config.model_id("deberta"))
td = analysis.tokenisation_damage(tok, bundle)
td_norm = analysis.tokenisation_damage(tok, bundle, text_col="text_norm").add_suffix("_after_norm")
td = td.join(td_norm)
td.to_csv(TABLES / "t5_tokenisation_damage.csv")
td
'''),
    code('''
from pi_hardening.perturb import perturb
word = "Please ignore the attached report"
for fam in ["clean"] + config.FAMILIES:
    t = perturb(word, fam, 1.0, seed=3)
    print(f"{fam:13s} {analysis.show_tokens(tok, t)}")
'''),
    md("## RQ2 · Hardening DeBERTa: normaliser, augmentation, both"),
    code('''
variants = {"Normaliser": ("deberta", "norm"), "Augmentation": ("deberta_aug", "raw"),
            "Both": ("deberta_aug", "norm")}
rec = analysis.recovery_table(metrics, "test_in", ("deberta", "raw"), variants).set_index("family")
rec.to_csv(TABLES / "t6_deberta_hardening_recovery.csv")
rec
'''),
    code('''
cols = {"No hardening": ("deberta", "raw"), **variants}
tab = pd.DataFrame({lab: analysis.robustness_matrix(metrics, "test_in", "tpr", c, [m]).loc[m]
                    for lab, (m, c) in cols.items()})
tab.to_csv(TABLES / "t7_deberta_variants_tpr.csv")
plots.grouped_bars(tab, ["clean"] + config.FAMILIES, {k: k for k in cols}, "DeBERTa-v3: detection rate by hardening method",
                   "fig4_deberta_hardening", "Attacks detected")
tab
'''),
    md("### Paired bootstrap: is the improvement over plain DeBERTa reliable?\nDifference in detection rate on the same attack samples, 95 % CI (2,000 resamples)."),
    code('''
rows = []
for fam in config.FAMILIES:
    for lab, (m, c) in variants.items():
        d, lo, hi, n = evaluate.paired_tpr_diff(preds, (m, c), ("deberta", "raw"), "test_in", fam)
        rows.append({"family": fam, "variant": lab, "diff": d, "ci_low": lo, "ci_high": hi, "n_attacks": n})
ci = pd.DataFrame(rows)
ci.to_csv(TABLES / "t8_paired_bootstrap_deberta.csv", index=False)
ci.pivot(index="family", columns="variant", values="diff")
'''),
    md("### The same methods on the Char-CNN"),
    code('''
cnn_cols = {"Char-CNN": ("char_cnn", "raw"), "+ normaliser": ("char_cnn", "norm"),
            "+ augmentation": ("char_cnn_aug", "raw"), "+ both": ("char_cnn_aug", "norm")}
cnn = pd.DataFrame({lab: analysis.robustness_matrix(metrics, "test_in", "tpr", c, [m]).loc[m]
                    for lab, (m, c) in cnn_cols.items() if m in available})
cnn.to_csv(TABLES / "t9_charcnn_variants_tpr.csv")
cnn
'''),
    md("### Normalisation in front of detectors we cannot retrain"),
    code('''
fixed = [m for m in ["rules", "protectai", "piguard", "llm_judge"] if m in available]
fx = pd.concat({LABEL[m]: pd.DataFrame({"raw": analysis.robustness_matrix(metrics, "test_in", "tpr", "raw", [m]).loc[m],
                                        "normalised": analysis.robustness_matrix(metrics, "test_in", "tpr", "norm", [m]).loc[m]})
                for m in fixed}, axis=1)
fx.to_csv(TABLES / "t10_normaliser_on_fixed_detectors.csv")
fx
'''),
    md("### Leave-one-family-out: does augmentation transfer to an unseen obfuscation? (seed 42)"),
    code('''
m42 = evaluate.metrics_table(preds[preds.seed.isin([42, 0])])
def tpr(model, fam, cond="raw"):
    r = m42.query("model == @model and family == @fam and condition == @cond and eval_set == 'test_in'")
    return float(r.tpr.iloc[0]) if len(r) else np.nan
lofo = pd.DataFrame([{"family": f, "No augmentation": tpr("deberta", f),
                      "Augmented without this family": tpr(f"deberta_aug_no_{f}", f),
                      "Augmented with all families": tpr("deberta_aug", f)} for f in config.FAMILIES]).set_index("family")
lofo.to_csv(TABLES / "t11_leave_one_family_out.csv")
plots.grouped_bars(lofo, config.FAMILIES, {k: k for k in lofo.columns},
                   "Augmentation transfer to a held-out obfuscation (DeBERTa-v3, seed 42)",
                   "fig5_leave_one_family_out", "Attacks detected")
lofo
'''),
    md("## RQ3 · Cost of hardening: over-defence on NotInject"),
    code('''
ni = metrics[(metrics.eval_set == "notinject") & metrics.model.isin(list(LABEL))].copy()
ni["label"] = ni.model.map(LABEL) + np.where(ni.condition == "norm", " + normaliser", "")
ni = ni.sort_values("fpr")
ni[["label", "fpr", "fpr_sd", "seeds"]].to_csv(TABLES / "t12_notinject_fpr.csv", index=False)
plots.hbar(ni["label"].tolist()[::-1], ni["fpr"].tolist()[::-1], "Over-defence: benign prompts with trigger words flagged",
           "fig6_notinject_fpr", "False-positive rate on NotInject (lower is better)")
ni[["label", "fpr", "fpr_sd"]]
'''),
    code('''
by_trig = (preds.query("eval_set == 'notinject'")
           .assign(flag=lambda d: d.score >= 0.5)
           .groupby(["model", "condition", "n_triggers"])["flag"].mean().unstack())
by_trig.to_csv(TABLES / "t13_notinject_by_trigger_count.csv")
by_trig
'''),
    md("## RQ3 · Latency (batch size 1)"),
    code('''
lat = pd.read_csv(config.RESULTS_DIR / "latency.csv")
lat["label"] = lat.model.map(LABEL).fillna(lat.model) + np.where(lat.with_normaliser, " + normaliser", "")
lat = lat.sort_values("median_ms")
lat.to_csv(TABLES / "t14_latency.csv", index=False)
plain = lat[~lat.with_normaliser]
plots.hbar(plain["label"].tolist()[::-1], plain["median_ms"].tolist()[::-1], "Median latency per prompt",
           "fig7_latency", "Milliseconds (log scale)", percent=False, log=True)
lat[["label", "device", "median_ms", "p95_ms"]]
'''),
    md("## Error analysis\nAttacks the best hardened configuration still misses, per family (counts), and a few examples."),
    code('''
best = ("deberta_aug", "norm")
best_model, best_cond = best
missed = (preds.query("model == @best_model and condition == @best_cond and eval_set == 'test_in' and label == 1")
          .groupby(["family", "uid"])["score"].mean().lt(0.5).groupby("family").agg(["sum", "size"]))
missed.columns = ["missed", "attacks"]
missed.to_csv(TABLES / "t15_missed_by_best.csv")
missed
'''),
    code('''
ex = analysis.error_examples(preds, bundle, *best, "test_in", "char_spacing", "missed", n=5)
ex.assign(clean_text=ex.clean_text.str.slice(0, 120))[["uid", "score", "clean_text"]]
'''),
    md("## Key numbers for the report"),
    code('''
def g(model, cond, fam, metric="tpr", es="test_in"):
    r = metrics.query("model == @model and condition == @cond and family == @fam and eval_set == @es")
    return float(r[metric].iloc[0]) if len(r) else float("nan")
fams = config.FAMILIES
print(f"DeBERTa clean TPR {g('deberta','raw','clean'):.1%}; mean over families {np.mean([g('deberta','raw',f) for f in fams]):.1%}")
print(f"+ normaliser mean {np.mean([g('deberta','norm',f) for f in fams]):.1%}; + aug mean {np.mean([g('deberta_aug','raw',f) for f in fams]):.1%}; "
      f"both {np.mean([g('deberta_aug','norm',f) for f in fams]):.1%}")
print(f"NotInject FPR: DeBERTa {g('deberta','raw','clean','fpr','notinject'):.1%}, both {g('deberta_aug','norm','clean','fpr','notinject'):.1%}")
'''),
]


def main():
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}}
        nbf.write(nb, NB / name)
        print("wrote", NB / name)


if __name__ == "__main__":
    main()
