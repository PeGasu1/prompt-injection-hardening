# Hardening Lightweight Prompt-Injection Detectors Against Obfuscation

Small classifiers placed in front of an LLM application are a cheap first line of defence
against prompt injection, but simple character tricks (invisible characters, look-alike
letters, spacing) can make them miss attacks (Hackett et al., 2025). This project measures how
much each kind of detector loses under six obfuscation families and compares three low-cost
ways to win it back: 

1. **Normalisation**: a rule-based text cleaner applied before any detector (no retraining).
2. **Obfuscation augmentation**: random obfuscation of *both* classes during training.
3. **A character-level model**: a byte-level CNN that has no sub-word tokeniser to break.

It also measures what hardening costs: false alarms on benign prompts that contain trigger
words (NotInject over-defence) and latency.

## Research questions

* **RQ1** How much do character-level obfuscations reduce detection for rule-based, CNN,
  Transformer and LLM detectors?
* **RQ2** Which hardening method recovers detection, including on an obfuscation family not
  seen during training?
* **RQ3** What is the cost in false alarms and latency?

## Detectors

| Detector | Type | Notes |
|---|---|---|
| Keyword rules | regular expressions | baseline |
| Char-CNN (± augmentation) | byte-level CNN, trained here | 3 seeds |
| DeBERTa-v3-base (± augmentation) | fine-tuned here | 3 seeds; plus 6 leave-one-family-out runs |
| ProtectAI deberta-v3-base-prompt-injection-v2 | off-the-shelf | trained on overlapping sources, so evaluated on deepset + NotInject only |
| PIGuard | off-the-shelf | designed to reduce over-defence |
| Qwen2.5-1.5B-Instruct judge | zero-shot LLM | P(Yes) vs P(No), one forward pass |

Every detector is evaluated on the raw input and after the normaliser.

## Data

| Dataset | Licence | Use |
|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 (card metadata also lists CC-BY-4.0) | train / val / in-domain test (English rows only) |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | attacks only: train / val / in-domain test |
| [alespalla/chatbot_instruction_prompts](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts) | Apache-2.0 | benign only (seeded sample of 1,000): train / val / in-domain test |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Apache-2.0 | out-of-distribution test |
| [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject) | MIT | over-defence test (benign prompts with trigger words) |

Data are downloaded from the Hugging Face Hub by notebook 01 and are not stored in this
repository. Splits are de-duplicated so no test text appears in training.

## Obfuscation families

`zero_width`, `homoglyph`, `diacritics`, `leetspeak`, `char_spacing`, `punct_split`
(see `src/pi_hardening/perturb.py`). They are applied only to existing dataset samples of both
classes, to measure robustness; no new attack content is written.

## Repository layout

```
src/pi_hardening/
  config.py        paths, model ids, hyper-parameters (one place for every setting)
  data.py          loading, cleaning, leak-free splits
  perturb.py       the six obfuscation families
  normalize.py     hardening method 1
  evaluate.py      shared test bundle, scoring, metrics, bootstrap CIs, latency
  analysis.py      robustness matrix, recovery, tokenisation damage, error examples
  plots.py         report figures
  smoke.py         synthetic stand-ins for a quick offline test
  detectors/       rules.py, char_cnn.py, transformer.py, llm_judge.py
notebooks/         01 → 05, run in order
results/tables/    CSV tables used in the report
figures/           PNG figures used in the report
```

## Setup (Windows, NVIDIA GPU)

```bat
conda create -n pi python=3.13 -y
conda activate pi
:: PyTorch with CUDA: copy the exact command for Windows / pip / CUDA from https://pytorch.org/get-started/locally/
:: e.g.  pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m ipykernel install --user --name pi --display-name "Python (pi)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Open the notebooks with Jupyter or VS Code and select the **Python (pi)** kernel.

**Fully automatic (Windows + Anaconda):** double-click `scripts\windows\run_everything.bat`.
It sets up the environment, runs notebooks 01–05 (retrying a failed one once, skipping ones
already marked done in `logs\done_XX.txt`) and keeps the PC awake; progress is in
`logs\pipeline.log`.

**Step by step:** double-click `scripts\windows\setup_env.bat`
once, then `scripts\windows\run_01.bat` … `run_05.bat` (or `run_all.bat`). The scripts find
your conda install, create the `pi` environment, execute the notebooks in place (outputs are
saved in the notebooks) and write logs to `logs\`.

**Colab:** the notebooks can also be opened in Google Colab from this repository; run
`pip install -r requirements.txt` first. Notebooks 03 and 04 need a GPU, and the full run is
longer than a free Colab session.

## What is in this repository

The executed notebooks (with their outputs), every prediction file in `results/predictions/`
(scores only, no text), the result tables in `results/tables/` and the report figures in
`figures/`. Raw dataset text and trained model weights are not included: the data are
downloaded again by notebook 01, and the models are retrained by notebooks 02–03. One output
cell in notebook 05 that printed original dataset prompts was cleared in this public copy.

## Running

| Notebook | Device | Time on RTX 4070 Super (approx.) |
|---|---|---|
| `01_data_and_obfuscation` | CPU | 1–2 min |
| `02_rules_and_char_cnn` | CPU or GPU | 5–20 min |
| `03_deberta_finetune` | GPU | 40–60 min |
| `04_offtheshelf_and_llm_judge` | GPU | 15–30 min |
| `05_results_analysis` | CPU | 1–2 min |

**Quick check first:** uncomment `os.environ["PI_SMOKE"] = "1"` in the first cell of each
notebook and run 01 → 05. This uses synthetic data and tiny random models (no download,
a few minutes on CPU) to confirm everything works. Its numbers are meaningless; comment the
line out again for the real run. Smoke outputs go to `smoke/`, never to `results/`.

## Interactive demo

After notebook 03 (which saves `models/deberta__s42` and `models/deberta_aug__s42`), run
`python app.py` or double-click `scripts\windows\run_demo.bat`, then open http://127.0.0.1:7860.
Type a prompt, choose an obfuscation, and compare keyword rules, plain DeBERTa-v3 and the hardened
pipeline (normaliser + DeBERTa-v3 with augmentation). The page also shows hidden/look-alike characters
and the sub-word tokens DeBERTa receives before and after normalisation.

## Reproducibility

Fixed seeds (13, 42, 77) for every trained model; perturbations are deterministic per sample.
All metrics come from the saved prediction files in `results/predictions/`.



## Key references

* Hackett, W., Birch, L., Trawicki, S., Suri, N., & Garraghan, P. (2025). Bypassing LLM
  guardrails: An empirical analysis of evasion attacks against prompt injection and jailbreak
  detection systems. *Proceedings of the First Workshop on LLM Security (LLMSEC)*.
* Li, H., Liu, X., & Xiao, C. (2024). InjecGuard: Benchmarking and mitigating over-defense in
  prompt injection guardrail models. arXiv:2410.22770. (Model later renamed PIGuard.)
* He, P., Gao, J., & Chen, W. (2023). DeBERTaV3. *ICLR*.
* Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text
  classification. *NeurIPS*.
* Unicode Technical Standard #39: Unicode Security Mechanisms (confusables).
