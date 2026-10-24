# Experiments for the rebuttal / camera-ready

Each script targets a criticism reviewers are likely to raise. Everything runs on one 24 GB GPU
(the paper used an RTX 4090). Runs save after every document and resume when re-run with the same command.

## Easiest: one notebook, run unattended

Open `experiments/run_all_experiments.ipynb` on the GPU machine, check the Settings cell, and choose
**Kernel → Restart & Run All**. It asks for the Hugging Face token at the start, then runs every experiment
below in priority order, each behind a crash guard (errors go to `results/run_all_log.txt` and the next
experiment starts). Re-running it resumes where it stopped. It ends with a status report
(`results/run_all_status.txt`) and the summary tables in `results/summary/`. The notebook is generated from
the scripts in this folder; after editing a script, regenerate it with `python experiments/build_notebook.py`.

The individual scripts below do the same things one at a time.

## Setup (on the GPU machine)

```bash
pip install -r requirements.txt            # plus: pip install bitsandbytes   (only for 4-bit 7-8B judges)
export HF_TOKEN=...                        # access to google/gemma-2-2b-it (and any other gated model you use)
huggingface-cli login --token $HF_TOKEN
```

## Run order (highest impact first)

| # | Criticism it answers | Command | GPU time (RTX 4090) |
|---|---|---|---|
| 1 | "O(1) / web-scale is never measured" | `python experiments/memory_scaling.py` | ~1 h |
| 1b | "Does accuracy hold on long documents?" | `python experiments/memory_scaling.py --lengths 8192 --auc-length 8192 --auc-docs 40` | ~1.5 h |
| 2 | "Baselines are weak; nothing reads the whole document" | `python experiments/baselines_full_document.py --benchmark conflictqa` | ~0.5 h |
| 2b | same, with a stronger long-context judge | `python experiments/baselines_full_document.py --benchmark conflictqa --judge-models Qwen/Qwen2.5-7B-Instruct --nli-models --load-4bit` | ~0.5 h |
| 3 | "Single seed" | `python experiments/run_air.py --benchmark conflictqa --seed 43` and `--seed 44` | ~2.2 h each |
| 3b | same for StrategyQA | `python experiments/run_air.py --benchmark strategyqa --seed 43` and `--seed 44` | ~1.5 h each |
| 4 | "StrategyQA leaks the label to the gate" | `python experiments/run_air.py --benchmark strategyqa_debiased` | ~1.5 h |
| 4b | baselines on the fixed benchmark | `python experiments/baselines_full_document.py --air-run results/runs/strategyqa_debiased_seed42_gemma-2-2b-it_r2_s10.json --benchmark strategyqa_debiased` | ~0.3 h |
| 5 | "Rank ablation has no numbers" | `python experiments/rank_ablation.py` (or `--n-docs 60`) | ~5.5 h (~3 h) |
| 6 | "Only one 2B model" | `python experiments/run_air.py --benchmark conflictqa --model Qwen/Qwen2.5-3B-Instruct` | ~3 h |
| 7 | "Synthetic conflicts only" | `python experiments/case_study.py --text-file article.txt --question "..." --claim "..."` | minutes |

Total is roughly 20 GPU-hours. Items 1-4 (about 10 hours) address the most serious concerns.

## Summarize (CPU, any machine)

```bash
python experiments/summarize.py               # all runs + the paper's seed-42 results: mean +/- std over seeds
python experiments/summarize.py --rank-table  # ablation table, also as LaTeX in results/summary/rank_table.tex
```

Outputs go to `results/summary/` (`summary.md`, `summary.json`). Every number uses the same gate code as
the paper (`analysis/gate_analysis.py`): confidence-only, lexical, semantic and hybrid gates at the paper's
floors (0.50 / 0.25 / 0.35) and with cross-validated floors, bootstrap CIs, paired tests against ICL, and
abstentions split by class.

## Where each result goes in the paper

| Result | Paper location |
|---|---|
| `results/memory_scaling/memory_scaling.pdf` + table | new figure in "Memory and Computational Cost"; replaces the qualitative claim |
| `results/baselines/conflictqa_baselines.json` | new rows in Table 2 (text-space baselines block) |
| `summary.md`, mean over seeds | Table 2 as mean ± std; sentence in "Robustness of the ConflictQA result" |
| de-biased StrategyQA run | replaces the StrategyQA columns, or is added next to them; shortens the caveat section |
| `rank_table.tex` | restores the rank ablation table with numbers that match Table 2 |
| second-model run | one sentence + one table row ("AIR generalizes to Qwen2.5-3B") |
| case study output | small qualitative figure or rebuttal example |

## Notes

* `run_air.py` with seed 42 rebuilds the paper's exact documents; LoRA initialization uses the torch RNG,
  so beliefs can differ slightly from the notebooks run to run. Seeds 43/44 change both the documents
  (items and distractors) and the initialization.
* Non-Gemma models read the answer as "Yes"/" Yes" vs. "No"/" No" token groups (`answer_token_ids(variants=True)`);
  Gemma keeps the notebooks' exact reading.
* Keep new result files anonymous until the review period ends (no names, no non-anonymous links).
