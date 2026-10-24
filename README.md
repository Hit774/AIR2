# Atomized Parametric AIR

Code and results for **Atomized Parametric AIR: O(1) Web-Scale Fact-Checking**.

AIR verifies a long document without ever putting it in the prompt. It splits the document into
chunks, self-distills each chunk into a transient rank-2 Micro-LoRA, reads out a Yes/No *belief
state* per chunk, and detects conflict as the maximum Jensen-Shannon divergence between the
beliefs that pass a **relevance-aware confidence gate** (`G_i = C_i * R_i`). At most one chunk is
in context at any time, so inference memory is constant in document length.

## Repository layout

```
notebooks/
  air_conflictqa.ipynb      full pipeline + baselines on ConflictQA (with saved outputs)
  air_strategyqa.ipynb      full pipeline + baselines on StrategyQA (with saved outputs)
results/
  conflictqa/conflicts_air_results_300.json    per-document labels, scores, chunk beliefs
  strategyqa/strategyqa_air_results_300.json
  figures/                  figures produced by the notebooks
air/
  pipeline.py               the AIR pipeline extracted from the notebooks (reused by the scripts)
scripts/
  evaluate_results.py       recompute metrics and bootstrap CIs from the JSON files (no GPU)
analysis/                   CPU-only robustness analyses (no GPU)
  rebuild_benchmarks.py     rebuild the exact 300-document test sets and verify them against results/
  gate_analysis.py          relevance gates, cross-validated rho, bootstrap CIs, tests vs. ICL, abstentions
  nli_sanity.py             label mapping, truncation and argument-order checks for the NLI baseline
experiments/                additional experiments; see experiments/README.md for order and GPU time
  memory_scaling.py         peak memory and time vs. document length, 2k-128k tokens (GPU)
  run_air.py                shared runner: seeds, de-biased StrategyQA, other base models, rank/steps (GPU)
  baselines_full_document.py  LLM judge over the whole document, retrieve-then-NLI, larger NLI models (GPU)
  rank_ablation.py          rank r and distillation-steps grid with the final gate (GPU)
  case_study.py             per-chunk beliefs and gate decisions on a real document (GPU)
  summarize.py              paper-ready tables from all runs: mean ± std over seeds, rank table + LaTeX (CPU)
  benchmarks.py, scoring.py shared benchmark builders and gate scoring
figures_src/                pipeline diagram (Figure 1 of the paper)
```

## Robustness analyses (no GPU)

```bash
pip install numpy scikit-learn transformers sentence-transformers huggingface_hub
python analysis/rebuild_benchmarks.py     # needs Hugging Face access to google/gemma-2-2b-it (tokenizer only)
python analysis/gate_analysis.py
python analysis/nli_sanity.py
```

`rebuild_benchmarks.py` replays the notebooks' random draws and re-chunks every document; all 600
documents match the saved results exactly. `gate_analysis.py` then recomputes every relevance gate
from the stored belief states, selects the relevance floor by 5-fold cross-validation (so no
document is scored with a floor tuned on it), and reports bootstrap confidence intervals and paired
tests against ICL. On ConflictQA the hybrid gate reaches 0.880 AUC with a cross-validated floor
(95% CI 0.830-0.918; AUC gain over ICL +0.050 to +0.152, p < 0.001).

## Memory scaling (GPU)

```bash
python experiments/memory_scaling.py --out results/memory_scaling
python experiments/memory_scaling.py --lengths 8192 --docs-per-length 1 --auc-length 8192 --auc-docs 40
```

Measures the additional peak GPU memory and time of AIR, full-context ICL and split-window ICL on
documents of 2k-128k tokens, and optionally conflict-detection AUC at a longer length.

## Quick check (no GPU)

```bash
pip install numpy scikit-learn
python scripts/evaluate_results.py
```

This reproduces ROC-AUC, precision, recall and F1 (threshold 0.15) for AIR with the
confidence-only gate and for the split-window ICL baseline, plus bootstrap 95% confidence
intervals, from the saved per-document scores.

## Full reproduction (GPU)

Requirements: a CUDA GPU with at least 16 GB of memory (experiments ran on a single
NVIDIA RTX 4090, 24 GB) and a Hugging Face account with access to
[`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it).

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # your Hugging Face token; never commit it
jupyter lab notebooks/
```

Each notebook downloads its dataset from the Hugging Face Hub
([`osunlp/ConflictQA`](https://huggingface.co/datasets/osunlp/ConflictQA), `conflictQA-popQA-gpt4.json`;
[`voidful/StrategyQA`](https://huggingface.co/datasets/voidful/StrategyQA), `strategyqa_train.json`),
builds the 300-document benchmark with seed 42, and runs AIR, the split-window ICL baseline, the
relevance-gated variants and the NLI baseline. A full run takes about 2-2.5 hours per benchmark
on an RTX 4090 (roughly 26 s per ConflictQA document). Run the cells top to bottom; the
`%pip install` cells at the top can be skipped if `requirements.txt` is installed. Notebook cells
that are commented out are earlier versions kept for reference.

## Configuration

| Component | Setting |
|---|---|
| Base model | `google/gemma-2-2b-it`, frozen, bfloat16 |
| Micro-LoRA | r = 2, alpha = 4, no dropout, all 7 projection modules |
| Distillation | 10 AdamW steps, lr 1e-3, grad-norm clip 1.0, KL(teacher with chunk ‖ student without) |
| Chunking | 256 tokens, 64-token overlap |
| Gate | top-K = 3; relevance floor 0.35 (MiniLM hybrid), 0.50 (lexical), 0.25 (MiniLM semantic), 0.25 (Gemma-embedding hybrid) |
| Benchmarks | 300 documents each, 150 conflict / 150 consistent, 20 distractor passages per document |

## Results (ROC-AUC, 300 documents per benchmark)

| Method | ConflictQA | StrategyQA |
|---|---|---|
| Split-window ICL | 0.774 | 0.656 |
| NLI cross-encoder (given the gold passage pair) | 0.349 | 1.000 |
| **AIR** (hybrid gate, MiniLM) | **0.829** | **0.996** |
| AIR, hybrid gate with Gemma embeddings | 0.851 | 0.767 |
| AIR, lexical gate | 0.835 | 0.996 |
| AIR, semantic gate (MiniLM) | 0.705 | 0.948 |
| AIR, confidence-only gate | 0.632 | 0.719 |

Where the numbers come from:

- **Confidence-only AIR and ICL:** stored in `results/*/*.json`; reproduced by `scripts/evaluate_results.py`.
- **Relevance-gated variants:** computed in the notebooks from the stored belief states. The
  "collective benchmark comparison" cell gives the MiniLM rows; the first ablation cell gives the
  Gemma-embedding hybrid row.
- **NLI baseline:** `cross-encoder/nli-deberta-v3-small`. In the ConflictQA notebook the NLI scores
  come from the dedicated NLI cell, which runs before the collective comparison cell and is
  reused by it.

On StrategyQA, consistent documents use a corroborating passage that does not restate the
question, so lexically anchored gates often abstain on them. This inflates the relevance-gated
StrategyQA scores; the paper discusses this in detail.

## Pipeline figure

```bash
pip install matplotlib
python figures_src/architecture.py <output_dir> <preview.png>
```

## Citation

Citation details will be added after the review period.

## License

Code is released under the MIT License (see `LICENSE`). The ConflictQA and StrategyQA datasets
and the Gemma model are subject to their own licenses and terms of use.
