"""Experiment 1.6: Micro-LoRA rank and distillation-steps ablation with the FINAL gate, on ConflictQA.

Runs run_air.py configurations on the same balanced subset of the paper's seed-42 ConflictQA documents
(default 100), loading the model once. Each configuration is saved and resumable; summarize.py --rank-table
turns them into the paper's ablation table (AUC and F1 for the hybrid gate at rho=0.35 and with CV rho,
adapter size, seconds per chunk).

Default grid: r in {1, 2, 4, 8, 16} at 10 steps (alpha = 2r), and steps in {3, 5, 20} at r = 2.
Cost on an RTX 4090 with 100 documents (about 930 chunks): roughly 45 min per 10-step configuration,
about 5.5 hours for the whole grid. Use --n-docs 60 to halve it.

Usage:
  python experiments/rank_ablation.py
  python experiments/rank_ablation.py --n-docs 60 --ranks 1 2 4 8 --steps-grid 5 20
  python experiments/summarize.py --rank-table
"""
import argparse
from pathlib import Path

import run_air as R

ROOT = Path(__file__).resolve().parent.parent


def run_rank_ablation(n_docs=100, ranks=(1, 2, 4, 8, 16), steps_grid=(3, 5, 20), model=R.DEFAULT_MODEL,
                      out_dir=None, tok_model=None):
    """Run every (rank, steps) configuration on the same balanced ConflictQA subset; returns the loaded model."""
    out_dir = out_dir or str(ROOT / "results" / "rank_ablation")
    configs = [(r, 10) for r in ranks] + [(2, s) for s in steps_grid if s != 10]
    for rank, steps in configs:
        print(f"\n### rank={rank} steps={steps}")
        out, tok_model = R.run("conflictqa", 42, model, rank, steps, n_docs, skip_icl=True, out_dir=out_dir, tok_model=tok_model)
        print(f"saved {out}")
    return tok_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps-grid", type=int, nargs="+", default=[3, 5, 20], help="extra step counts tried at r=2")
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "rank_ablation"))
    a = ap.parse_args()
    run_rank_ablation(a.n_docs, a.ranks, a.steps_grid, a.model, a.out_dir)


if __name__ == "__main__":
    main()
