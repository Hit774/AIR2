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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps-grid", type=int, nargs="+", default=[3, 5, 20], help="extra step counts tried at r=2")
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "rank_ablation"))
    a = ap.parse_args()

    configs = [(r, 10) for r in a.ranks] + [(2, s) for s in a.steps_grid if s != 10]
    loaded = None
    for rank, steps in configs:
        print(f"\n### rank={rank} steps={steps}")
        out, loaded = R.run("conflictqa", 42, a.model, rank, steps, a.n_docs, skip_icl=True, out_dir=a.out_dir, tok_model=loaded)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
