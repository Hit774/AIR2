"""Recompute conflict-detection metrics from the saved per-document results.

The JSON files written by the notebooks store, for every document, the gold label,
the AIR conflict score with the confidence-only gate, the split-window ICL score,
and per-chunk metadata (belief states and confidences). This script needs no GPU.

Usage:
    python scripts/evaluate_results.py
    python scripts/evaluate_results.py --bootstrap 2000 --threshold 0.15
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = {
    "ConflictQA": ROOT / "results" / "conflictqa" / "conflicts_air_results_300.json",
    "StrategyQA": ROOT / "results" / "strategyqa" / "strategyqa_air_results_300.json",
}


def prf(y, scores, threshold):
    p, r, f1, _ = precision_recall_fscore_support(
        y, (scores > threshold).astype(int), average="binary", zero_division=0)
    return p, r, f1


def bootstrap_auc(y, a, b, n_boot, seed=0):
    """95% percentile CIs for AUC(a), AUC(b) and AUC(a) - AUC(b) over resampled documents."""
    rng = np.random.default_rng(seed)
    auc_a, auc_b = [], []
    while len(auc_a) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        auc_a.append(roc_auc_score(y[idx], a[idx]))
        auc_b.append(roc_auc_score(y[idx], b[idx]))
    auc_a, auc_b = np.array(auc_a), np.array(auc_b)
    ci = lambda v: np.percentile(v, [2.5, 97.5])
    return ci(auc_a), ci(auc_b), ci(auc_a - auc_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15, help="decision threshold on the JSD score")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    args = ap.parse_args()

    for name, path in RESULTS.items():
        d = json.load(open(path, encoding="utf-8"))
        y = np.array(d["y_true"])
        air = np.array(d["scores_air"], dtype=np.float64)
        icl = np.array(d["scores_icl"], dtype=np.float64)
        n_chunks = [m["num_chunks"] for m in d["metadata"]]

        print(f"\n=== {name} (N={len(y)}, {int(y.sum())} conflict / {int((1 - y).sum())} consistent) ===")
        print(f"chunks per document: {min(n_chunks)}-{max(n_chunks)} (mean {np.mean(n_chunks):.1f})")
        print(f"{'method':<28} {'AUC':>6} {'P':>6} {'R':>6} {'F1':>6} {'mean conflict':>14} {'mean consistent':>16}")
        for label, s in [("AIR, confidence-only gate", air), ("Split-window ICL", icl)]:
            p, r, f1 = prf(y, s, args.threshold)
            print(f"{label:<28} {roc_auc_score(y, s):6.3f} {p:6.3f} {r:6.3f} {f1:6.3f} "
                  f"{s[y == 1].mean():14.4f} {s[y == 0].mean():16.4f}")

        if args.bootstrap:
            ci_air, ci_icl, ci_diff = bootstrap_auc(y, air, icl, args.bootstrap)
            print(f"bootstrap 95% CI  AIR AUC [{ci_air[0]:.3f}, {ci_air[1]:.3f}]  "
                  f"ICL AUC [{ci_icl[0]:.3f}, {ci_icl[1]:.3f}]  "
                  f"AIR-ICL [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}]")

    print("\nRelevance-gated AIR variants and the NLI baseline are computed inside the notebooks;"
          " see the README for where each number comes from.")


if __name__ == "__main__":
    main()
