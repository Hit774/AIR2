"""CPU-only re-analysis of AIR's relevance-aware gates from the saved belief states.

Needs analysis/cache/*_chunks.json from rebuild_benchmarks.py. For each benchmark it
  1. recomputes the lexical / MiniLM-semantic / hybrid gates and checks the AUCs against the notebooks,
  2. selects the relevance floor rho by cross-validation (rho is never chosen on the documents it scores),
  3. reports bootstrap 95% CIs and a paired bootstrap test of AUC(AIR) - AUC(ICL),
  4. breaks down abstentions (fewer than 2 relevant chunks -> score 0) by gold label.
Writes analysis/gate_analysis_results.json.

Usage: python analysis/gate_analysis.py [--repeats 20] [--bootstrap 2000]
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 3
BENCHMARKS = {
    "ConflictQA": ("conflictqa_chunks.json", "results/conflictqa/conflicts_air_results_300.json",
                   {"lexical": 0.835, "semantic": 0.705, "hybrid": 0.829}),
    "StrategyQA": ("strategyqa_chunks.json", "results/strategyqa/strategyqa_air_results_300.json",
                   {"lexical": 0.996, "semantic": 0.948, "hybrid": 0.996}),
}
# Floors used in the paper's main table (collective-comparison cell of the notebooks).
PAPER_RHO = {"lexical": 0.50, "semantic": 0.25, "hybrid": 0.35}
RHO_GRID = np.round(np.arange(0.0, 0.81, 0.05), 2)
STOPWORDS = {
    "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
    "is", "was", "are", "were", "did", "does", "do", "has", "have", "had",
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "from", "by", "with", "about",
}
# The StrategyQA notebook extended the stopword list.
STRATEGYQA_EXTRA = {"can", "could", "would", "should", "will", "more", "than", "most"}


def lexical_relevance(question, chunk, stopwords):
    q = [t.lower() for t in re.findall(r"\w+", question) if t.lower() not in stopwords and len(t) > 2]
    if not q:
        return 1.0
    c = chunk.lower()
    return sum(1 for t in q if t in c) / len(q)


def jsd(p, q, eps=1e-10):
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def gated_score(beliefs, confs, rel, rho):
    """Relevance-aware gate as in the notebooks: returns (score, abstained, insufficient)."""
    valid = [i for i, r in enumerate(rel) if r >= rho]
    if len(valid) < 2:
        return 0.0, True, True
    top = sorted(valid, key=lambda i: confs[i] * rel[i], reverse=True)[:TOP_K]
    score = max(jsd(beliefs[a], beliefs[b]) for a, b in combinations(top, 2))
    return score, False, len(valid) < TOP_K


def scores_for(docs, rels, rho):
    out = [gated_score(d["beliefs"], d["confs"], r, rho) for d, r in zip(docs, rels)]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]))


def cv_auc(y, docs, rels, repeats, seed=0):
    """Nested selection of rho: pick rho on 4 folds by AUC, score the held-out fold, pool, compute AUC.

    Also returns the pooled held-out scores of the first repeat, for the bootstrap test against ICL.
    """
    table = {rho: scores_for(docs, rels, rho)[0] for rho in RHO_GRID}
    aucs, chosen, first = [], [], None
    for r in range(repeats):
        held_out = np.zeros(len(y))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y):
            best = max(RHO_GRID, key=lambda rho: roc_auc_score(y[train], table[rho][train]))
            chosen.append(float(best))
            held_out[test] = table[best][test]
        aucs.append(roc_auc_score(y, held_out))
        if first is None:
            first = held_out.copy()
    return float(np.mean(aucs)), float(np.std(aucs)), chosen, first


def bootstrap(y, s, base, n, seed=0):
    rng = np.random.default_rng(seed)
    a, d = [], []
    while len(a) < n:
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        x = roc_auc_score(y[i], s[i])
        a.append(x)
        d.append(x - roc_auc_score(y[i], base[i]))
    a, d = np.array(a), np.array(d)
    p_value = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return [float(v) for v in np.percentile(a, [2.5, 97.5])], [float(v) for v in np.percentile(d, [2.5, 97.5])], float(min(p_value, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    report = {}

    for name, (cache_file, results_file, notebook_auc) in BENCHMARKS.items():
        rows = json.load(open(ROOT / "analysis/cache" / cache_file, encoding="utf-8"))
        saved = json.load(open(ROOT / results_file, encoding="utf-8"))
        y = np.array(saved["y_true"])
        icl = np.array(saved["scores_icl"], dtype=float)
        conf_only = np.array(saved["scores_air"], dtype=float)
        stop = STOPWORDS | (STRATEGYQA_EXTRA if name == "StrategyQA" else set())
        docs = [{"beliefs": [np.array(b, dtype=float) for b in m["all_belief_states"]], "confs": m["confidences"]}
                for m in saved["metadata"]]

        r_lex, r_sem = [], []
        for row in rows:
            q = row["raw_question"]
            r_lex.append([lexical_relevance(q, c, stop) for c in row["chunks"]])
            e = st.encode([q] + row["chunks"], normalize_embeddings=True, show_progress_bar=False)
            r_sem.append(np.clip(e[0] @ e[1:].T, 0.0, 1.0).tolist())
        r_hyb = [[0.5 * a + 0.5 * b for a, b in zip(l, s)] for l, s in zip(r_lex, r_sem)]
        relevance = {"lexical": r_lex, "semantic": r_sem, "hybrid": r_hyb}

        print(f"\n=== {name} ===")
        print(f"{'method':<22}{'AUC@paper rho':>14}{'notebook':>10}{'CV AUC':>16}{'95% CI':>18}{'AUC-ICL CI':>20}{'p':>8}"
              f"{'abst. cons/conf':>17}")
        rep = {"icl_auc": float(roc_auc_score(y, icl)), "methods": {}}
        for label, s in [("ICL", icl), ("AIR confidence-only", conf_only)]:
            ci, dci, p = bootstrap(y, s, icl, args.bootstrap)
            rep["methods"][label] = {"auc": float(roc_auc_score(y, s)), "ci": ci, "diff_vs_icl_ci": dci, "p_vs_icl": p}
            print(f"{label:<22}{roc_auc_score(y, s):>14.3f}{'':>10}{'':>16}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18}"
                  f"{f'[{dci[0]:+.3f}, {dci[1]:+.3f}]' if label != 'ICL' else '':>20}{(f'{p:.3f}' if label != 'ICL' else ''):>8}")
        for gate, rels in relevance.items():
            s, abst, insuff = scores_for(docs, rels, PAPER_RHO[gate])
            cv_mean, cv_std, chosen, cv_scores = cv_auc(y, docs, rels, args.repeats)
            ci, dci, p = bootstrap(y, s, icl, args.bootstrap)
            cv_ci, cv_dci, cv_p = bootstrap(y, cv_scores, icl, args.bootstrap)
            a_cons, a_conf = int(abst[y == 0].sum()), int(abst[y == 1].sum())
            auc = roc_auc_score(y, s)
            rep["methods"][f"AIR {gate}"] = {
                "rho": PAPER_RHO[gate], "auc": float(auc), "notebook_auc": notebook_auc[gate],
                "cv_auc_mean": cv_mean, "cv_auc_std": cv_std, "cv_rho_chosen": sorted(set(chosen)),
                "cv_rho_counts": {str(r): chosen.count(r) for r in sorted(set(chosen))},
                "cv_ci": cv_ci, "cv_diff_vs_icl_ci": cv_dci, "cv_p_vs_icl": cv_p,
                "ci": ci, "diff_vs_icl_ci": dci, "p_vs_icl": p,
                "abstained_consistent": a_cons, "abstained_conflict": a_conf, "insufficient": int(insuff.sum()),
                "auc_non_abstained": float(roc_auc_score(y[~abst], s[~abst])) if len(set(y[~abst])) > 1 else None,
                "n_non_abstained": int((~abst).sum()),
            }
            print(f"{'AIR ' + gate:<22}{auc:>14.3f}{notebook_auc[gate]:>10.3f}{f'{cv_mean:.3f}±{cv_std:.3f}':>16}"
                  f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18}{f'[{dci[0]:+.3f}, {dci[1]:+.3f}]':>20}{p:>8.3f}{f'{a_cons}/{a_conf}':>17}")
            print(f"{'   (CV rho)':<22}{'':>14}{'':>10}{'':>16}{f'[{cv_ci[0]:.3f}, {cv_ci[1]:.3f}]':>18}"
                  f"{f'[{cv_dci[0]:+.3f}, {cv_dci[1]:+.3f}]':>20}{cv_p:>8.3f}   rho chosen: {rep['methods'][f'AIR {gate}']['cv_rho_counts']}")
        report[name] = rep

    out = ROOT / "analysis" / "gate_analysis_results.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
