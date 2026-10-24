"""CPU-side scoring shared by summarize.py and baselines_full_document.py.

Turns run records (per-chunk beliefs + chunk texts) into conflict scores for every gate, using the same
gate code as analysis/gate_analysis.py, and loads the paper's original notebook results as a "run".
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import gate_analysis as G  # noqa: E402

PAPER_RHO = {"lexical": 0.50, "semantic": 0.25, "hybrid": 0.35}
NOTEBOOK_RESULTS = {"conflictqa": ROOT / "results/conflictqa/conflicts_air_results_300.json",
                    "strategyqa": ROOT / "results/strategyqa/strategyqa_air_results_300.json"}
_ST = None


def sentence_model():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        import torch
        _ST = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    return _ST


def stopwords(benchmark):
    return G.STOPWORDS | (G.STRATEGYQA_EXTRA if benchmark.startswith("strategyqa") else set())


def relevance(records, benchmark):
    """Lexical, MiniLM-semantic and hybrid relevance of every chunk to the raw question."""
    st, stop = sentence_model(), stopwords(benchmark)
    lex, sem = [], []
    for r in records:
        q = r["raw_question"]
        lex.append([G.lexical_relevance(q, c, stop) for c in r["chunks"]])
        e = st.encode([q] + r["chunks"], normalize_embeddings=True, show_progress_bar=False)
        sem.append(np.clip(e[0] @ e[1:].T, 0.0, 1.0).tolist())
    hyb = [[0.5 * a + 0.5 * b for a, b in zip(l, s)] for l, s in zip(lex, sem)]
    return {"lexical": lex, "semantic": sem, "hybrid": hyb}


def gate_docs(records):
    return [{"beliefs": [np.array(b, dtype=float) for b in r["all_belief_states"]], "confs": r["confidences"]} for r in records]


def prf(y, s, threshold):
    p, r, f, _ = precision_recall_fscore_support(y, (np.asarray(s) > threshold).astype(int), average="binary", zero_division=0)
    return float(p), float(r), float(f)


def score_run(records, benchmark, n_boot=2000, cv_repeats=20):
    """All metrics for one run: confidence-only, ICL, and each relevance gate at the paper floor and with CV."""
    y = np.array([r["label"] for r in records])
    docs, rels = gate_docs(records), relevance(records, benchmark)
    has_icl = all("icl_split_window" in r for r in records)
    icl = np.array([r["icl_split_window"] for r in records], dtype=float) if has_icl else None
    out = {"n": int(len(y)), "methods": {}}

    def add(name, s, extra=None, threshold=0.15):
        m = {"auc": float(roc_auc_score(y, s))}
        m["precision"], m["recall"], m["f1"] = prf(y, s, threshold)
        if icl is not None and name != "ICL split-window":
            ci, dci, p = G.bootstrap(y, np.asarray(s), icl, n_boot)
            m.update(ci=ci, diff_vs_icl_ci=dci, p_vs_icl=p)
        m.update(extra or {})
        out["methods"][name] = m

    if icl is not None:
        add("ICL split-window", icl, {"ci": G.bootstrap(y, icl, icl, n_boot)[0]})
    add("AIR confidence-only", np.array([r["air_confidence_only"] for r in records]))
    for gate, rho in PAPER_RHO.items():
        s, abst, insuff = G.scores_for(docs, rels[gate], rho)
        cv_mean, cv_std, chosen, cv_scores = G.cv_auc(y, docs, rels[gate], cv_repeats)
        extra = {"rho": rho, "abstained_consistent": int(abst[y == 0].sum()), "abstained_conflict": int(abst[y == 1].sum()),
                 "insufficient": int(insuff.sum()), "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
                 "cv_rho_counts": {str(r): chosen.count(r) for r in sorted(set(chosen))}}
        if icl is not None:
            extra["cv_p_vs_icl"] = G.bootstrap(y, cv_scores, icl, n_boot)[2]
        add(f"AIR {gate}", s, extra)
    return out


def load_notebook_run(benchmark):
    """The paper's original seed-42 Gemma results, as run records (chunks re-derived from the rebuilt documents)."""
    import benchmarks as B
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from air import pipeline as P
    saved = json.load(open(NOTEBOOK_RESULTS[benchmark], encoding="utf-8"))
    docs = B.build(benchmark, 42)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    records = []
    for d, meta, s_air, s_icl in zip(docs, saved["metadata"], saved["scores_air"], saved["scores_icl"]):
        chunks = P.chunk_document(d["document"], tok)
        assert len(chunks) == meta["num_chunks"], "rebuilt document does not match the saved results"
        records.append({"label": d["label"], "raw_question": d["raw_question"], "query": d["query"], "chunks": chunks,
                        "confidences": meta["confidences"], "all_belief_states": meta["all_belief_states"],
                        "air_confidence_only": s_air, "icl_split_window": s_icl, "document": d["document"],
                        "doc_a": d["doc_a"], "doc_b": d["doc_b"]})
    config = {"benchmark": benchmark, "seed": 42, "model": "google/gemma-2-2b-it", "rank": 2, "steps": 10,
              "source": "paper notebooks"}
    return {"config": config, "records": records}
