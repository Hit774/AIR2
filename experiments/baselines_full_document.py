"""Experiment 1.2: stronger baselines that see the WHOLE document, on the paper's exact test documents.

  llm-judge     An instruction-tuned LLM reads the entire document in one prompt and is asked whether it
                contains passages that give conflicting answers to the question; score = P(Yes).
                Run it with the paper's base model (gemma-2-2b-it) and with a larger long-context model
                (e.g. Qwen/Qwen2.5-7B-Instruct with --load-4bit, or meta-llama/Llama-3.1-8B-Instruct).
  retrieve-nli  The realistic long-document NLI pipeline: rank chunks by the same hybrid relevance AIR
                uses, take the top-K, run an NLI cross-encoder on every ordered pair, score = max P(contradiction).
  oracle-nli    The paper's reference setting (NLI given the two gold passages), for larger NLI models.

AIR is compared on the same documents with paired bootstrap tests, using the paper's seed-42 results
(or any run file from run_air.py via --air-run).

Usage:
  python experiments/baselines_full_document.py --benchmark conflictqa
  python experiments/baselines_full_document.py --benchmark conflictqa --judge-models google/gemma-2-2b-it Qwen/Qwen2.5-7B-Instruct --load-4bit
Cost on an RTX 4090: minutes for NLI; ~10-30 min per LLM judge for 300 documents.
"""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from air import pipeline as P  # noqa: E402
import scoring as S  # noqa: E402
import run_air as R  # noqa: E402

JUDGE_PROMPT = (
    "Read the document below. Then answer the question with a single word, Yes or No.\n\n"
    "Document:\n{doc}\n\n"
    "Question: Does the document contain two passages that give conflicting answers to the question \"{q}\"?"
)


@torch.no_grad()
def llm_judge(records, model_name, load_4bit):
    tok, model = R.load_model(model_name, load_4bit)
    answer_ids = P.answer_token_ids(tok, variants=True)
    scores = []
    for i, r in enumerate(records):
        msgs = [{"role": "user", "content": JUDGE_PROMPT.format(doc=r["document"], q=r["raw_question"])}]
        if tok.chat_template:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = msgs[0]["content"] + "\nAnswer:"
        x = tok(text, return_tensors="pt", add_special_tokens=not tok.chat_template).input_ids.to(model.device)
        scores.append(float(P.two_way(model(x).logits[:, -1, :], answer_ids)[0]))
        if (i + 1) % 50 == 0:
            print(f"  judge {model_name}: {i + 1}/{len(records)}")
    del model
    torch.cuda.empty_cache()
    return scores


class NLI:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to("cuda").eval()
        self.contra = [k for k, v in self.model.config.id2label.items() if "contradict" in v.lower()][0]

    @torch.no_grad()
    def contradiction(self, premise, hypothesis):
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        return float(torch.softmax(self.model(**enc).logits, dim=-1)[0, self.contra])


def retrieve_nli(records, nli, rels, top_k):
    scores = []
    for r, rel in zip(records, rels):
        top = sorted(range(len(r["chunks"])), key=lambda i: rel[i], reverse=True)[:top_k]
        scores.append(max(nli.contradiction(r["chunks"][a], r["chunks"][b]) for a, b in permutations(top, 2)))
    return scores


def oracle_nli(records, nli):
    return [nli.contradiction(f"Question: {r['query']} Evidence: {r['doc_a']}", r["doc_b"]) for r in records]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="conflictqa", choices=["conflictqa", "strategyqa", "strategyqa_debiased"])
    ap.add_argument("--air-run", default=None, help="run_air.py output to compare against (default: paper results, seed 42)")
    ap.add_argument("--judge-models", nargs="*", default=["google/gemma-2-2b-it"])
    ap.add_argument("--load-4bit", action="store_true", help="load judge models in 4-bit (for 7-8B models)")
    ap.add_argument("--nli-models", nargs="*", default=["cross-encoder/nli-deberta-v3-small", "cross-encoder/nli-deberta-v3-base",
                                                          "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "baselines"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.air_run:
        run = json.load(open(a.air_run, encoding="utf-8"))
        import benchmarks as B
        cfg = run["config"]
        docs = B.build(cfg["benchmark"], cfg["seed"])
        if cfg.get("n_docs") and cfg["n_docs"] < len(docs):
            docs = B.balanced_subset(docs, cfg["n_docs"])
        for rec, d in zip(run["records"], docs):
            rec.update(document=d["document"], doc_a=d["doc_a"], doc_b=d["doc_b"])
    else:
        run = S.load_notebook_run(a.benchmark)
    records = run["records"]
    y = np.array([r["label"] for r in records])
    rels = S.relevance(records, a.benchmark)

    scores = {}
    for m in a.judge_models:
        scores[f"LLM judge ({m.split('/')[-1]})"] = llm_judge(records, m, a.load_4bit)
    for m in a.nli_models:
        nli = NLI(m)
        scores[f"Retrieve-then-NLI top-{a.top_k} ({m.split('/')[-1]})"] = retrieve_nli(records, nli, rels["hybrid"], a.top_k)
        scores[f"Oracle-pair NLI ({m.split('/')[-1]})"] = oracle_nli(records, nli)
        del nli
        torch.cuda.empty_cache()

    # AIR on the same documents
    docs = S.gate_docs(records)
    air = {f"AIR hybrid (rho={S.PAPER_RHO['hybrid']})": S.G.scores_for(docs, rels["hybrid"], S.PAPER_RHO["hybrid"])[0],
           "AIR hybrid (CV rho)": S.G.cv_auc(y, docs, rels["hybrid"], 20)[3]}
    if all("icl_split_window" in r for r in records):
        scores["ICL split-window"] = [r["icl_split_window"] for r in records]

    report = {"benchmark": a.benchmark, "n": int(len(y)), "air_source": run["config"], "baselines": {}, "air": {}}
    print(f"\n=== {a.benchmark} (N={len(y)}) ===")
    for name, s in air.items():
        report["air"][name] = {"auc": float(roc_auc_score(y, s))}
        print(f"{name:<58} AUC {roc_auc_score(y, s):.3f}")
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        entry = {"auc": float(roc_auc_score(y, s)), "scores": s.tolist()}
        for air_name, air_s in air.items():
            ci, dci, p = S.G.bootstrap(y, np.asarray(air_s), s, a.bootstrap)
            entry[f"{air_name} minus baseline"] = {"diff_ci": dci, "p": p}
        report["baselines"][name] = entry
        cv = entry["AIR hybrid (CV rho) minus baseline"]
        print(f"{name:<58} AUC {entry['auc']:.3f} | AIR(CV) - baseline CI [{cv['diff_ci'][0]:+.3f}, {cv['diff_ci'][1]:+.3f}] p={cv['p']:.3f}")
    path = out / f"{a.benchmark}_baselines.json"
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
