"""Rebuild the exact 300-document ConflictQA and StrategyQA evaluation sets used in the notebooks.

Replays the notebooks' random calls in the same order (seed 42), re-chunks every document with the
Gemma tokenizer, and checks the chunk count of each document against the saved results JSON.
Writes analysis/cache/<benchmark>_chunks.json for the CPU-only gate analyses. No GPU needed.

Usage: python analysis/rebuild_benchmarks.py [--strategyqa-json PATH]
Requires a Hugging Face login with access to google/gemma-2-2b-it (tokenizer only).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "analysis" / "cache"
SEED = 42
CHUNK_SIZE, CHUNK_OVERLAP, NUM_DISTRACTORS, NUM_SAMPLES = 256, 64, 20, 300


def chunk_document(text, tok, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    token_ids = tok.encode(text, add_special_tokens=False)
    stride = chunk_size - overlap
    chunks, start = [], 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(tok.decode(token_ids[start:end], skip_special_tokens=True))
        if end >= len(token_ids):
            break
        start += stride
    return chunks


def construct_long_document(doc_a, doc_b, distractors, num_distractors=NUM_DISTRACTORS):
    chosen = random.sample(distractors, min(num_distractors, len(distractors)))
    mid = len(chosen) // 2
    passages = chosen[:mid // 2] + [doc_a] + chosen[mid // 2:mid] + [doc_b] + chosen[mid:]
    return " ".join(passages)


def filtered_pool(probe_query, pool):
    query_terms = [w for w in probe_query.split() if len(w) > 4]
    filtered = [d for d in pool if not any(t.lower() in d.lower() for t in query_terms[:3])]
    return filtered if len(filtered) >= NUM_DISTRACTORS else pool


def build_conflictqa():
    path = hf_hub_download(repo_id="osunlp/ConflictQA", filename="conflictQA-popQA-gpt4.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        ds = json.load(f) if first == "[" else [json.loads(l) for l in f if l.strip()]

    half, benchmark, cats = NUM_SAMPLES // 2, [], {"cons": 0, "conf": 0}
    for idx, item in enumerate(ds):
        question, ans = item["question"].strip(), item["memory_answer"].strip()
        ev_orig = item["parametric_memory"].strip()
        ev_aligned = item.get("parametric_memory_aligned_evidence", "").strip()
        ev_counter = item["counter_memory"].strip()
        if len(ev_orig) < 40 or len(ev_counter) < 40:
            continue
        claim = f"The answer to the question '{question}' is {ans}."
        if cats["cons"] < half and len(ev_aligned) >= 40 and ev_aligned != ev_orig:
            benchmark.append({"probe_query": claim, "raw_question": question, "doc_a": ev_orig, "doc_b": ev_aligned, "label": 0})
            cats["cons"] += 1
        elif cats["conf"] < half:
            benchmark.append({"probe_query": claim, "raw_question": question, "doc_a": ev_orig, "doc_b": ev_counter, "label": 1})
            cats["conf"] += 1
        if cats["cons"] == half and cats["conf"] == half:
            break
    random.shuffle(benchmark)

    pool = [it.get("parametric_memory", "").strip() for it in list(ds)[-500:]]
    pool = [p for p in pool if len(p) > 50]
    out = []
    for item in benchmark:
        doc = construct_long_document(item["doc_a"], item["doc_b"], filtered_pool(item["probe_query"], pool))
        out.append({**item, "query": f"Is this claim true or false: {item['probe_query']}", "document": doc})
    return out


def build_strategyqa(json_path):
    items = json.load(open(json_path, encoding="utf-8"))
    limit, benchmark = NUM_SAMPLES // 4, []
    cats = {"T_cons": 0, "T_conf": 0, "F_cons": 0, "F_conf": 0}
    a_true = ("Based on established facts, the answer to the question '{q}' is unquestionably True. Multiple reliable "
              "sources have independently verified this claim. The evidence supporting this conclusion is well-documented "
              "and has been confirmed through rigorous analysis by domain experts in the relevant field.")
    a_false = ("Based on established facts, the answer to the question '{q}' is unquestionably False. Multiple reliable "
               "sources have independently verified this conclusion. The evidence against this claim is well-documented and "
               "has been confirmed through rigorous analysis by domain experts in the relevant field.")
    b_true_cons = ("Logical deduction confirms that the premise is accurate and True. Additional supporting evidence has emerged from "
                   "peer-reviewed studies that further corroborate this finding. Independent researchers have reached the same "
                   "conclusion through different methodological approaches and experimental designs.")
    b_false_cons = ("Logical deduction confirms that the premise is accurate and False. Additional supporting evidence has emerged "
                    "from peer-reviewed studies that further corroborate this negative finding. Independent researchers have reached "
                    "the same conclusion through different methodological approaches and experimental designs.")
    b_conf = ("New evidence completely refutes this. The answer to the question '{q}' is absolutely {v}. Recent "
              "investigations have uncovered critical flaws in earlier analyses. Updated research conducted by leading "
              "authorities demonstrates that the original conclusion was based on incomplete or erroneous data.")
    for item in items:
        q, ans = item["question"].strip(), item["answer"]
        if ans is True:
            if cats["T_cons"] < limit:
                benchmark.append({"q": q, "doc_a": a_true.format(q=q), "doc_b": b_true_cons, "label": 0}); cats["T_cons"] += 1
            elif cats["T_conf"] < limit:
                benchmark.append({"q": q, "doc_a": a_true.format(q=q), "doc_b": b_conf.format(q=q, v="False"), "label": 1}); cats["T_conf"] += 1
        elif ans is False:
            if cats["F_cons"] < limit:
                benchmark.append({"q": q, "doc_a": a_false.format(q=q), "doc_b": b_false_cons, "label": 0}); cats["F_cons"] += 1
            elif cats["F_conf"] < limit:
                benchmark.append({"q": q, "doc_a": a_false.format(q=q), "doc_b": b_conf.format(q=q, v="True"), "label": 1}); cats["F_conf"] += 1
        if all(v == limit for v in cats.values()):
            break
    random.shuffle(benchmark)

    pool = []
    for item in items:
        for fact in item.get("facts", []):
            if len(fact) > 5:
                pool.append(f"Research indicates that {fact[0].lower()}{fact[1:]} This has been confirmed by multiple studies and is "
                            f"widely accepted within the academic community. Numerous independent investigations have corroborated "
                            f"this observation through controlled experiments and systematic reviews of the available literature.")
    out = []
    for item in benchmark:
        doc = construct_long_document(item["doc_a"], item["doc_b"], filtered_pool(item["q"], pool))
        out.append({"raw_question": item["q"], "query": f"Is this claim true or false: {item['q']}", "document": doc,
                    "doc_a": item["doc_a"], "doc_b": item["doc_b"], "label": item["label"]})
    return out


def verify_and_save(name, evaluation_set, results_path, tok):
    saved = json.load(open(results_path, encoding="utf-8"))
    rows, mismatches = [], 0
    for i, (item, meta, y) in enumerate(zip(evaluation_set, saved["metadata"], saved["y_true"])):
        chunks = chunk_document(item["document"], tok)
        if len(chunks) != meta["num_chunks"] or item["label"] != y:
            mismatches += 1
        rows.append({**item, "chunks": chunks})
    print(f"{name}: {len(rows)} documents, {mismatches} mismatches in chunk count or label vs saved results")
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(CACHE / f"{name.lower()}_chunks.json", "w", encoding="utf-8"))
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategyqa-json", default=None, help="local strategyqa_train.json (downloaded from voidful/StrategyQA if omitted)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    random.seed(SEED); np.random.seed(SEED)
    cqa = build_conflictqa()
    bad = verify_and_save("ConflictQA", cqa, ROOT / "results/conflictqa/conflicts_air_results_300.json", tok)

    sqa_json = args.strategyqa_json or hf_hub_download(repo_id="voidful/StrategyQA", filename="strategyqa_train.json", repo_type="dataset")
    random.seed(SEED); np.random.seed(SEED)
    sqa = build_strategyqa(sqa_json)
    bad += verify_and_save("StrategyQA", sqa, ROOT / "results/strategyqa/strategyqa_air_results_300.json", tok)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
