"""Benchmark builders shared by the experiment scripts.

conflictqa / strategyqa reuse the notebooks' construction (analysis/rebuild_benchmarks.py), so seed 42
reproduces the paper's 300 documents exactly; other seeds draw different items and distractors.

strategyqa_debiased fixes the leak described in the paper: in the original StrategyQA benchmark only
conflicting documents restate the question in the second target passage, so a lexical relevance gate can
tell the classes apart without any reasoning. Here both target passages restate the question in BOTH
conditions, and the second passage uses the same template in both conditions - only the answer word
differs. The first passage also carries the item's own StrategyQA decomposition facts as evidence.
"""
import random
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import rebuild_benchmarks as RB  # noqa: E402

NAMES = ["conflictqa", "strategyqa", "strategyqa_debiased"]

# Several phrasings per role so no fixed wording correlates with the label.
DEBIASED_A = [
    "{facts} Based on these facts, the answer to the question '{q}' is {v}.",
    "{facts} Taken together, this evidence indicates that the answer to the question '{q}' is {v}.",
    "{facts} From this, it follows that the answer to the question '{q}' is {v}.",
]
DEBIASED_B = [
    "A separate analysis of the question '{q}' concludes that the answer is {v}. This assessment draws on independent sources and a review of the relevant literature.",
    "An independent review of the question '{q}' reports that the answer is {v}. The review compiles evidence from several published references.",
    "According to another source that examined the question '{q}', the answer is {v}. That source summarizes findings from multiple studies.",
]


def strategyqa_json():
    local = ROOT.parent.parent / "strategyQA" / "strategyqa_train (1).json"
    if local.exists():
        return str(local)
    return hf_hub_download(repo_id="voidful/StrategyQA", filename="strategyqa_train.json", repo_type="dataset")


def build_strategyqa_debiased(json_path, n=RB.NUM_SAMPLES):
    import json
    items = json.load(open(json_path, encoding="utf-8"))
    limit, bench = n // 4, []
    cats = {"T_cons": 0, "T_conf": 0, "F_cons": 0, "F_conf": 0}
    for item in items:
        q, ans, facts = item["question"].strip(), item["answer"], item.get("facts", [])
        if not isinstance(ans, bool) or not facts:
            continue
        gold, other = ("True", "False") if ans else ("False", "True")
        key = "T" if ans else "F"
        if cats[f"{key}_cons"] < limit:
            label, v_b = 0, gold
            cats[f"{key}_cons"] += 1
        elif cats[f"{key}_conf"] < limit:
            label, v_b = 1, other
            cats[f"{key}_conf"] += 1
        else:
            continue
        doc_a = random.choice(DEBIASED_A).format(facts=" ".join(facts), q=q, v=gold)
        doc_b = random.choice(DEBIASED_B).format(q=q, v=v_b)
        bench.append({"q": q, "doc_a": doc_a, "doc_b": doc_b, "label": label})
        if all(v == limit for v in cats.values()):
            break
    random.shuffle(bench)

    pool = []
    for item in items:
        for fact in item.get("facts", []):
            if len(fact) > 5:
                pool.append(f"Research indicates that {fact[0].lower()}{fact[1:]} This has been confirmed by multiple studies and is "
                            f"widely accepted within the academic community. Numerous independent investigations have corroborated "
                            f"this observation through controlled experiments and systematic reviews of the available literature.")
    out = []
    for item in bench:
        doc = RB.construct_long_document(item["doc_a"], item["doc_b"], RB.filtered_pool(item["q"], pool))
        out.append({"raw_question": item["q"], "query": f"Is this claim true or false: {item['q']}", "document": doc,
                    "doc_a": item["doc_a"], "doc_b": item["doc_b"], "label": item["label"]})
    return out


def build(name, seed=42):
    """Return a list of {query, raw_question, document, doc_a, doc_b, label} for the named benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    if name == "conflictqa":
        return RB.build_conflictqa()
    if name == "strategyqa":
        return RB.build_strategyqa(strategyqa_json())
    if name == "strategyqa_debiased":
        return build_strategyqa_debiased(strategyqa_json())
    raise ValueError(f"unknown benchmark {name}; choose from {NAMES}")


def balanced_subset(docs, n):
    """First n/2 consistent and n/2 conflicting documents, in benchmark order."""
    cons = [d for d in docs if d["label"] == 0][: n // 2]
    conf = [d for d in docs if d["label"] == 1][: n - n // 2]
    keep = {id(d) for d in cons + conf}
    return [d for d in docs if id(d) in keep]
