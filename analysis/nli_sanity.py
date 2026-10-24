"""Sanity checks for the NLI cross-encoder baseline (CPU).

1. Prints the model's label mapping and runs obvious entailment / contradiction pairs.
2. Measures how often premise + hypothesis exceed the 512-token limit on the ConflictQA pairs.
3. Re-scores the 300 ConflictQA pairs in both argument orders and with the full 3-way output,
   to show whether the below-chance AUC comes from a bug or from the data.

Usage: python analysis/nli_sanity.py   (needs analysis/cache/conflictqa_chunks.json)
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
NAME = "cross-encoder/nli-deberta-v3-small"

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()
id2label = model.config.id2label
contra = [k for k, v in id2label.items() if "contradict" in v.lower()][0]
entail = [k for k, v in id2label.items() if "entail" in v.lower()][0]
print("id2label:", id2label, "| contradiction index:", contra)


@torch.no_grad()
def probs(premise, hypothesis):
    enc = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    return torch.softmax(model(**enc).logits, dim=-1).squeeze(0).numpy()


checks = [
    ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France.", "entailment"),
    ("The Eiffel Tower is in Paris.", "The Eiffel Tower is in Berlin.", "contradiction"),
    ("Marie Curie was born in Warsaw.", "Marie Curie was born in Lyon.", "contradiction"),
]
for p, h, expected in checks:
    pr = probs(p, h)
    print(f"  expected {expected:<13} -> predicted {id2label[int(pr.argmax())]:<13} {np.round(pr, 3)}")

rows = json.load(open(ROOT / "analysis/cache/conflictqa_chunks.json", encoding="utf-8"))
y = np.array([r["label"] for r in rows])
lengths, s_orig, s_swap, s_ent = [], [], [], []
for r in rows:
    premise = f"Question: {r['query']} Evidence: {r['doc_a']}"
    lengths.append(len(tok(premise, r["doc_b"])["input_ids"]))
    p1 = probs(premise, r["doc_b"])
    p2 = probs(r["doc_b"], premise)
    s_orig.append(p1[contra]); s_swap.append(p2[contra]); s_ent.append(p1[entail])
lengths = np.array(lengths)
print(f"\nConflictQA pairs: median {int(np.median(lengths))} tokens, {(lengths > 512).mean():.0%} exceed 512 and are truncated")
print(f"AUC, P(contradiction), notebook order:      {roc_auc_score(y, s_orig):.3f}")
print(f"AUC, P(contradiction), swapped order:       {roc_auc_score(y, s_swap):.3f}")
print(f"AUC, 1 - P(entailment), notebook order:     {roc_auc_score(y, 1 - np.array(s_ent)):.3f}")
print(f"mean P(contradiction): consistent {np.mean(np.array(s_orig)[y == 0]):.3f}, conflict {np.mean(np.array(s_orig)[y == 1]):.3f}")
