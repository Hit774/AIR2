"""Experiment 1.7 (qualitative): run AIR on a real document and show every chunk's belief and gate decision.

Give it a real multi-page web article (saved as plain text) that contains a known contradiction about some
question, e.g. two sections reporting different figures or dates. It prints, for every chunk, the relevance,
confidence, gate score G_i = C_i * R_i, the Yes/No belief, and whether the chunk was admitted by the gate,
followed by the document-level conflict score, plus the split-window ICL score for comparison.
The printed table is ready to adapt into a paper figure or a rebuttal example.

Usage:
  python experiments/case_study.py --text-file article.txt \
      --question "When was the bridge opened?" --claim "The bridge was opened in 1932." [--rho 0.35]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from air import pipeline as P  # noqa: E402
import run_air as R  # noqa: E402
import scoring as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--question", required=True, help="the raw question, used for relevance")
    ap.add_argument("--claim", required=True, help="the claim to verify, used as the query")
    ap.add_argument("--rho", type=float, default=0.35)
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args()

    document = Path(a.text_file).read_text(encoding="utf-8")
    query = f"Is this claim true or false: {a.claim}"
    tok, model = R.load_model(a.model)
    answer_ids = P.answer_token_ids(tok, variants="gemma" not in a.model.lower())
    beliefs, confs, chunks, model = P.air_beliefs(document, query, model, tok, P.lora_config(2), answer_ids)
    rec = {"raw_question": a.question, "chunks": chunks, "confidences": confs,
           "all_belief_states": [(float(b[0]), float(b[1])) for b in beliefs]}
    rel = S.relevance([rec], "case")["hybrid"][0]

    valid = [i for i, r in enumerate(rel) if r >= a.rho]
    admitted = set(sorted(valid, key=lambda i: confs[i] * rel[i], reverse=True)[:P.TOP_K]) if len(valid) >= 2 else set()
    score, abstained, _ = S.G.gated_score([np.array(b) for b in rec["all_belief_states"]], confs, rel, a.rho)

    print(f"\n{'#':>3} {'rel':>5} {'conf':>5} {'G':>5} {'P(Yes)':>7}  gate  chunk")
    for i, ch in enumerate(chunks):
        mark = "IN" if i in admitted else ("--" if rel[i] >= a.rho else "")
        print(f"{i:>3} {rel[i]:5.2f} {confs[i]:5.2f} {confs[i] * rel[i]:5.2f} {rec['all_belief_states'][i][0]:7.2f}  {mark:>4}  "
              f"{ch[:90].replace(chr(10), ' ')}...")
    icl = P.icl_split_window(document, query, model, tok, answer_ids)
    verdict = "INSUFFICIENT EVIDENCE" if abstained else ("CONFLICT" if score > 0.15 else "CONSISTENT")
    print(f"\nAIR conflict score {score:.4f} (ln 2 = 0.693 is the maximum) -> {verdict}")
    print(f"Split-window ICL score {icl:.4f}")
    if a.out:
        json.dump({**rec, "relevance": rel, "admitted": sorted(admitted), "air_score": score, "abstained": abstained,
                   "icl_split_window": icl, "question": a.question, "claim": a.claim},
                  open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
