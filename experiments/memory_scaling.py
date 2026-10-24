"""Experiment 1.1: peak GPU memory and wall-clock time vs. document length (AIR vs. ICL).

For each target length (2k ... 128k tokens) it builds ConflictQA documents with the same construction as the
paper (two target passages placed among distractors, padded with more distractors to reach the length) and
measures, on the same GPU and base model:
  * AIR:              per-chunk Micro-LoRA distillation, one chunk in context at a time
  * full-context ICL: the whole document in a single prompt (no truncation)
  * split-window ICL: each half of the document in its own prompt (no truncation)
Peak memory is torch.cuda.max_memory_allocated() minus the memory held by the loaded model, i.e. the
*additional* memory each method needs. Out-of-memory errors are caught and recorded as "OOM".

Optionally (--auc-length), it also measures conflict-detection AUC at one longer length on a balanced subset,
to show accuracy is preserved as documents grow.

Rough cost on an RTX 4090 (about 2.8 s per chunk for AIR):
  scaling sweep, 1 document per length, 2k-128k  -> about 1 hour
  --auc-length 8192 --auc-docs 40                -> about 1.5 hours

Usage:
  python experiments/memory_scaling.py --out results/memory_scaling
  python experiments/memory_scaling.py --lengths 2048 8192 32768 --docs-per-length 2 --auc-length 8192 --auc-docs 40
Requires: HF_TOKEN with access to google/gemma-2-2b-it, a CUDA GPU, and `pip install -r requirements.txt`.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from air import pipeline as P  # noqa: E402

GiB = 1024 ** 3


def load_conflictqa():
    path = hf_hub_download(repo_id="osunlp/ConflictQA", filename="conflictQA-popQA-gpt4.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        return json.load(f) if first == "[" else [json.loads(l) for l in f if l.strip()]


def make_items(ds, n, seed):
    """Balanced (consistent / conflicting) ConflictQA items, built exactly like the paper's benchmark."""
    rng = random.Random(seed)
    cons, conf = [], []
    for item in ds:
        q, ans = item["question"].strip(), item["memory_answer"].strip()
        a, al, c = item["parametric_memory"].strip(), item.get("parametric_memory_aligned_evidence", "").strip(), item["counter_memory"].strip()
        if len(a) < 40 or len(c) < 40:
            continue
        query = f"Is this claim true or false: The answer to the question '{q}' is {ans}."
        if len(al) >= 40 and al != a:
            cons.append({"query": query, "raw_question": q, "doc_a": a, "doc_b": al, "label": 0})
        conf.append({"query": query, "raw_question": q, "doc_a": a, "doc_b": c, "label": 1})
    rng.shuffle(cons)
    rng.shuffle(conf)
    items = cons[: n // 2] + conf[: n - n // 2]
    rng.shuffle(items)
    return items


def build_document(item, pool, target_tokens, tok, rng):
    """Two target passages among distractors (5 before d_a, 5 between, rest after), padded to target_tokens."""
    terms = [w.lower() for w in item["query"].split() if len(w) > 4][:5]
    candidates = [p for p in pool if not any(t in p.lower() for t in terms)]
    rng.shuffle(candidates)
    chosen, total, k = [], len(tok.encode(item["doc_a"] + " " + item["doc_b"], add_special_tokens=False)), 0
    while total < target_tokens:
        p = candidates[k % len(candidates)]
        k += 1
        chosen.append(p)
        total += len(tok.encode(p, add_special_tokens=False)) + 1
    before, between = chosen[:5], chosen[5:10]
    after = chosen[10:]
    return " ".join(before + [item["doc_a"]] + between + [item["doc_b"]] + after)


def measure(fn):
    """Run fn() and return (result, additional peak GiB, seconds) or ("OOM", None, seconds)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        out = fn()
        torch.cuda.synchronize()
        return out, (torch.cuda.max_memory_allocated() - base) / GiB, time.time() - t0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM", None, time.time() - t0


@torch.no_grad()
def full_context_icl(document, query, model, tok, answer_ids):
    x = tok(P.format_prompt(document, query), return_tensors="pt").input_ids.to(model.device)
    return P.two_way(model(x).logits[:, -1, :], answer_ids).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--docs-per-length", type=int, default=1)
    ap.add_argument("--auc-length", type=int, default=None, help="also measure AUC at this length")
    ap.add_argument("--auc-docs", type=int, default=40)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.15, 0.35], help="hybrid-gate floors for the AUC run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "memory_scaling"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", dtype=torch.bfloat16, device_map="cuda").eval()
    answer_ids = P.answer_token_ids(tok)
    cfg = P.lora_config(r=2)
    print(f"GPU: {torch.cuda.get_device_name(0)}, model weights {torch.cuda.memory_allocated() / GiB:.2f} GiB")

    ds = load_conflictqa()
    pool = [it["parametric_memory"].strip() for it in ds[-3000:] if len(it.get("parametric_memory", "").strip()) > 50]
    rng = random.Random(args.seed)

    # ---- memory / time sweep -------------------------------------------------------------------------
    rows = []
    sweep_items = make_items(ds, max(args.docs_per_length, 2), args.seed)
    for L in args.lengths:
        for d in range(args.docs_per_length):
            item = sweep_items[d % len(sweep_items)]
            doc = build_document(item, pool, L, tok, rng)
            n_tok = len(tok.encode(doc, add_special_tokens=False))
            row = {"target_tokens": L, "doc_tokens": n_tok, "doc": d}

            res, mem, sec = measure(lambda: P.air_beliefs(doc, item["query"], model, tok, cfg, answer_ids))
            if isinstance(res, str):
                raise SystemExit("AIR ran out of memory, which should not happen; the LoRA wrapper may still be attached")
            model = res[3]
            row["air_chunks"] = len(res[2])
            row.update(air_peak_gib=mem, air_seconds=sec, air_oom=res == "OOM")

            res, mem, sec = measure(lambda: full_context_icl(doc, item["query"], model, tok, answer_ids))
            row.update(full_icl_peak_gib=mem, full_icl_seconds=sec, full_icl_oom=isinstance(res, str))

            res, mem, sec = measure(lambda: P.icl_split_window(doc, item["query"], model, tok, answer_ids, max_length=10 ** 7))
            row.update(split_icl_peak_gib=mem, split_icl_seconds=sec, split_icl_oom=isinstance(res, str))

            rows.append(row)
            fmt = lambda v: "OOM" if v is None else f"{v:6.2f}"
            print(f"L={n_tok:>7} | AIR +{fmt(row['air_peak_gib'])} GiB {row['air_seconds']:7.1f}s | "
                  f"full ICL +{fmt(row['full_icl_peak_gib'])} GiB | split ICL +{fmt(row['split_icl_peak_gib'])} GiB")
            json.dump(rows, open(out / "memory_scaling.json", "w"), indent=1)

    # ---- optional AUC at a longer length ---------------------------------------------------------------
    if args.auc_length:
        from sentence_transformers import SentenceTransformer
        import gate_analysis as G
        st = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        y, icl_split, conf_only, hybrid = [], [], [], {r: [] for r in args.rho}
        for i, item in enumerate(make_items(ds, args.auc_docs, args.seed + 1)):
            doc = build_document(item, pool, args.auc_length, tok, rng)
            beliefs, confs, chunks, model = P.air_beliefs(doc, item["query"], model, tok, cfg, answer_ids)
            states, idx = P.confidence_gate(beliefs, confs)
            conf_only.append(P.max_cross_jsd(states, [confs[j] for j in idx]))
            lex = [G.lexical_relevance(item["raw_question"], c, G.STOPWORDS) for c in chunks]
            e = st.encode([item["raw_question"]] + chunks, normalize_embeddings=True, show_progress_bar=False)
            rel = [0.5 * a + 0.5 * b for a, b in zip(lex, np.clip(e[0] @ e[1:].T, 0, 1))]
            npb = [b.numpy() for b in beliefs]
            for r in args.rho:
                hybrid[r].append(G.gated_score(npb, confs, rel, r)[0])
            icl_split.append(P.icl_split_window(doc, item["query"], model, tok, answer_ids, max_length=10 ** 7))
            y.append(item["label"])
            print(f"  AUC run {i + 1}/{args.auc_docs} done ({len(chunks)} chunks)")
        y = np.array(y)
        summary = {"length": args.auc_length, "n": len(y),
                   "auc_air_confidence_only": roc_auc_score(y, conf_only),
                   "auc_icl_split_window": roc_auc_score(y, icl_split)}
        for r in args.rho:
            summary[f"auc_air_hybrid_rho_{r}"] = roc_auc_score(y, hybrid[r])
        json.dump({"summary": summary, "y": y.tolist(), "air_hybrid": {str(r): v for r, v in hybrid.items()},
                   "air_confidence_only": conf_only, "icl_split_window": icl_split},
                  open(out / f"auc_at_{args.auc_length}.json", "w"), indent=1)
        print(json.dumps(summary, indent=1))

    # ---- plot ------------------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
        for key, label, style in [("air", "AIR", "-o"), ("full_icl", "Full-context ICL", "-s"), ("split_icl", "Split-window ICL", "-^")]:
            pts = [(r["doc_tokens"], r[f"{key}_peak_gib"], r[f"{key}_seconds"]) for r in rows if r[f"{key}_peak_gib"] is not None]
            if pts:
                xs, ms, ts = zip(*pts)
                ax[0].plot(xs, ms, style, label=label, ms=3)
                ax[1].plot(xs, ts, style, label=label, ms=3)
            oom = [r["doc_tokens"] for r in rows if r[f"{key}_oom"]]
            if oom:
                ax[0].axvline(min(oom), ls=":", lw=0.8, color="grey")
                ax[0].text(min(oom), ax[0].get_ylim()[1] * 0.9, f"{label} OOM", fontsize=6, rotation=90, va="top")
        for a, ylab in zip(ax, ["Additional peak memory (GiB)", "Wall-clock time (s)"]):
            a.set_xscale("log", base=2)
            a.set_xlabel("Document length (tokens)")
            a.set_ylabel(ylab)
            a.grid(alpha=0.3)
        ax[1].set_yscale("log")
        ax[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "memory_scaling.pdf")
        print(f"saved {out / 'memory_scaling.pdf'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
