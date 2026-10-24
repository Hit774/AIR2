"""Shared GPU runner: AIR per-chunk beliefs + split-window ICL on one benchmark configuration.

One run = one (benchmark, seed, base model, rank, steps) setting. Every document's chunk texts, beliefs,
confidences and ICL score are saved, so all gates (confidence-only, lexical, semantic, hybrid, any rho,
cross-validated rho) are computed afterwards on CPU by summarize.py. Results are written after every
document; re-running the same command resumes where it stopped.

Experiments it covers (see experiments/README.md):
  seeds              --benchmark conflictqa --seed 43        (and 44, and the same for strategyqa)
  de-biased SQA      --benchmark strategyqa_debiased
  second base model  --model Qwen/Qwen2.5-3B-Instruct
  rank / steps       --rank 4 --steps 10 --n-docs 100 --skip-icl   (driven by rank_ablation.py)

Usage:
  python experiments/run_air.py --benchmark conflictqa --seed 43
  python experiments/run_air.py --benchmark strategyqa_debiased
  python experiments/run_air.py --benchmark conflictqa --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from air import pipeline as P  # noqa: E402
import benchmarks as B  # noqa: E402

DEFAULT_MODEL = "google/gemma-2-2b-it"


def run_name(benchmark, seed, model, rank, steps, n_docs):
    short = re.sub(r"[^A-Za-z0-9.]+", "-", model.split("/")[-1]).lower()
    name = f"{benchmark}_seed{seed}_{short}_r{rank}_s{steps}"
    return name + (f"_n{n_docs}" if n_docs else "")


def load_model(name, load_4bit=False):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return tok, model


def adapter_bytes(model, cfg):
    """Memory of one Micro-LoRA adapter for this model and config (trainable LoRA tensors)."""
    from peft import get_peft_model
    m = get_peft_model(model, cfg)
    size = sum(p.numel() * p.element_size() for p in m.parameters() if p.requires_grad)
    model = m.unload()
    return size, model


def run(benchmark="conflictqa", seed=42, model_name=DEFAULT_MODEL, rank=2, steps=10, n_docs=0,
        skip_icl=False, load_4bit=False, out_dir=None, tok_model=None):
    """Run (or resume) one configuration. tok_model=(tok, model) reuses an already-loaded model."""
    out_dir = Path(out_dir or ROOT / "results" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_name(benchmark, seed, model_name, rank, steps, n_docs)}.json"

    docs = B.build(benchmark, seed)
    if n_docs:
        docs = B.balanced_subset(docs, n_docs)

    tok, model = tok_model or load_model(model_name, load_4bit)
    torch.manual_seed(seed)
    variants = "gemma" not in model_name.lower()  # Gemma keeps the notebooks' exact answer reading
    answer_ids = P.answer_token_ids(tok, variants=variants)
    cfg = P.lora_config(r=rank)

    state = json.load(open(out, encoding="utf-8")) if out.exists() else None
    if state is None:
        size, model = adapter_bytes(model, cfg)
        state = {"config": {"benchmark": benchmark, "seed": seed, "model": model_name, "rank": rank, "alpha": 2 * rank,
                            "steps": steps, "n_docs": len(docs), "answer_ids": answer_ids, "adapter_bytes": size,
                            "gpu": torch.cuda.get_device_name(0)},
                 "records": []}
    done = len(state["records"])
    print(f"{out.name}: {done}/{len(docs)} documents already done")

    for i in range(done, len(docs)):
        d = docs[i]
        t0 = time.time()
        beliefs, confs, chunks, model = P.air_beliefs(d["document"], d["query"], model, tok, cfg, answer_ids, steps=steps)
        air_seconds = time.time() - t0
        gated, idx = P.confidence_gate(beliefs, confs)
        rec = {
            "label": d["label"], "query": d["query"], "raw_question": d["raw_question"],
            "num_chunks": len(chunks), "chunks": chunks, "confidences": confs,
            "all_belief_states": [(float(b[0]), float(b[1])) for b in beliefs],
            "air_confidence_only": P.max_cross_jsd(gated, [confs[j] for j in idx]),
            "air_seconds": air_seconds,
        }
        if not skip_icl:
            rec["icl_split_window"] = P.icl_split_window(d["document"], d["query"], model, tok, answer_ids)
        state["records"].append(rec)
        json.dump(state, open(out, "w", encoding="utf-8"))
        print(f"  [{i + 1}/{len(docs)}] label={d['label']} chunks={len(chunks)} {air_seconds:.1f}s")
    return out, (tok, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=B.NAMES, default="conflictqa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n-docs", type=int, default=0, help="balanced subset size (0 = all 300)")
    ap.add_argument("--skip-icl", action="store_true")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit quantization (needs bitsandbytes)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out, _ = run(a.benchmark, a.seed, a.model, a.rank, a.steps, a.n_docs, a.skip_icl, a.load_4bit, a.out_dir)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
