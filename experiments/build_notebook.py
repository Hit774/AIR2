"""Generate experiments/run_all_experiments.ipynb from the tested modules.

The notebook is self-contained: the library code is copied in from the repository files (with the
cross-file imports, path setup and command-line entry points removed), followed by orchestration cells
that run every experiment unattended, each behind a crash guard, and resume after interruptions.

Re-run after editing any module so the notebook stays in sync:
    python experiments/build_notebook.py
"""
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "run_all_experiments.ipynb"

# (file, functions/assignments to drop). Order matters: later modules use earlier ones.
MODULES = [
    ("analysis/rebuild_benchmarks.py", {"verify_and_save", "main", "ROOT"}),
    ("analysis/gate_analysis.py", {"main", "ROOT", "BENCHMARKS"}),
    ("air/pipeline.py", set()),
    ("experiments/benchmarks.py", {"ROOT"}),
    ("experiments/run_air.py", {"main", "ROOT"}),
    ("experiments/scoring.py", {"ROOT"}),
    ("experiments/memory_scaling.py", {"main", "ROOT"}),
    ("experiments/baselines_full_document.py", {"main", "ROOT", "HERE"}),
    ("experiments/rank_ablation.py", {"main", "ROOT"}),
    ("experiments/summarize.py", {"main", "ROOT"}),
]
ALIAS = re.compile(r"\b(?:RB|G|P|S|R|B)\.")
LOCAL_IMPORT = re.compile(r"^\s*(import (rebuild_benchmarks|gate_analysis|benchmarks|scoring|run_air)( as \w+)?"
                          r"|from air import pipeline as P)\s*(#.*)?$")


def strip_module(path, drop):
    src = (REPO / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    doc = ast.get_docstring(tree) or ""
    remove = set()
    for node in tree.body:
        start, end = node.lineno - 1, node.end_lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        name = getattr(node, "name", None)
        targets = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        is_doc = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and node is tree.body[0]
        is_main_guard = isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        if is_doc or is_main_guard or name in drop or any(t in drop for t in targets):
            remove.update(range(start, end))
    kept = []
    for i, line in enumerate(lines):
        if i in remove or LOCAL_IMPORT.match(line) or "sys.path.insert" in line:
            continue
        kept.append(line)
    code = "\n".join(kept)
    for _ in range(3):  # S.G.x -> G.x -> x
        code = ALIAS.sub("", code)
    code = re.sub(r"\n{3,}", "\n\n\n", code).strip() + "\n"
    ast.parse(code)  # must still be valid Python
    return doc, code


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = r"""
# AIR — all additional experiments, unattended

Runs every experiment that strengthens the paper, one after another, on a single GPU (tested setup: RTX 4090, 24 GB).

**How to use**
1. Clone or copy the AIR repository to the GPU machine (this notebook needs the paper's saved results in `results/`).
2. Open this notebook from the repository (`experiments/` or the repository root), edit the **Settings** cell if needed.
3. **Kernel → Restart & Run All.** The first cells install missing packages and ask for your Hugging Face token
   *immediately*; after that nothing waits for input, so you can leave the machine.

**Safe to leave running**
* Each experiment runs inside a crash guard: if one fails, the full error is written to `results/run_all_log.txt`
  and the notebook continues with the next experiment.
* Every experiment saves after each document. If the machine restarts, **Run All** again: finished work is skipped.
* The final cells print a status table and write paper-ready tables to `results/summary/`.

**Approximate GPU time (RTX 4090), in run order:** memory scaling 1 h · whole-document baselines 0.5 h ·
ConflictQA seeds 4.5 h · de-biased StrategyQA 2 h · long-document AUC 1.5 h · StrategyQA seeds 3 h ·
rank ablation 5.5 h · large judge 1 h · second base model 3 h — about **22 hours** with everything enabled.

The library code below is generated from the repository files by `experiments/build_notebook.py`; edit the
`.py` files and regenerate rather than editing those cells here.
"""

SETTINGS = r'''
# ======================= Settings: edit, then Kernel -> Restart & Run All =======================
RUN = {                                # set False to skip an experiment
    "memory_scaling": True,            # peak memory / time vs. length, 2k-128k tokens        ~1 h
    "baselines_conflictqa": True,      # LLM judge (Gemma) + retrieve-then-NLI + NLI models     ~0.5 h
    "seeds_conflictqa": True,          # AIR + ICL on ConflictQA, extra seeds                  ~2.2 h per seed
    "strategyqa_debiased": True,       # fixed StrategyQA benchmark + its baselines             ~2 h
    "auc_long_documents": True,        # conflict-detection AUC on 8k-token documents          ~1.5 h
    "seeds_strategyqa": True,          # AIR + ICL on StrategyQA, extra seeds                  ~1.5 h per seed
    "rank_ablation": True,             # r in {1,2,4,8,16}, steps in {3,5,10,20}                ~5.5 h
    "baselines_large_judge": True,     # 7B long-context judge in 4-bit (installs bitsandbytes) ~1 h
    "second_model": True,              # AIR + ICL with a second base model on ConflictQA       ~3 h
    "summary": True,                   # paper-ready tables (CPU)                               minutes
}
EXTRA_SEEDS = [43, 44]
MEMORY_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
AUC_LENGTH, AUC_DOCS = 8192, 40
RANK_N_DOCS = 100                      # 60 halves the rank-ablation time
LARGE_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
SECOND_MODEL = "Qwen/Qwen2.5-3B-Instruct"
INSTALL_MISSING_PACKAGES = True
SHUTDOWN_WHEN_DONE = False             # True powers the machine off at the end (useful on rented GPUs)
'''

SETUP = r'''
import datetime, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path

# Repository root: the folder that contains results/ with the paper's saved results.
ROOT = next((p for p in [Path.cwd(), Path.cwd().parent] if (p / "results" / "conflictqa").exists()), None)
if ROOT is None:
    raise SystemExit("Open this notebook from inside the AIR repository (results/conflictqa/ was not found).")
(ROOT / "results").mkdir(exist_ok=True)
LOG = ROOT / "results" / "run_all_log.txt"

packages = {"torch": "torch", "transformers": "transformers", "peft": "peft", "accelerate": "accelerate",
            "sentence_transformers": "sentence-transformers", "sklearn": "scikit-learn",
            "matplotlib": "matplotlib", "huggingface_hub": "huggingface_hub"}
if RUN.get("baselines_large_judge"):
    packages["bitsandbytes"] = "bitsandbytes"
missing = [pip for mod, pip in packages.items() if importlib.util.find_spec(mod) is None]
if missing:
    if not INSTALL_MISSING_PACKAGES:
        raise SystemExit(f"Missing packages: {missing}")
    print("installing", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# Ask for the Hugging Face token NOW, so the run never waits for input later.
from huggingface_hub import get_token, login
if not (os.environ.get("HF_TOKEN") or get_token()):
    from getpass import getpass
    login(token=getpass("Hugging Face token (needs access to google/gemma-2-2b-it): "))

# Fail fast on model access problems instead of hours into the run.
from transformers import AutoTokenizer
needed = ["google/gemma-2-2b-it"]
if RUN.get("baselines_large_judge"):
    needed.append(LARGE_JUDGE)
if RUN.get("second_model"):
    needed.append(SECOND_MODEL)
for name in needed:
    AutoTokenizer.from_pretrained(name)
    print(f"access OK: {name}")

import torch
HAS_GPU = torch.cuda.is_available()
print(f"repository: {ROOT}")
print(f"GPU: {torch.cuda.get_device_name(0) if HAS_GPU else 'none - GPU experiments will be skipped'}")
'''

GUARD = r'''
STATUS = {}
GPU_EXPERIMENTS = {k for k in RUN if k != "summary"}
_SHARED = {"tok_model": None}


def log(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


def gemma():
    """The shared Gemma model, loaded once and reused across experiments."""
    if _SHARED["tok_model"] is None:
        _SHARED["tok_model"] = load_model(DEFAULT_MODEL)
    return _SHARED["tok_model"]


def release_model():
    """Drop the shared model (after a failure it may still carry a LoRA wrapper) and free GPU memory."""
    _SHARED["tok_model"] = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def guarded(name, fn):
    """Run one experiment; on any error, log the traceback and continue with the next experiment."""
    if not RUN.get(name):
        STATUS[name] = "skipped (disabled)"
        return None
    if name in GPU_EXPERIMENTS and not HAS_GPU:
        STATUS[name] = "skipped (no GPU)"
        return None
    log(f"START {name}")
    t0 = time.time()
    try:
        result = fn()
        STATUS[name] = f"done ({(time.time() - t0) / 3600:.2f} h)"
        log(f"DONE  {name} in {(time.time() - t0) / 3600:.2f} h")
        return result
    except Exception:
        STATUS[name] = f"FAILED after {(time.time() - t0) / 3600:.2f} h - see {LOG.name}"
        log(f"FAILED {name}\n{traceback.format_exc()}")
        release_model()
        return None
'''

EXPERIMENTS = [
    ("1. Memory and time vs. document length",
     "Answers: *“the O(1) / web-scale claim is never measured.”* Output: `results/memory_scaling/`.",
     r'''
def _memory():
    tok, model = gemma()
    _, model = memory_sweep(tok, model, ROOT / "results" / "memory_scaling", MEMORY_LENGTHS, 1, 42)
    _SHARED["tok_model"] = (tok, model)

guarded("memory_scaling", _memory)
'''),
    ("2. Whole-document baselines on ConflictQA",
     "Answers: *“the baselines are weak; nothing reads the whole document.”* Gemma as a whole-document judge, "
     "retrieve-then-NLI and three NLI models, each tested against AIR on the paper's documents. Output: `results/baselines/`.",
     r'''
guarded("baselines_conflictqa", lambda: run_baselines("conflictqa", out=ROOT / "results" / "baselines"))
'''),
    ("3. Extra seeds on ConflictQA",
     "Answers: *“single seed.”* New items, distractors and LoRA initialization per seed. Output: `results/runs/`.",
     r'''
def _seeds_conflictqa():
    for seed in EXTRA_SEEDS:
        _, _SHARED["tok_model"] = run("conflictqa", seed, tok_model=gemma(), out_dir=ROOT / "results" / "runs")

guarded("seeds_conflictqa", _seeds_conflictqa)
'''),
    ("4. De-biased StrategyQA",
     "Answers: *“StrategyQA leaks the label to the gate.”* Both target passages restate the question in both "
     "classes. Runs AIR + ICL, then the whole-document baselines on the same documents.",
     r'''
def _debiased():
    out, _SHARED["tok_model"] = run("strategyqa_debiased", 42, tok_model=gemma(), out_dir=ROOT / "results" / "runs")
    run_baselines("strategyqa_debiased", air_run=out, out=ROOT / "results" / "baselines")

guarded("strategyqa_debiased", _debiased)
'''),
    ("5. Accuracy on longer documents",
     "Answers: *“does accuracy hold beyond 2k tokens?”* Output: `results/memory_scaling/auc_at_8192.json`.",
     r'''
def _auc_long():
    tok, model = gemma()
    _, model = auc_at_length(tok, model, ROOT / "results" / "memory_scaling", AUC_LENGTH, AUC_DOCS)
    _SHARED["tok_model"] = (tok, model)

guarded("auc_long_documents", _auc_long)
'''),
    ("6. Extra seeds on StrategyQA", "Same as 3, for the original StrategyQA benchmark.",
     r'''
def _seeds_strategyqa():
    for seed in EXTRA_SEEDS:
        _, _SHARED["tok_model"] = run("strategyqa", seed, tok_model=gemma(), out_dir=ROOT / "results" / "runs")

guarded("seeds_strategyqa", _seeds_strategyqa)
'''),
    ("7. Rank and distillation-steps ablation",
     "Answers: *“the rank ablation has no numbers.”* Uses the final gate. Output: `results/rank_ablation/`.",
     r'''
def _rank():
    _SHARED["tok_model"] = run_rank_ablation(RANK_N_DOCS, out_dir=str(ROOT / "results" / "rank_ablation"), tok_model=gemma())

guarded("rank_ablation", _rank)
'''),
    ("8. Large long-context judge",
     "A 7B instruction model in 4-bit reads each whole document. Frees the shared Gemma model first.",
     r'''
def _large_judge():
    release_model()
    run_baselines("conflictqa", judge_models=[LARGE_JUDGE], load_4bit=True, nli_models=[],
                  out=ROOT / "results" / "baselines", tag="_large_judge")

guarded("baselines_large_judge", _large_judge)
'''),
    ("9. Second base model",
     "Answers: *“only one 2B model.”* AIR + ICL on ConflictQA with another model family.",
     r'''
def _second_model():
    release_model()
    out, tok_model = run("conflictqa", 42, model_name=SECOND_MODEL, out_dir=ROOT / "results" / "runs")
    del tok_model
    release_model()

guarded("second_model", _second_model)
'''),
    ("10. Paper-ready summary",
     "Mean ± std over seeds for every method, per-run significance tests, and the rank table (also as LaTeX). "
     "Output: `results/summary/`.",
     r'''
def _summary():
    summarize_runs(ROOT / "results" / "runs", include_paper=True, out=ROOT / "results" / "summary")
    if any((ROOT / "results" / "rank_ablation").glob("*.json")):
        rank_table(ROOT / "results" / "rank_ablation", ROOT / "results" / "summary")

guarded("summary", _summary)
'''),
]

FINISH = r'''
log("ALL EXPERIMENTS FINISHED")
width = max(len(k) for k in STATUS)
report = "\n".join(f"{k:<{width}}  {v}" for k, v in STATUS.items())
print(report)
(ROOT / "results" / "run_all_status.txt").write_text(report + "\n", encoding="utf-8")
print(f"\nlog: {LOG}\nsummary tables: {ROOT / 'results' / 'summary'}")

if SHUTDOWN_WHEN_DONE:
    log("shutting down")
    os.system("shutdown /s /t 60" if os.name == "nt" else "shutdown -h now")
'''


def main():
    cells = [md(INTRO), md("## Settings"), code(SETTINGS), md("## Setup: packages, Hugging Face login, GPU check"), code(SETUP),
             md("## Library code\nGenerated from the repository files; edit the `.py` files and run "
                "`python experiments/build_notebook.py` instead of editing these cells.")]
    for path, drop in MODULES:
        doc, src = strip_module(path, drop)
        title = doc.splitlines()[0] if doc else path
        cells.append(md(f"### `{path}`\n{title}"))
        cells.append(code(src))
    cells += [md("## Crash guard and shared model"), code(GUARD), md("## Experiments")]
    for title, text, src in EXPERIMENTS:
        cells += [md(f"### {title}\n{text}"), code(src)]
    cells += [md("## Status report"), code(FINISH)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
