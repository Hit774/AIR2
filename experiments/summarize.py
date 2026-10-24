"""Turn experiment runs into paper-ready tables (CPU only).

  python experiments/summarize.py                      # every run in results/runs + the paper's seed-42 results
  python experiments/summarize.py --rank-table         # results/rank_ablation -> ablation table (+ LaTeX)
  python experiments/summarize.py --no-paper           # only new runs

For each run it computes, with the same gate code as the paper: confidence-only AIR, split-window ICL, and the
lexical / semantic / hybrid gates at the paper's floors and with cross-validated floors, plus bootstrap CIs,
paired tests against ICL and abstention counts by class. Runs that differ only in seed are aggregated as
mean +/- std. Writes results/summary/summary.json and summary.md.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import scoring as S

ROOT = Path(__file__).resolve().parent.parent
KEY_METRICS = [("AIR confidence-only", "auc"), ("ICL split-window", "auc"), ("AIR hybrid", "auc"),
               ("AIR hybrid", "cv_auc_mean"), ("AIR lexical", "auc"), ("AIR semantic", "auc")]


def load_runs(paths):
    runs = []
    for p in paths:
        run = json.load(open(p, encoding="utf-8"))
        if len(run.get("records", [])) < run["config"].get("n_docs", 0):
            print(f"skipping {p.name}: incomplete ({len(run['records'])}/{run['config']['n_docs']} documents)")
            continue
        run["config"]["file"] = p.name
        runs.append(run)
    return runs


def fmt(v, digits=3):
    return "--" if v is None else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--no-paper", action="store_true", help="do not include the paper's seed-42 results")
    ap.add_argument("--rank-table", action="store_true")
    ap.add_argument("--rank-dir", default=str(ROOT / "results" / "rank_ablation"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "summary"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = []

    if a.rank_table:
        rows = []
        for run in load_runs(sorted(Path(a.rank_dir).glob("*.json"))):
            c, recs = run["config"], run["records"]
            m = S.score_run(recs, "conflictqa", n_boot=200)["methods"]
            chunks = sum(r["num_chunks"] for r in recs)
            rows.append({"rank": c["rank"], "alpha": c["alpha"], "steps": c["steps"], "adapter_mb": c["adapter_bytes"] / 2 ** 20,
                         "sec_per_chunk": sum(r["air_seconds"] for r in recs) / chunks, "n": len(recs),
                         "auc_conf_only": m["AIR confidence-only"]["auc"], "auc_hybrid": m["AIR hybrid"]["auc"],
                         "f1_hybrid": m["AIR hybrid"]["f1"], "cv_auc_hybrid": m["AIR hybrid"]["cv_auc_mean"]})
        rows.sort(key=lambda r: (r["steps"] != 10, r["rank"], r["steps"]))
        md += ["## Rank / steps ablation (ConflictQA)", "",
               "| r | alpha | steps | adapter MB | s/chunk | AUC conf-only | AUC hybrid (rho=0.35) | F1 hybrid | CV AUC hybrid |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['rank']} | {r['alpha']} | {r['steps']} | {r['adapter_mb']:.2f} | {r['sec_per_chunk']:.2f} | {r['auc_conf_only']:.3f} | "
               f"{r['auc_hybrid']:.3f} | {r['f1_hybrid']:.3f} | {r['cv_auc_hybrid']:.3f} |" for r in rows]
        latex = ["\\begin{tabular}{c c c r c c c}", "\\toprule",
                 "\\textbf{$r$} & \\textbf{$\\alpha$} & \\textbf{Steps} & \\textbf{Adapter} & \\textbf{s/chunk} & \\textbf{AUC} & \\textbf{F1} \\\\",
                 "\\midrule"]
        for r in rows:
            latex.append(f"{r['rank']} & {r['alpha']} & {r['steps']} & {r['adapter_mb']:.2f}\\,MB & {r['sec_per_chunk']:.2f} & "
                         f"{r['auc_hybrid']:.3f} & {r['f1_hybrid']:.3f} \\\\")
        latex += ["\\bottomrule", "\\end{tabular}"]
        (out / "rank_table.tex").write_text("\n".join(latex), encoding="utf-8")
        json.dump(rows, open(out / "rank_ablation.json", "w"), indent=1)
        print("\n".join(md))
        print(f"\nLaTeX table written to {out / 'rank_table.tex'}")
        return

    runs = load_runs(sorted(Path(a.runs).glob("*.json"))) if Path(a.runs).exists() else []
    if not a.no_paper:
        for b in ["conflictqa", "strategyqa"]:
            runs.append(S.load_notebook_run(b))
    if not runs:
        raise SystemExit("no runs found")

    scored = []
    for run in runs:
        c = run["config"]
        print(f"scoring {c['benchmark']} seed={c['seed']} model={c['model']} r={c['rank']} steps={c['steps']} ...")
        scored.append({"config": c, "metrics": S.score_run(run["records"], c["benchmark"], n_boot=a.bootstrap)})

    groups = defaultdict(list)
    for s in scored:
        c = s["config"]
        groups[(c["benchmark"], c["model"], c["rank"], c["steps"], s["metrics"]["n"])].append(s)

    md += ["## Mean over seeds (ROC-AUC)", "",
           "| benchmark | model | r | steps | seeds | conf-only | ICL | hybrid (rho=0.35) | hybrid (CV rho) | lexical | semantic |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for (bench, model, rank, steps, n), items in sorted(groups.items()):
        row = {"benchmark": bench, "model": model, "rank": rank, "steps": steps, "n": n,
               "seeds": sorted(i["config"]["seed"] for i in items)}
        cells = []
        for meth, key in KEY_METRICS:
            vals = [i["metrics"]["methods"][meth][key] for i in items if meth in i["metrics"]["methods"]]
            row[f"{meth} {key}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
            cells.append("--" if not vals else (f"{np.mean(vals):.3f}" + (f" ± {np.std(vals):.3f}" if len(vals) > 1 else "")))
        summary.append(row)
        md.append(f"| {bench} | {model.split('/')[-1]} | {rank} | {steps} | {', '.join(map(str, row['seeds']))} | " + " | ".join(cells) + " |")

    md += ["", "## Per run: hybrid gate detail", "",
           "| benchmark | seed | model | AUC [95% CI] | p vs ICL | CV AUC | CV p vs ICL | abstained cons/conf |",
           "|---|---|---|---|---|---|---|---|"]
    for s in scored:
        c, h = s["config"], s["metrics"]["methods"]["AIR hybrid"]
        ci = h.get("ci")
        md.append(f"| {c['benchmark']} | {c['seed']} | {c['model'].split('/')[-1]} | {h['auc']:.3f}"
                  + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f" | {fmt(h.get('p_vs_icl'))} | {h['cv_auc_mean']:.3f} | "
                  f"{fmt(h.get('cv_p_vs_icl'))} | {h['abstained_consistent']}/{h['abstained_conflict']} |")

    json.dump({"groups": summary, "runs": scored}, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
