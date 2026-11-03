"""
Feature-map study: ZZ configurations and other Pauli feature maps
=================================================================

The earlier work only used the ZZ feature map. This study varies it and
compares it with other encodings, every run through the same corrected
pipeline (dedup, train-only preprocessing, 5-fold CV of α and C on the
training split, 3 seeds, 8 qubits, binary and multiclass):

  ZZ settings   reps 1–4 · entanglement linear / circular / full ·
                data map φ(xᵢ,xⱼ) = (π−xᵢ)(π−xⱼ) [Havlíček], xᵢxⱼ,
                π·cos xᵢ cos xⱼ, exp(|xᵢ−xⱼ|²/(8/ln π)) ·
                fixed bandwidth α ∈ {0.05 … 1.5} (C still tuned)
  other maps    Rx angle encoding (product state), Z map (product state,
                Qiskit ZFeatureMap), Pauli maps [Y,ZZ], [X,ZZ], [Z,XX], [Z,YY],
                [Y,XX], [Z,ZZ,ZZZ]

  python feature_map_study.py               # run everything (resumable), then summarise
  python feature_map_study.py --summary     # only rebuild the tables / figure
  python feature_map_study.py --quick       # 1 seed

Results: outputs/feature_maps/<mode>/<config>/{results.json, log.txt}
         outputs/feature_maps/summary.{md,json}, feature_maps.png
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "feature_maps"
SEEDS = [0, 1, 2]
ALPHAS_FIXED = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]

# (group, name, description, extra argv)
CONFIGS = (
    [("zz", f"zz_r{r}_linear", f"ZZ reps={r}, linear", ["--n-reps", r]) for r in (1, 2, 3, 4)]
    + [("zz", f"zz_r2_{e}", f"ZZ reps=2, {e}", ["--entanglement", e])
       for e in ("circular", "full")]
    + [("zz", f"zz_r2_{d}", f"ZZ reps=2, φ={d}", ["--data-map", d])
       for d in ("product", "cosine", "exp")]
    + [("alpha", f"zz_r2_a{a}", f"ZZ reps=2, α={a} fixed", ["--alphas", a])
       for a in ALPHAS_FIXED]
    + [("maps", "rx", "Rx angle encoding", ["--kernel", "rx"]),
       ("maps", "pauli_Z", "Z map (no entanglement)", ["--kernel", "pauli", "--paulis", "Z"])]
    + [("maps", "pauli_" + "-".join(p), "Pauli [" + ", ".join(p) + "]",
        ["--kernel", "pauli", "--paulis", *p])
       for p in (["Y", "ZZ"], ["X", "ZZ"], ["Z", "XX"], ["Z", "YY"], ["Y", "XX"],
                 ["Z", "ZZ", "ZZZ"])]
)


def run(mode, name, argv, seeds):
    out_dir = OUT / mode / name
    if (out_dir / "results.json").exists():
        print(f"  [skip] {mode}/{name}")
        return
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline.main(["--mode", mode, "--n-qubits", "8", *map(str, argv), "--quiet",
                       "--no-save-kernels", "--seeds", *map(str, seeds),
                       "--out-dir", str(out_dir)])
    (out_dir / "log.txt").write_text(buf.getvalue())
    s = json.load(open(out_dir / "results.json"))["summary"]
    print(f"  [done] {mode}/{name:<22} QSVM {s['qsvm']['accuracy']['mean']*100:6.2f}%  "
          f"RBF {s['rbf_svm']['accuracy']['mean']*100:6.2f}%  ({time.time()-t0:.0f}s)",
          flush=True)


# ── Summary ──────────────────────────────────────────────────────────────────

def collect(mode):
    rows = []
    for group, name, desc, _ in CONFIGS:
        p = OUT / mode / name / "results.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        runs = d["runs"]
        acc = np.array([r["qsvm"]["accuracy"] for r in runs])
        rbf = np.array([r["rbf_svm"]["accuracy"] for r in runs])
        kd = [r["kernel_diagnostics"]["quantum"] for r in runs]
        rows.append({
            "group": group, "name": name, "desc": desc, "n_seeds": len(runs),
            "acc": float(acc.mean()), "acc_std": float(acc.std()),
            "f1": float(np.mean([r["qsvm"]["macro_f1"] for r in runs])),
            "rbf": float(rbf.mean()), "delta": float((acc - rbf).mean()),
            "sig": int(sum(r["mcnemar_qsvm_vs_rbf"]["p_value"] < 0.05 for r in runs)),
            "alphas": [r["qsvm"]["alpha"] for r in runs],
            "offdiag": float(np.mean([k["offdiag_mean"] for k in kd])),
            "align": float(np.mean([k["alignment"] for k in kd])),
            "g": float(np.mean([k["geometric_difference"] for k in kd])),
            "train_s": float(np.mean([r["qsvm"]["train_s"] for r in runs])),
        })
    return rows


def table(rows, mode):
    head = (f"| Configuration | QSVM acc. % | macro-F1 | RBF acc. % | Δ (pts) | p<0.05 | "
            f"α chosen | off-diag K | alignment | g(K_C‖K_Q) | train s |")
    lines = [f"### {mode}", "", head, "|" + "---|" * 11]
    for r in rows:
        a = ", ".join(f"{v:g}" for v in r["alphas"])
        lines.append(
            f"| {r['desc']} | {r['acc']*100:.2f} ± {r['acc_std']*100:.2f} | {r['f1']:.4f} | "
            f"{r['rbf']*100:.2f} | {r['delta']*100:+.2f} | {r['sig']}/{r['n_seeds']} | {a} | "
            f"{r['offdiag']:.3f} | {r['align']:.3f} | {r['g']:.1f} | {r['train_s']:.0f} |")
    return "\n".join(lines)


BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b7280"


def figure(all_rows):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [2, 1]})
    for row, mode in enumerate(["binary", "multiclass"]):
        rows = all_rows.get(mode, [])
        if not rows:
            continue
        ax = axes[row, 0]
        bars = [r for r in rows if r["group"] != "alpha"]
        y = np.arange(len(bars))[::-1]
        for yi, r in zip(y, bars):
            # dots, not bars: the axis does not start at 0
            c = BLUE if r["group"] == "zz" else ORANGE
            ax.errorbar(r["acc"] * 100, yi, xerr=r["acc_std"] * 100, fmt="o", ms=8, color=c,
                        ecolor=c, elinewidth=2, capsize=0)
            ax.text(r["acc"] * 100 + r["acc_std"] * 100 + 0.08, yi, f"{r['acc']*100:.2f}",
                    va="center", fontsize=8, color="#374151")
        rbf = np.mean([r["rbf"] for r in rows]) * 100
        ax.axvline(rbf, color=GREY, ls="--", lw=1.5)
        ax.text(rbf, len(bars) - 0.3, f" tuned RBF-SVM {rbf:.2f}", color="#374151", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["desc"] for r in bars], fontsize=9)
        lo = min(r["acc"] - r["acc_std"] for r in bars) * 100
        hi = max(r["acc"] + r["acc_std"] for r in bars) * 100
        ax.set_xlim(lo - 0.5, min(hi + 0.6, 100.2))
        ax.set_xlabel("test accuracy (%), mean ± std over seeds (α, C tuned by CV)")
        ax.set_title(f"{mode}: ZZ settings (blue), other feature maps (orange)",
                     loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        ax = axes[row, 1]
        sweep = sorted([r for r in rows if r["group"] == "alpha"], key=lambda r: r["alphas"][0])
        if sweep:
            a = [r["alphas"][0] for r in sweep]
            ax.errorbar(a, [r["acc"] * 100 for r in sweep], [r["acc_std"] * 100 for r in sweep],
                        color=BLUE, lw=2, marker="o", ms=6, capsize=2, label="QSVM (ZZ, reps=2)")
            ax.axhline(rbf, color=GREY, ls="--", lw=1.5, label="tuned RBF-SVM")
            for r in sweep:
                ax.annotate(f"K̄={r['offdiag']:.2f}", (r["alphas"][0], r["acc"] * 100),
                            textcoords="offset points", xytext=(0, -14), ha="center",
                            fontsize=7, color="#374151")
            ax.set_xscale("log")
            ax.set(xlabel="bandwidth α (fixed; C tuned)", ylabel="test accuracy (%)")
            ax.set_title(f"{mode}: ZZ bandwidth sweep\n(K̄ = mean off-diagonal kernel value)",
                         loc="left", fontsize=11)
            ax.legend(fontsize=8, loc="lower left")
            ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT / "feature_maps.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def summarise():
    all_rows, md = {}, ["# Feature-map study (8 qubits, corrected pipeline)", "",
                        "Mean over seeds; α and C chosen by 5-fold CV on the training split "
                        "unless α is fixed. Δ = QSVM − RBF-SVM accuracy; p<0.05 counts seeds "
                        "where McNemar's test is significant. g = geometric difference "
                        "(Huang et al. 2021).", ""]
    for mode in ["binary", "multiclass"]:
        rows = collect(mode)
        if rows:
            all_rows[mode] = rows
            md += [table(rows, mode), ""]
    if not all_rows:
        print("no results yet")
        return
    text = "\n".join(md)
    (OUT / "summary.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(all_rows, indent=2))
    print("\n" + text)
    print(f"\nFigure: {figure(all_rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["binary", "multiclass"])
    a = ap.parse_args()
    if not a.summary:
        for mode in a.modes:
            print(f"\n=== {mode} ===")
            for group, name, desc, argv in CONFIGS:
                run(mode, name, argv, SEEDS[:1] if a.quick else SEEDS)
    summarise()
