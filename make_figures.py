"""
Build report figures and LaTeX tables from outputs/benchmark/.

  python make_figures.py          → report/figures/*.pdf, report/tables/*.tex
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "outputs" / "benchmark"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#6b7280", "#dc2626", "#7c3aed"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})


def load(exp, name):
    p = BENCH / exp / name / "results.json"
    return json.load(open(p)) if p.exists() else None


def mean(r, model, metric):
    v = r["summary"].get(model, {}).get(metric)
    return (v["mean"], v["std"]) if v else (np.nan, np.nan)


def per_seed(r, path):
    out = []
    for run in r["runs"]:
        v = run
        for k in path:
            v = v[k]
        out.append(v)
    return np.array(out, dtype=float)


def pct(ms):
    m, s = ms
    return "--" if np.isnan(m) else f"{m*100:.2f} $\\pm$ {s*100:.2f}"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=140)
    plt.close(fig)


# ── Headline table ────────────────────────────────────────────────────────────

def headline():
    rows = []
    methods = [("zz", "ZZ entangling (paper circuit)"), ("rx", "Analytical $R_x$ (separable)"),
               ("rxshots", "Sampling, 1024 shots ($R_x$)"), ("zz16r2", "ZZ, 16 qubits, 2 reps")]
    for mode in ["binary", "multiclass"]:
        for key, label in methods:
            for proto in ["legacy", "improved"]:
                r = load("headline", f"{mode}_{key}_{proto}")
                if r is None:
                    continue
                p = [x["mcnemar_qsvm_vs_rbf"]["p_value"] for x in r["runs"]]
                rows.append((mode, label, proto, pct(mean(r, "qsvm", "accuracy")),
                             pct(mean(r, "rbf_svm", "accuracy")),
                             pct(mean(r, "qsvm", "macro_f1")),
                             f"{sum(v < 0.05 for v in p)}/{len(p)}"))
    with open(TAB / "headline.tex", "w") as f:
        last = None
        for mode, label, proto, qa, ca, qf, sig in rows:
            if last and mode != last:
                f.write("\\midrule\n")
            last = mode
            f.write(f"{mode} & {label} & {proto} & {qa} & {ca} & {qf} & {sig} \\\\\n")

    # bar figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, mode in zip(axes, ["binary", "multiclass"]):
        labels, q, c, qe, ce = [], [], [], [], []
        for key, short in [("zz", "ZZ"), ("rx", "Rx"), ("rxshots", "Rx+shots")]:
            for proto in ["legacy", "improved"]:
                r = load("headline", f"{mode}_{key}_{proto}")
                if r is None:
                    continue
                labels.append(f"{short}\n{proto}")
                (a, b), (cc, d) = mean(r, "qsvm", "accuracy"), mean(r, "rbf_svm", "accuracy")
                q.append(a * 100); qe.append(b * 100); c.append(cc * 100); ce.append(d * 100)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, q, 0.4, yerr=qe, color=BLUE, label="QSVM", capsize=3)
        ax.bar(x + 0.2, c, 0.4, yerr=ce, color=ORANGE, label="RBF-SVM (same PCA features)", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        lo = min(q + c) - 4
        ax.set_ylim(max(lo, 50), 100.5)
        ax.set_title(f"{mode.title()} — accuracy (mean ± std, 5 seeds)")
        ax.set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, "headline")


# ── Ablation ──────────────────────────────────────────────────────────────────

ABL = [("0_legacy", "Original protocol"), ("1_dedup", "+ remove duplicates"),
       ("2_trainonly_nolog", "+ train-only scaling"), ("3_trainonly_log", "+ log transform"),
       ("4_tuned", "+ CV tuning ($\\alpha$, $C$, $\\gamma$)")]


def ablation():
    with open(TAB / "ablation.tex", "w") as f:
        for mode in ["binary", "multiclass"]:
            for key, label in ABL:
                r = load("ablation", f"{mode}_{key}")
                if r is None:
                    continue
                ov = np.mean([x["data"]["test_rows_with_exact_copy_in_train"] for x in r["runs"]])
                q, c = mean(r, "qsvm", "accuracy"), mean(r, "rbf_svm", "accuracy")
                gap = (q[0] - c[0]) * 100
                p = [x["mcnemar_qsvm_vs_rbf"]["p_value"] for x in r["runs"]]
                f.write(f"{mode} & {label} & {ov*100:.1f}\\% & {pct(q)} & {pct(c)} & "
                        f"{gap:+.2f} & {sum(v < 0.05 for v in p)}/{len(p)} \\\\\n")
            if mode == "binary":
                f.write("\\midrule\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mode in zip(axes, ["binary", "multiclass"]):
        q, c, qe, ce, labels = [], [], [], [], []
        for key, label in ABL:
            r = load("ablation", f"{mode}_{key}")
            if r is None:
                continue
            labels.append(label.replace("$\\alpha$", "α").replace("$C$", "C")
                          .replace("$\\gamma$", "γ").replace("+ ", "+").replace(" (", "\n("))
            a, b = mean(r, "qsvm", "accuracy"); cc, d = mean(r, "rbf_svm", "accuracy")
            q.append(a * 100); qe.append(b * 100); c.append(cc * 100); ce.append(d * 100)
        x = np.arange(len(labels))
        ax.errorbar(x, q, qe, marker="o", color=BLUE, label="QSVM (ZZ)", capsize=3)
        ax.errorbar(x, c, ce, marker="s", color=ORANGE, label="RBF-SVM", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_title(f"{mode.title()}: effect of each fix")
        ax.set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=8)
    save(fig, "ablation")


# ── Bandwidth ─────────────────────────────────────────────────────────────────

def bandwidth():
    alphas = [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = []
    for mode, ls in [("binary", "-"), ("multiclass", "--")]:
        for nq, col in [(8, BLUE), (16, PURPLE)]:
            acc, off, ali, A = [], [], [], []
            for a in alphas:
                r = load("bandwidth", f"{mode}_{nq}q_a{a}")
                if r is None:
                    continue
                A.append(a)
                acc.append(mean(r, "qsvm", "accuracy")[0] * 100)
                off.append(np.mean(per_seed(r, ["kernel_diagnostics", "quantum", "offdiag_mean"])))
                ali.append(np.mean(per_seed(r, ["kernel_diagnostics", "quantum", "alignment"])))
                rows.append((mode, nq, a, acc[-1], off[-1], ali[-1]))
            lab = f"{mode}, {nq}q"
            axes[0].plot(A, acc, ls, marker="o", color=col, label=lab)
            axes[1].plot(A, off, ls, marker="o", color=col, label=lab)
            axes[2].plot(A, ali, ls, marker="o", color=col, label=lab)
    for ax, t, yl in zip(axes, ["Test accuracy", "Kernel concentration", "Kernel–target alignment"],
                         ["Accuracy (%)", "mean off-diagonal K", "centred alignment"]):
        ax.set_xscale("log"); ax.set_xlabel("bandwidth α"); ax.set_title(t); ax.set_ylabel(yl)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    save(fig, "bandwidth")
    with open(TAB / "bandwidth.tex", "w") as f:
        for mode, nq, a, acc, off, ali in rows:
            f.write(f"{mode} & {nq} & {a:g} & {acc:.2f} & {off:.2e} & {ali:.3f} \\\\\n")


# ── Qubits ────────────────────────────────────────────────────────────────────

def qubits():
    nqs = [2, 4, 6, 8, 10, 12, 14, 16]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = []
    for ax, mode in zip(axes[:2], ["binary", "multiclass"]):
        for kern, col, lab in [("zz", BLUE, "QSVM ZZ"), ("rx", GREEN, "QSVM Rx")]:
            x, y, e, c, ce, var = [], [], [], [], [], []
            for nq in nqs:
                r = load("qubits", f"{mode}_{kern}_{nq}q")
                if r is None:
                    continue
                x.append(nq)
                m, s = mean(r, "qsvm", "accuracy"); y.append(m * 100); e.append(s * 100)
                m, s = mean(r, "rbf_svm", "accuracy"); c.append(m * 100); ce.append(s * 100)
                var.append(np.mean([run["pca_variance"] for run in r["runs"]]) * 100)
                if kern == "zz":
                    g = np.mean(per_seed(r, ["kernel_diagnostics", "quantum", "geometric_difference"]))
                    rows.append((mode, nq, var[-1], y[-1], c[-1], g,
                                 np.mean([run["qsvm"]["alpha"] for run in r["runs"]])))
            ax.errorbar(x, y, e, marker="o", color=col, label=lab, capsize=3)
            if kern == "zz":
                ax.errorbar(x, c, ce, marker="s", color=ORANGE, label="RBF-SVM (same PCA)", capsize=3)
                axes[2].plot(x, var, marker="o", ls="-" if mode == "binary" else "--",
                             color=GREY, label=f"PCA variance ({mode})")
        ax.set(xlabel="qubits = PCA components", ylabel="Accuracy (%)", title=mode.title())
        ax.legend(fontsize=8)
    axes[2].set(xlabel="qubits = PCA components", ylabel="variance retained (%)",
                title="PCA variance retained")
    axes[2].legend(fontsize=8)
    save(fig, "qubits")
    with open(TAB / "qubits.tex", "w") as f:
        for mode, nq, var, q, c, g, a in rows:
            f.write(f"{mode} & {nq} & {var:.1f} & {q:.2f} & {c:.2f} & {a:.3f} & {g:.2f} \\\\\n")


# ── Shots ─────────────────────────────────────────────────────────────────────

def shots():
    S = [16, 64, 256, 1024, 4096]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rows = []
    for mode, col in [("binary", BLUE), ("multiclass", PURPLE)]:
        x, y, e = [], [], []
        for s in S:
            r = load("shots", f"{mode}_8q_s{s}")
            if r is None:
                continue
            m, sd = mean(r, "qsvm", "accuracy")
            x.append(s); y.append(m * 100); e.append(sd * 100)
            rows.append((mode, str(s), m * 100, sd * 100,
                         np.mean([run["qsvm"]["alpha"] for run in r["runs"]])))
        ax.errorbar(x, y, e, marker="o", color=col, label=f"{mode} (shots)", capsize=3)
        r = load("shots", f"{mode}_8q_exact")
        if r:
            m, sd = mean(r, "qsvm", "accuracy")
            ax.axhline(m * 100, color=col, ls=":", label=f"{mode} exact")
            rows.append((mode, "exact", m * 100, sd * 100,
                         np.mean([run["qsvm"]["alpha"] for run in r["runs"]])))
    ax.set(xscale="log", xlabel="shots per kernel entry", ylabel="Accuracy (%)",
           title="Finite-shot kernel estimation (8 qubits)")
    ax.legend(fontsize=8)
    save(fig, "shots")
    with open(TAB / "shots.tex", "w") as f:
        for mode, s, m, sd, a in rows:
            f.write(f"{mode} & {s} & {m:.2f} $\\pm$ {sd:.2f} & {a:.3f} \\\\\n")


# ── Architecture ──────────────────────────────────────────────────────────────

def architecture():
    with open(TAB / "architecture.tex", "w") as f:
        for mode in ["binary", "multiclass"]:
            for reps in [1, 2, 3]:
                cells = []
                for ent in ["linear", "circular", "full"]:
                    r = load("architecture", f"{mode}_r{reps}_{ent}")
                    cells.append(pct(mean(r, "qsvm", "accuracy")) if r else "--")
                f.write(f"{mode} & {reps} & " + " & ".join(cells) + " \\\\\n")
            if mode == "binary":
                f.write("\\midrule\n")


# ── Features & samples ────────────────────────────────────────────────────────

def features():
    with open(TAB / "features.tex", "w") as f:
        for mode in ["binary", "multiclass"]:
            for fs, label in [("paper", "paper (61 listed, 49 non-constant)"),
                              ("all", "all numeric CICFlowMeter columns")]:
                r = load("features", f"{mode}_{fs}")
                if r is None:
                    continue
                f.write(f"{mode} & {label} & {pct(mean(r, 'qsvm', 'accuracy'))} & "
                        f"{pct(mean(r, 'rbf_svm', 'accuracy'))} & "
                        f"{pct(mean(r, 'random_forest', 'accuracy'))} \\\\\n")


def samples():
    N = [250, 500, 1000, 2000, 4000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mode in zip(axes, ["binary", "multiclass"]):
        for model, col, lab in [("qsvm", BLUE, "QSVM ZZ (8q)"), ("rbf_svm", ORANGE, "RBF-SVM"),
                                ("random_forest", GREEN, "Random Forest (all features)")]:
            x, y, e = [], [], []
            for n in N:
                r = load("samples", f"{mode}_n{n}")
                if r is None:
                    continue
                x.append(np.mean([run["data"]["n_train"] for run in r["runs"]]))
                m, s = mean(r, model, "accuracy"); y.append(m * 100); e.append(s * 100)
            ax.errorbar(x, y, e, marker="o", color=col, label=lab, capsize=3)
        ax.set(xscale="log", xlabel="training samples", ylabel="Accuracy (%)",
               title=f"{mode.title()} learning curve")
        ax.legend(fontsize=8)
    save(fig, "samples")


# ── Speed ─────────────────────────────────────────────────────────────────────

def speed():
    p = BENCH / "speed" / "speed.json"
    if not p.exists():
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    nq = [r["n_qubits"] for r in d["rows"]]
    ax.plot(nq, [r["legacy_total_s_est"] for r in d["rows"]], "o-", color=RED,
            label="original: 2 simulations per entry (est.)")
    ax.plot(nq, [r["cirq_cached_total_s_est"] for r in d["rows"]], "s-", color=ORANGE,
            label="Cirq, one simulation per sample (est.)")
    ax.plot(nq, [r["fast_total_s"] for r in d["rows"]], "^-", color=BLUE,
            label="fast NumPy engine (measured)")
    for h, lab in [(60, "1 min"), (3600, "1 h"), (86400, "1 day")]:
        ax.axhline(h, color=GREY, lw=0.6, ls=":")
        ax.text(nq[0], h * 1.15, lab, fontsize=7, color=GREY)
    ax.set(yscale="log", xlabel="qubits", ylabel="wall-clock seconds",
           title="Train (1600²) + test (400×1600) kernel cost")
    ax.legend(fontsize=7)
    save(fig, "speed")
    with open(TAB / "speed.tex", "w") as f:
        for r in d["rows"]:
            samp = (f"{r['legacy_sampling_total_s_est']/3600:.1f} h"
                    if "legacy_sampling_total_s_est" in r else "--")
            f.write(f"{r['n_qubits']} & {r['legacy_per_entry_s']*1000:.1f} ms & "
                    f"{r['legacy_total_s_est']/3600:.1f} h & {samp} & "
                    f"{r['cirq_cached_total_s_est']:.1f} s & {r['fast_total_s']:.2f} s & "
                    f"{r['legacy_total_s_est']/r['fast_total_s']:,.0f}$\\times$ \\\\\n")


# ── Diagnostics (from the 16-qubit headline run) ──────────────────────────────

def diagnostics():
    with open(TAB / "diagnostics.tex", "w") as f:
        for mode in ["binary", "multiclass"]:
            for key, label in [("zz_legacy", "ZZ, original ($\\alpha{=}1$)"),
                               ("zz_improved", "ZZ, tuned $\\alpha$"),
                               ("zz16r2_improved", "ZZ 16q/2 reps, tuned $\\alpha$"),
                               ("rx_improved", "$R_x$, tuned $\\alpha$")]:
                r = load("headline", f"{mode}_{key}")
                if r is None:
                    continue
                g = lambda k: np.mean(per_seed(r, ["kernel_diagnostics", "quantum", k]))
                ga = lambda k: np.mean(per_seed(r, ["kernel_diagnostics", "rbf", k]))
                n = r["runs"][0]["kernel_diagnostics"]["quantum"]["geometric_difference_n"]
                f.write(f"{mode} & {label} & {g('offdiag_mean'):.3g} & {g('offdiag_std'):.3g} & "
                        f"{g('alignment'):.3f} & {ga('alignment'):.3f} & "
                        f"{g('geometric_difference'):.2f} (vs $\\sqrt{{N}}$={np.sqrt(n):.0f}) \\\\\n")
            if mode == "binary":
                f.write("\\midrule\n")


def detail_tables():
    """Per-metric table for the improved headline runs (incl. RF, FAR, DR)."""
    with open(TAB / "metrics.tex", "w") as f:
        for mode in ["binary", "multiclass"]:
            r = load("headline", f"{mode}_zz_improved")
            if r is None:
                continue
            for model, label in [("qsvm", "QSVM ZZ (tuned)"), ("rbf_svm", "RBF-SVM (tuned)"),
                                 ("random_forest", "Random Forest")]:
                cells = [pct(mean(r, model, m)) for m in
                         ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "auc",
                          "false_alarm_rate", "detection_rate"]]
                f.write(f"{mode} & {label} & " + " & ".join(cells) + " \\\\\n")
            if mode == "binary":
                f.write("\\midrule\n")


def duplicates():
    """Duplicate statistics per class on the full file."""
    import pandas as pd
    import sys
    sys.path.insert(0, str(ROOT))
    from data_loader import _feature_frame, _read_clean
    df = _read_clean(str(ROOT / "output.csv"))
    X = _feature_frame(df, "paper")
    key = X.copy(); key["L"] = df["Label"].values
    dup = key.duplicated()
    with open(TAB / "duplicates.tex", "w") as f:
        for lab, g in key.groupby("L"):
            d = dup[g.index].sum()
            f.write(f"{lab} & {len(g):,} & {d:,} & {len(g)-d:,} & {d/len(g)*100:.1f}\\% \\\\\n")
        f.write(f"\\midrule\nTotal & {len(key):,} & {dup.sum():,} & {len(key)-dup.sum():,} & "
                f"{dup.mean()*100:.1f}\\% \\\\\n")
    # feature skew figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    col = "Flow Bytes/s"
    v = X[col].values
    axes[0].hist(v / v.max() * np.pi, bins=60, color=GREY)
    axes[0].set(title=f"'{col}' after min–max to [0, π]", xlabel="angle (rad)", ylabel="flows",
                yscale="log")
    lv = np.sign(v) * np.log1p(np.abs(v))
    axes[1].hist((lv - lv.min()) / (lv.max() - lv.min()) * np.pi, bins=60, color=BLUE)
    axes[1].set(title=f"'{col}' after signed log, then min–max", xlabel="angle (rad)",
                ylabel="flows", yscale="log")
    save(fig, "skew")


def run_figures():
    import shutil
    for mode in ["binary", "multiclass"]:
        src = BENCH / "headline" / f"{mode}_zz_improved" / "results_seed0.png"
        if src.exists():
            shutil.copy(src, FIG / f"run_{mode}.png")


if __name__ == "__main__":
    for fn in [run_figures, duplicates, speed, headline, ablation, bandwidth, qubits, shots,
               architecture, features, samples, diagnostics, detail_tables]:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:  # keep going so partial results still render
            print(f"FAIL {fn.__name__}: {e!r}")
