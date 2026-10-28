"""
Main Pipeline — Quantum Network Defense (CIC-DDoS2019)
=======================================================
Reads output.csv directly.

Usage (original flags still work):
  python main.py --csv ./output.csv --samples 2000
  python main.py --csv ./output.csv --samples 1000 --n-qubits 12 --n-reps 1
  python main.py --csv ./output.csv --samples 500  --mode multiclass

New options (see TESTING.md for the full list of tested commands):
  --kernel zz|rx               ZZ entangling map or the paper's separable Rx map
  --entanglement linear|circular|full
  --alpha A --C C --no-tune    fixed hyperparameters instead of CV tuning
  --seeds 0 1 2 3 4            repeat with different samples/splits, report mean ± std
  --legacy                     reproduce the ORIGINAL protocol (global scaling,
                               duplicates kept, α=1, C=1, untuned RBF) for comparison
  --backend cirq               use the Cirq simulator instead of the fast engine
  --psd                        with --sampling: PSD-project the shot-noise train kernel
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

from data_loader import duplicate_overlap, load_csv
from evaluation import (
    classification_metrics, kernel_diagnostics, mcnemar, rbf_kernel_matrix,
    standardize, tune_quantum, tune_rbf,
)
from preprocessing import LegacyPreprocessor, QuantumPreprocessor, legacy_global_scale
from qsvm import QSVM, QuantumKernel

DEFAULT_ALPHAS = [0.05, 0.1, 0.2, 0.4, 1.0]
DEFAULT_CS = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMAS = ["scale", 0.01, 0.03, 0.1, 0.3, 1.0]


def _scores(model, X, binary):
    if binary:
        return model.decision_function(X)
    return model.predict_proba(X)


def run_once(args, seed: int, verbose: bool = True) -> dict:
    binary = args.mode == "binary"
    res = {"seed": seed}

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {args.csv} (max={args.samples:,}, seed={seed})...")
    X, y, label_names, info = load_csv(
        csv_path=args.csv, mode=args.mode, max_samples=args.samples,
        balance=True, dedup=not args.keep_duplicates,
        feature_set=args.features, seed=seed, verbose=verbose,
    )
    res["data"] = {k: v for k, v in info.items() if k != "columns"}
    target_names = [label_names[i] for i in sorted(label_names)]
    benign = next((k for k, v in label_names.items() if v == "BENIGN"), 0)

    if args.legacy_scaling:
        X = legacy_global_scale(X)          # the original (leaky) global scaling
        prep = LegacyPreprocessor(args.n_qubits, seed=seed)
    else:
        prep = QuantumPreprocessor(args.n_qubits, log_transform=not args.no_log, seed=seed)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    overlap = duplicate_overlap(X_train, X_test)
    res["data"].update({"n_train": len(y_train), "n_test": len(y_test),
                        "test_rows_with_exact_copy_in_train": overlap})
    print(f"\n[2/6] Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"test rows with an exact copy in train: {overlap*100:.1f}%")

    A_train = prep.fit_transform(X_train)
    print(f"      Preprocessing: {prep.n_input_features} features → PCA {args.n_qubits} "
          f"({prep.explained_variance*100:.1f}% variance, fitted on train only)")
    res["pca_variance"] = prep.explained_variance

    # ── 3. QSVM ───────────────────────────────────────────────────────────────
    def make_kernel(alpha):
        return QuantumKernel(args.n_qubits, args.n_reps, args.shots, args.sampling,
                             args.kernel, args.entanglement, alpha, args.backend, seed,
                             psd_project=args.psd)

    tune = not args.no_tune
    print(f"\n[3/6] QSVM [{args.kernel}, {args.n_qubits}q, reps={args.n_reps}, "
          f"{args.entanglement}] {'tuning α, C by 5-fold CV on train' if tune else ''}")
    t0 = time.time()
    if tune:
        alpha, C, K_train, table = tune_quantum(
            lambda a: make_kernel(a).compute_kernel_matrix(A_train, verbose=False),
            y_train, args.alphas, args.Cs, seed=seed, verbose=verbose)
        res["qsvm_tuning"] = table
    else:
        alpha, C, K_train = args.alpha, args.C, None
    t_tune = time.time() - t0

    qsvm = QSVM(args.n_qubits, args.n_reps, C, args.shots, args.sampling, args.kernel,
                args.entanglement, alpha, args.backend, preprocessor=prep, seed=seed,
                psd_project=args.psd)
    t0 = time.time()
    qsvm.fit(X_train, y_train, K_train=K_train)
    t_q_train = time.time() - t0 + t_tune
    t0 = time.time()
    y_pred_q = qsvm.predict(X_test)
    s_q = _scores(qsvm, X_test, binary)
    t_q_infer = time.time() - t0
    m_q = classification_metrics(y_test, y_pred_q, s_q, benign)
    m_q.update({"alpha": alpha, "C": C, "train_s": t_q_train, "infer_s": t_q_infer})
    res["qsvm"] = m_q
    print(f"      α={alpha}, C={C} | accuracy {m_q['accuracy']*100:.2f}% | "
          f"macro-F1 {m_q['macro_f1']:.4f} | train {t_q_train:.1f}s | infer {t_q_infer:.1f}s")
    if verbose:
        print(classification_report(y_test, y_pred_q, target_names=target_names, digits=4))

    # ── 4. Classical RBF-SVM on the same PCA features ─────────────────────────
    print("[4/6] Classical SVM (RBF) on the same PCA features...")
    Z_train, Z_test = standardize(prep.transform_pca(X_train), prep.transform_pca(X_test))
    t0 = time.time()
    if tune:
        csvm, params, _ = tune_rbf(Z_train, y_train, args.Cs, RBF_GAMMAS, seed=seed)
    else:
        csvm = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed).fit(Z_train, y_train)
        params = {"C": 1.0, "gamma": "scale"}
    t_c_train = time.time() - t0
    t0 = time.time()
    y_pred_c = csvm.predict(Z_test)
    s_c = _scores(csvm, Z_test, binary)
    t_c_infer = time.time() - t0
    m_c = classification_metrics(y_test, y_pred_c, s_c, benign)
    m_c.update({**{k: v for k, v in params.items()}, "train_s": t_c_train, "infer_s": t_c_infer})
    res["rbf_svm"] = m_c
    print(f"      {params} | accuracy {m_c['accuracy']*100:.2f}% | macro-F1 {m_c['macro_f1']:.4f}")
    res["mcnemar_qsvm_vs_rbf"] = mcnemar(y_test, y_pred_q, y_pred_c)

    # ── 5. Optional: Random Forest on all raw features (classical reference) ──
    if args.extra_baselines:
        print("[5/6] Random Forest (all raw features, no PCA)...")
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        s_rf = rf.predict_proba(X_test)[:, 1] if binary else rf.predict_proba(X_test)
        res["random_forest"] = {**classification_metrics(y_test, y_pred_rf, s_rf, benign),
                                "train_s": time.time() - t0}
        print(f"      accuracy {res['random_forest']['accuracy']*100:.2f}%")

    # ── 6. Kernel diagnostics ─────────────────────────────────────────────────
    gamma = csvm.gamma if csvm.gamma != "scale" else 1.0 / (Z_train.shape[1] * Z_train.var())
    res["kernel_diagnostics"] = {
        "quantum": kernel_diagnostics(qsvm.K_train_, y_train,
                                      rbf_kernel_matrix(Z_train, gamma), seed=seed),
        "rbf": kernel_diagnostics(rbf_kernel_matrix(Z_train, gamma), y_train),
    }
    kd = res["kernel_diagnostics"]["quantum"]
    print(f"[6/6] Quantum kernel: off-diag mean {kd['offdiag_mean']:.4f} ± {kd['offdiag_std']:.4f}"
          f" | alignment {kd['alignment']:.4f} | geometric difference g={kd['geometric_difference']:.2f}"
          f" (√N={np.sqrt(kd['geometric_difference_n']):.1f})")

    res["_plot"] = dict(y_test=y_test, s_q=s_q, s_c=s_c, K_train=qsvm.K_train_,
                        y_train=y_train, target_names=target_names)
    return res


def plot_run(res, args, out_dir: Path):
    p = res["_plot"]
    binary = args.mode == "binary"
    names = p["target_names"]
    n = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(f"Quantum Network Defense — {args.kernel.upper()} kernel "
                 f"({args.mode}, {args.n_qubits} qubits, seed {res['seed']})",
                 fontsize=14, fontweight="bold")

    for ax, key, title, cmap in [
        (axes[0, 0], "qsvm", f"QSVM (α={res['qsvm']['alpha']}, C={res['qsvm']['C']})", "Blues"),
        (axes[0, 1], "rbf_svm", "Classical RBF-SVM", "Oranges"),
    ]:
        cm = np.array(res[key]["confusion_matrix"])
        ax.imshow(cm, cmap=cmap)
        ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
               xlabel="Predicted", ylabel="True",
               title=f"{title}\nacc={res[key]['accuracy']*100:.2f}%")
        ax.tick_params(axis="x", rotation=30)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax = axes[0, 2]
    order = np.argsort(p["y_train"], kind="stable")
    ax.imshow(p["K_train"][np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Training quantum kernel (sorted by class)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 0]
    if binary:
        for s, label, c in [(p["s_q"], "QSVM", "#1f77b4"), (p["s_c"], "RBF-SVM", "#ff7f0e")]:
            fpr, tpr, _ = roc_curve(p["y_test"], s)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, f"Multiclass OvR macro AUC\nQSVM {res['qsvm']['auc']:.4f}\n"
                f"RBF  {res['rbf_svm']['auc']:.4f}", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        ax.set_title("ROC (one-vs-rest AUC)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "qsvm_tuning" in res:
        tab = res["qsvm_tuning"]
        for C in sorted({t["C"] for t in tab}):
            rows = [t for t in tab if t["C"] == C]
            ax.plot([t["alpha"] for t in rows], [t["cv_macro_f1"] for t in rows],
                    marker="o", label=f"C={C:g}")
        ax.set(xscale="log", xlabel="bandwidth α", ylabel="CV macro-F1",
               title="Bandwidth / C tuning (train CV)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    ax = axes[1, 2]
    methods = [("QSVM", "qsvm", "#1f77b4"), ("RBF-SVM", "rbf_svm", "#ff7f0e")]
    if "random_forest" in res:
        methods.append(("Random Forest", "random_forest", "#2ca02c"))
    vals = [res[k]["macro_f1"] * 100 for _, k, _ in methods]
    bars = ax.bar([m[0] for m in methods], vals, color=[m[2] for m in methods], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set(ylabel="Macro-F1 (%)", title="Macro-F1 comparison", ylim=[max(0, min(vals) - 10), 101])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"results_seed{res['seed']}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize(runs):
    out = {}
    for key in ["qsvm", "rbf_svm", "random_forest"]:
        if key not in runs[0]:
            continue
        out[key] = {}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "auc",
                       "false_alarm_rate", "detection_rate", "train_s", "infer_s"]:
            vals = [r[key][metric] for r in runs if r[key].get(metric) is not None]
            if vals:
                out[key][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="./output.csv", help="Path to output.csv")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--mode", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--sampling", action="store_true",
                        help="finite-shot kernel (fast: binomial emulation; cirq: real circuits)")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--psd", action="store_true",
                        help="with --sampling: clip negative eigenvalues of the "
                             "finite-shot training kernel")
    parser.add_argument("--n-qubits", type=int, default=16)
    parser.add_argument("--n-reps", type=int, default=2)
    parser.add_argument("--kernel", default="zz", choices=["zz", "rx"])
    parser.add_argument("--entanglement", default="linear",
                        choices=["linear", "circular", "full"])
    parser.add_argument("--backend", default="fast", choices=["fast", "cirq"])
    parser.add_argument("--alpha", type=float, default=1.0, help="bandwidth when --no-tune")
    parser.add_argument("--C", type=float, default=1.0, help="SVM C when --no-tune")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--Cs", type=float, nargs="+", default=DEFAULT_CS)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="disable signed log transform")
    parser.add_argument("--features", default="paper", choices=["paper", "all"])
    parser.add_argument("--extra-baselines", action="store_true",
                        help="also train a Random Forest on all raw features")
    parser.add_argument("--legacy-scaling", action="store_true",
                        help="original preprocessing: global MinMax before the split, "
                             "PCA on [0,π] features, no log transform")
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce the original protocol = --legacy-scaling "
                             "--keep-duplicates --no-tune --alpha 1 --C 1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.legacy:
        args.legacy_scaling = args.keep_duplicates = args.no_tune = True
        args.alpha, args.C = 1.0, 1.0

    tag = (f"{args.mode}_{args.kernel}_{args.n_qubits}q"
           f"{'_legacy' if args.legacy else ''}{'_shots' if args.sampling else ''}")
    out_dir = Path(args.out_dir or os.path.join("outputs", tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  QUANTUM NETWORK DEFENSE — CIC-DDoS2019 Pipeline")
    print(f"  Kernel: {args.kernel} | protocol: {'LEGACY (original)' if args.legacy else 'improved'}")
    print("=" * 72)

    runs = []
    for i, seed in enumerate(args.seeds):
        r = run_once(args, seed, verbose=not args.quiet)
        if i == 0:
            print(f"      Plot: {plot_run(r, args, out_dir)}")
        r.pop("_plot")
        runs.append(r)

    summary = summarize(runs)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runs": runs}, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("\n" + "=" * 72)
    print(f"  RESULTS SUMMARY  ({args.mode}, {len(runs)} seed(s): {args.seeds})")
    print("=" * 72)
    print(f"  {'Method':<16}{'Accuracy':>16}{'Macro-F1':>16}{'AUC':>16}{'FalseAlarm':>14}")
    for key, name in [("qsvm", f"QSVM ({args.kernel})"), ("rbf_svm", "RBF-SVM"),
                      ("random_forest", "Random Forest")]:
        if key not in summary:
            continue
        s = summary[key]
        fmt = lambda m: (f"{s[m]['mean']*100:6.2f} ± {s[m]['std']*100:4.2f}" if m in s else "—")
        print(f"  {name:<16}{fmt('accuracy'):>16}{fmt('macro_f1'):>16}{fmt('auc'):>16}"
              f"{fmt('false_alarm_rate'):>14}")
    p = [r["mcnemar_qsvm_vs_rbf"]["p_value"] for r in runs]
    print(f"\n  McNemar p-value (QSVM vs RBF) per seed: {', '.join(f'{v:.3g}' for v in p)}")
    print(f"  Saved: {out_dir / 'results.json'}")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    main()
