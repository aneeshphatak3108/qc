"""
Main Pipeline — Quantum Network Defense (CIC-DDoS2019)
=======================================================
Reads output.csv directly — no cic_loader dependency.

Usage:
  python main.py --csv ./output.csv --samples 2000
  python main.py --csv ./output.csv --samples 1000 --n-qubits 12 --n-reps 1
  python main.py --csv ./output.csv --samples 500  --mode multiclass
"""

import argparse
import time
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)


def run_pipeline(
    csv_path: str = "./output.csv",
    max_samples: int = 2000,
    mode: str = "binary",
    use_sampling: bool = False,
    n_qubits: int = 16,
    n_reps: int = 2,
):
    from data_loader import load_csv
    from qsvm import QSVM

    print("=" * 68)
    print("  QUANTUM NETWORK DEFENSE — CIC-DDoS2019 Pipeline")
    print("  Feature map: ZZ entangling (Havlíček et al., 2019)")
    print("=" * 68)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {csv_path} (max={max_samples:,})...")
    t0 = time.time()
    X, y, label_names = load_csv(
        csv_path=csv_path, mode=mode,
        max_samples=max_samples, balance=True, verbose=True
    )
    print(f"      Loaded in {time.time()-t0:.2f}s | X={X.shape} | classes={label_names}")

    # ── 2. Train/test split ────────────────────────────────────────────────────
    print(f"\n[2/5] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train[:3000]
    y_train = y_train[:3000]
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. QSVM (ZZ feature map) ──────────────────────────────────────────────
    print(f"\n[3/5] Training QSVM (ZZ feature map, {n_qubits} qubits, {n_reps} reps)...")
    print(f"      Entangled kernel — full statevector simulation required.")
    qsvm = QSVM(n_qubits=n_qubits, n_reps=n_reps, C=1.0, use_sampling=use_sampling)

    t_q0 = time.time()
    qsvm.fit(X_train, y_train)
    t_q_train = time.time() - t_q0

    t_q1 = time.time()
    y_pred_q  = qsvm.predict(X_test)
    y_proba_q = qsvm.predict_proba(X_test)
    t_q_infer = time.time() - t_q1

    acc_q    = accuracy_score(y_test, y_pred_q)
    cm_q     = confusion_matrix(y_test, y_pred_q)
    target_names = [label_names.get(i, f"Class {i}") for i in sorted(label_names.keys())]
    report_q = classification_report(y_test, y_pred_q, target_names=target_names)

    print(f"\n      QSVM accuracy:  {acc_q*100:.2f}%")
    print(f"      Train time:     {t_q_train:.2f}s")
    print(f"      Inference time: {t_q_infer:.2f}s")
    print(f"\n      Classification Report:\n{report_q}")

    # ── 4. Classical SVM baseline ──────────────────────────────────────────────
    print(f"\n[4/5] Classical SVM (RBF kernel, same PCA-reduced features)...")
    # Use same PCA projection as QSVM for fair comparison
    X_train_pca = qsvm.pca.transform(X_train)
    X_test_pca  = qsvm.pca.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_pca)
    X_test_s  = scaler.transform(X_test_pca)

    csvm = SVC(kernel="rbf", C=1.0, probability=True)
    t_c0 = time.time()
    csvm.fit(X_train_s, y_train)
    t_c_train = time.time() - t_c0

    t_c1 = time.time()
    y_pred_c  = csvm.predict(X_test_s)
    y_proba_c = csvm.predict_proba(X_test_s)
    t_c_infer = time.time() - t_c1

    acc_c = accuracy_score(y_test, y_pred_c)
    cm_c  = confusion_matrix(y_test, y_pred_c)
    report_c = classification_report(y_test, y_pred_c, target_names=target_names)

    print(f"      Classical SVM accuracy: {acc_c*100:.2f}%")
    print(f"      Train time:             {t_c_train:.2f}s")
    print(f"\n      Classification Report:\n{report_c}")

    # ── 5. Plots ───────────────────────────────────────────────────────────────
    print(f"\n[5/5] Generating plots...")
    n_classes = len(label_names)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Quantum Network Defense — ZZ Feature Map ({mode.title()}, N={len(X):,})",
        fontsize=14, fontweight="bold"
    )

    for ax, cm, title, cmap in [
        (axes[0, 0], cm_q, f"QSVM Confusion Matrix\n(ZZ, {n_qubits}q, {n_reps} reps)", "Blues"),
        (axes[0, 1], cm_c, "Classical SVM Confusion Matrix\n(RBF kernel)", "Oranges"),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax)
        ticks = list(range(n_classes))
        ax.set(xticks=ticks, yticks=ticks,
               xticklabels=target_names, yticklabels=target_names,
               ylabel="True label", xlabel="Predicted label", title=title)
        ax.tick_params(axis='x', rotation=30)
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9 if n_classes > 4 else 12)

    ax_roc = axes[1, 0]
    if mode == "binary":
        for y_proba, label, color in [
            (y_proba_q[:, 1], f"QSVM ZZ ({n_qubits}q)", "#1f77b4"),
            (y_proba_c[:, 1], "Classical SVM (RBF)",    "#ff7f0e"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, lw=2,
                        label=f"{label} (AUC={roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
        ax_roc.set(xlim=[0,1], ylim=[0,1.02],
                   xlabel="False Positive Rate", ylabel="True Positive Rate",
                   title="ROC Curves")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)
    else:
        ax_roc.text(0.5, 0.5, "ROC: binary mode only",
                    ha="center", va="center", transform=ax_roc.transAxes,
                    fontsize=12, color="gray")
        ax_roc.set_title("ROC Curves")

    ax_bar = axes[1, 1]
    methods = [f"QSVM\n(ZZ, {n_qubits}q)", "SVM\n(Classical RBF)"]
    accs    = [acc_q * 100, acc_c * 100]
    bars    = ax_bar.bar(methods, accs, color=["#1f77b4", "#ff7f0e"],
                         width=0.4, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax_bar.set(ylabel="Accuracy (%)", title="Accuracy Comparison",
               ylim=[max(0, min(accs) - 10), 103])
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = "outputs/qnd_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"      Saved: {out_png}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RESULTS SUMMARY")
    print("=" * 68)
    print(f"  Dataset:      output.csv (CIC-DDoS2019)")
    print(f"  Mode:         {mode}")
    print(f"  N:            {len(X):,}  ({len(X_train):,} train / {len(X_test):,} test)")
    print(f"  Feature map:  ZZ entangling ({n_qubits} qubits, {n_reps} reps)")
    print(f"  PCA variance: {qsvm.pca.explained_variance_ratio_.sum()*100:.1f}% retained")
    print()
    print(f"  {'Method':<28} {'Accuracy':>10} {'Train(s)':>10} {'Infer(s)':>10}")
    print(f"  {'-'*60}")
    print(f"  {'QSVM (ZZ entangling)':<28} {acc_q*100:>9.2f}% {t_q_train:>10.2f} {t_q_infer:>10.2f}")
    print(f"  {'SVM (Classical RBF)':<28} {acc_c*100:>9.2f}% {t_c_train:>10.2f} {t_c_infer:>10.2f}")
    print("=" * 68)

    return {"qsvm_acc": acc_q, "csvm_acc": acc_c}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="./output.csv",
                        help="Path to output.csv")
    parser.add_argument("--samples",   type=int, default=2000)
    parser.add_argument("--mode",      default="binary",
                        choices=["binary", "multiclass"])
    parser.add_argument("--sampling",  action="store_true")
    parser.add_argument("--n-qubits",  type=int, default=16)
    parser.add_argument("--n-reps",    type=int, default=2)
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        max_samples=args.samples,
        mode=args.mode,
        use_sampling=args.sampling,
        n_qubits=args.n_qubits,
        n_reps=args.n_reps,
    )