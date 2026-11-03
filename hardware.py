"""
QSVM on real IBM quantum hardware (8 qubits): comparative analysis
=================================================================

Estimates the ZZ quantum kernel on an IBM Quantum processor and compares it,
entry by entry and through the resulting classifier, with the same kernel
computed every other way:

  exact            noiseless statevector (fast engine; = Qiskit statevector)
  shots            exact kernel + binomial shot noise (same shot count)
  noisy_sim        Aer simulation with the device's noise model
                   (live calibration of the chosen backend, or a fake
                   backend snapshot when no token is set)
  hardware         the real QPU
  *_mitigated      device-noise mitigation by normalisation:
                   K̃ᵢⱼ = Kᵢⱼ / √(Kᵢᵢ Kⱼⱼ) using the *measured* diagonal
                   (ideally 1), then PSD projection of the training kernel
                   (Hubregtsen et al., PRA 106, 042431, 2022)

Each kernel entry K(a, b) = |⟨φ(a)|φ(b)⟩|² is the probability of measuring
|0…0⟩ after U(b) followed by U†(a) ("compute–uncompute"), with U the
Qiskit zz_feature_map bound to α·θ. With N_train training and N_test test
flows the job contains
    N_train(N_train-1)/2   training pairs (symmetric, i < j)
  + N_train + N_test       diagonal circuits K(x, x)  (noise / mitigation)
  + N_test · N_train       test–train pairs
circuits. Real hardware time is scarce, so the hardware run uses a small,
balanced subset of the corrected binary pipeline (dedup + train-only
preprocessing fitted on the full 1600-flow training split); α and C are
tuned on that full training split by 5-fold CV of the exact kernel.

Usage (Python ≥ 3.10 env with Qiskit, see README):
  ./venv-qiskit/bin/python hardware.py --dry-run          # no QPU: exact/shots/noisy-sim only
  TOKEN=... ./venv-qiskit/bin/python hardware.py          # submit to least-busy IBM QPU
  TOKEN=... ./venv-qiskit/bin/python hardware.py --job-id <id>   # fetch a submitted job
  options: --backend ibm_fez --n-train 40 --n-test 40 --shots 256 --reps 2

The IBM API key is read from the TOKEN environment variable (optionally
INSTANCE for the instance CRN); it is never written to disk.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler import generate_preset_pass_manager
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data_loader import load_csv
from evaluation import (
    centered_alignment, classification_metrics, offdiag_mean, standardize, tune_quantum,
    tune_rbf,
)
from main import print_kernel
from preprocessing import QuantumPreprocessor
from qsvm import QuantumKernel, nearest_psd

ROOT = Path(__file__).resolve().parent
RBF_GAMMAS = ["scale", 0.01, 0.03, 0.1, 0.3, 1.0]


# ── Circuits ─────────────────────────────────────────────────────────────────

def kernel_circuit(a, b, n_qubits, reps, entanglement, alpha, measure=True) -> QuantumCircuit:
    """U†(a)·U(b)|0⟩; P(|0…0⟩) = |⟨φ(a)|φ(b)⟩|²."""
    fm = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    qc = fm.assign_parameters(alpha * np.asarray(b))
    # The barrier stops the transpiler from cancelling U†(a)U(b) against itself.
    # Without it the diagonal circuits U†(x)U(x) compile to almost nothing and
    # measure only readout error, which would make the mitigation meaningless.
    qc.barrier()
    qc.compose(fm.assign_parameters(alpha * np.asarray(a)).inverse(), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def circuit_index(n_tr: int, n_te: int):
    """(kind, i, j) for every circuit, in submission order."""
    idx = [("train", i, j) for i in range(n_tr) for j in range(i + 1, n_tr)]
    idx += [("diag_train", i, i) for i in range(n_tr)]
    idx += [("diag_test", i, i) for i in range(n_te)]
    idx += [("test", i, j) for i in range(n_te) for j in range(n_tr)]
    return idx


def circuit_points(kind, i, j, P):
    """(a, b) of the circuit U†(a)U(b) for one entry of circuit_index."""
    if kind == "train" or kind == "diag_train":
        return P["A_tr"][i], P["A_tr"][j]
    if kind == "diag_test":
        return P["A_te"][i], P["A_te"][i]
    return P["A_te"][i], P["A_tr"][j]                     # test row i, train column j


def assemble(p0: np.ndarray, index, n_tr: int, n_te: int):
    """Measured P(0…0) values → K_train (raw diagonal), K_test, test diagonal."""
    K_tr, K_te, d_te = np.zeros((n_tr, n_tr)), np.zeros((n_te, n_tr)), np.zeros(n_te)
    for v, (kind, i, j) in zip(p0, index):
        if kind == "train":
            K_tr[i, j] = K_tr[j, i] = v
        elif kind == "diag_train":
            K_tr[i, i] = v
        elif kind == "diag_test":
            d_te[i] = v
        else:
            K_te[i, j] = v
    return K_tr, K_te, d_te


def mitigate(K_tr, K_te, d_te):
    """Normalise by the measured diagonal, then PSD-project the training kernel."""
    d_tr = np.clip(np.diag(K_tr), 1e-3, None)
    d_te = np.clip(d_te, 1e-3, None)
    Kt = np.clip(K_tr / np.sqrt(np.outer(d_tr, d_tr)), 0, 1)
    np.fill_diagonal(Kt, 1.0)
    Ks = np.clip(K_te / np.sqrt(np.outer(d_te, d_tr)), 0, 1)
    return nearest_psd(Kt), Ks


def counts_p0(counts: dict, n_qubits: int, shots: int) -> float:
    return counts.get("0" * n_qubits, 0) / shots


# ── Problem: data, preprocessing, hyperparameters ────────────────────────────

def build_problem(args) -> dict:
    X, y, names, _ = load_csv(args.csv, mode="binary", max_samples=args.samples,
                              seed=args.seed, verbose=False)
    target_names = [names[i] for i in sorted(names)]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=args.seed,
                                              stratify=y)
    prep = QuantumPreprocessor(args.n_qubits, seed=args.seed)
    A_tr_full, A_te_full = prep.fit_transform(X_tr), prep.transform(X_te)
    print(f"  Data: {len(y_tr)} train / {len(y_te)} test flows (binary, dedup), "
          f"PCA {args.n_qubits} ({prep.explained_variance*100:.1f}% variance, train only)")

    rng = np.random.default_rng(args.seed)
    pick = lambda yy, n: np.sort(np.concatenate(
        [rng.choice(np.where(yy == c)[0], n // 2, replace=False) for c in (0, 1)]))
    i_tr, i_te = pick(y_tr, args.n_train), pick(y_te, args.n_test)

    # α and C: 5-fold CV of the exact kernel on the full training split
    if args.alpha is None or args.C is None:
        kfn = lambda a: QuantumKernel(args.n_qubits, args.reps, entanglement=args.entanglement,
                                      alpha=a).compute_kernel_matrix(A_tr_full, verbose=False)
        alpha, C, K_full, _ = tune_quantum(kfn, y_tr, [0.05, 0.1, 0.2, 0.4, 1.0],
                                           [0.1, 1.0, 10.0, 100.0], seed=args.seed,
                                           verbose=False)
    alpha = args.alpha if args.alpha is not None else alpha
    C = args.C if args.C is not None else C
    print(f"  Hyperparameters (CV on the full training split): α={alpha}, C={C}")

    # Reference: exact QSVM and RBF-SVM on the full split
    qk = QuantumKernel(args.n_qubits, args.reps, entanglement=args.entanglement, alpha=alpha)
    svc = SVC(kernel="precomputed", C=C).fit(qk.compute_kernel_matrix(A_tr_full, verbose=False),
                                              y_tr)
    acc_full = accuracy_score(y_te, svc.predict(qk.compute_kernel_matrix(A_te_full, A_tr_full,
                                                                         verbose=False)))
    print(f"  Reference, exact QSVM on the full {len(y_tr)}/{len(y_te)} split: "
          f"accuracy {acc_full*100:.2f}%")

    Z_tr, Z_te = standardize(prep.transform_pca(X_tr[i_tr]), prep.transform_pca(X_te[i_te]))
    return dict(A_tr=A_tr_full[i_tr], A_te=A_te_full[i_te], y_tr=y_tr[i_tr], y_te=y_te[i_te],
                Z_tr=Z_tr, Z_te=Z_te, alpha=float(alpha), C=float(C),
                target_names=target_names, full_split_exact_accuracy=float(acc_full),
                pca_variance=prep.explained_variance)


# ── Evaluation of one kernel source ──────────────────────────────────────────

def evaluate(name, K_tr, K_te, P, exact, d_te=None):
    y_tr, y_te = P["y_tr"], P["y_te"]
    svc = SVC(kernel="precomputed", C=P["C"]).fit(K_tr, y_tr)
    pred = svc.predict(K_te)
    m = classification_metrics(y_te, pred, svc.decision_function(K_te), benign_label=0)
    E_tr, E_te = exact
    iu = np.triu_indices(len(K_tr), 1)
    off = K_tr[iu]
    r = {
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "auc": m["auc"],
        "false_alarm_rate": m["false_alarm_rate"], "detection_rate": m["detection_rate"],
        "confusion_matrix": m["confusion_matrix"],
        "agreement_with_exact_predictions": None,
        "n_support": int(len(svc.support_)),
        "mae_train": float(np.mean(np.abs(off - E_tr[iu]))),
        "mae_test": float(np.mean(np.abs(K_te - E_te))),
        "max_abs_err": float(max(np.abs(off - E_tr[iu]).max(), np.abs(K_te - E_te).max())),
        "rel_frobenius_err": float(np.linalg.norm(np.concatenate([off - E_tr[iu],
                                                                  (K_te - E_te).ravel()]))
                                   / np.linalg.norm(np.concatenate([E_tr[iu], E_te.ravel()]))),
        "pearson_r": float(np.corrcoef(np.concatenate([off, K_te.ravel()]),
                                       np.concatenate([E_tr[iu], E_te.ravel()]))[0, 1]),
        "diag_mean_train": float(np.mean(np.diag(K_tr))),
        "diag_mean_test": float(np.mean(d_te)) if d_te is not None else None,
        "offdiag_mean": offdiag_mean(K_tr),
        "alignment": centered_alignment(K_tr, y_tr),
        "min_eigenvalue": float(np.linalg.eigvalsh(K_tr).min()),
        "_pred": pred,
    }
    return r


# ── Backends ─────────────────────────────────────────────────────────────────

def get_service(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("TOKEN")
    if not token:
        return None
    try:
        return QiskitRuntimeService(channel=args.channel, token=token,
                                    instance=os.environ.get("INSTANCE") or args.instance)
    except Exception as e:
        if "No matching instances" in str(e):
            raise SystemExit(
                "  The API key was accepted, but its IBM Cloud account has no Qiskit Runtime "
                "instance.\n  Create one (Open plan is free) at quantum.cloud.ibm.com → "
                "Instances → Create instance,\n  or set INSTANCE=<instance CRN>, then rerun.")
        if "blocked" in str(e.__cause__ or e):
            raise SystemExit(
                "  IBM Cloud refused the login: the account that owns this API key is blocked "
                "(IAM error 400).\n  This is an account status, not a code problem: log in at "
                "cloud.ibm.com to see why,\n  or create a key in an account that is active.")
        raise


def pick_backend(args, service):
    if service is None:
        from qiskit_ibm_runtime import fake_provider
        name = args.fake_backend
        cls = getattr(fake_provider, "Fake" + name.replace("ibm_", "").capitalize())
        return cls()
    if args.backend:
        return service.backend(args.backend)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=args.n_qubits)


def transpile_all(circuits, backend, seed):
    """Choose one layout on the first circuit, then pin it for every circuit."""
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=seed)
    first = pm.run(circuits[0])
    layout = first.layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      initial_layout=layout, seed_transpiler=seed)
    t0 = time.time()
    out = pm.run(circuits)
    print(f"  Transpiled {len(out)} circuits in {time.time()-t0:.0f}s onto physical qubits "
          f"{layout}")
    return out, layout


def circuit_stats(tqc, backend, layout):
    ops = [dict(c.count_ops()) for c in tqc]
    two_q = [sum(v for k, v in o.items() if k in ("cz", "ecr", "cx")) for o in ops]
    s = {"depth_mean": float(np.mean([c.depth() for c in tqc])),
         "depth_max": int(max(c.depth() for c in tqc)),
         "two_qubit_gates_mean": float(np.mean(two_q)),
         "two_qubit_gates_max": int(max(two_q)),
         "physical_qubits": [int(q) for q in layout]}
    try:
        t = backend.target
        g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
        e2 = [p.error for qargs, p in t[g2].items()
              if p is not None and p.error is not None and set(qargs) <= set(layout)]
        e_ro = [t["measure"][(q,)].error for q in layout
                if t["measure"].get((q,)) is not None]
        s.update({"two_qubit_gate": g2, "median_2q_error_on_layout": float(np.median(e2)),
                  "median_readout_error_on_layout": float(np.median(e_ro))})
    except Exception as e:                                     # calibration may be missing
        s["calibration_note"] = repr(e)
    return s


def run_noisy_sim(tqc, backend, shots, seed, n_qubits):
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    t0 = time.time()
    res = sim.run(tqc, shots=shots).result()
    print(f"  Noisy simulation ({backend.name} noise model) of {len(tqc)} circuits "
          f"in {time.time()-t0:.0f}s")
    return np.array([counts_p0(res.get_counts(i), n_qubits, shots) for i in range(len(tqc))])


def run_hardware(tqc, backend, args, out_dir, meta):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = args.shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.max_execution_time = args.max_seconds
    if args.twirl:
        sampler.options.twirling.enable_gates = True
    job = sampler.run(tqc)
    meta = {**meta, "job_id": job.job_id(), "backend": backend.name,
            "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "job.json").write_text(json.dumps(meta, indent=2))
    print(f"  Submitted job {job.job_id()} to {backend.name}; saved to {out_dir/'job.json'}")
    print(f"  (if interrupted, resume with: --job-id {job.job_id()})")
    return job


def wait_job(job):
    last = None
    while True:
        st = str(job.status())
        if st != last:
            print(f"  [{time.strftime('%H:%M:%S')}] job status: {st}", flush=True)
            last = st
        if st in ("DONE", "ERROR", "CANCELLED", "JobStatus.DONE", "JobStatus.ERROR",
                  "JobStatus.CANCELLED"):
            break
        time.sleep(30)
    return job.result()


# ── Report ───────────────────────────────────────────────────────────────────

ORDER = ["exact", "shots", "noisy_sim", "noisy_sim_mitigated", "hardware", "hardware_mitigated"]
LABEL = {"exact": "Exact (statevector)", "shots": "Ideal + shot noise",
         "noisy_sim": "Noisy simulator", "noisy_sim_mitigated": "Noisy sim, mitigated",
         "hardware": "IBM hardware (raw)", "hardware_mitigated": "IBM hardware, mitigated",
         "rbf": "Classical RBF-SVM"}
COLOR = {"exact": "#2a78d6", "shots": "#9ca3af", "noisy_sim": "#eb6834",
         "noisy_sim_mitigated": "#eda100", "hardware": "#e87ba4",
         "hardware_mitigated": "#1baf7a", "rbf": "#4b5563"}


def print_table(results, P, exact_pred):
    print("\n" + "=" * 100)
    print(f"  COMPARISON — {len(P['y_tr'])} train / {len(P['y_te'])} test flows, "
          f"α={P['alpha']}, C={P['C']}")
    print("=" * 100)
    print(f"  {'Kernel source':<26}{'Acc %':>7}{'F1':>7}{'AUC':>7}{'=exact':>8}"
          f"{'MAE':>8}{'max|Δ|':>8}{'r':>7}{'K(x,x)':>8}{'offdiag':>8}{'align':>7}{'λmin':>10}")
    for k in ORDER + ["rbf"]:
        if k not in results:
            continue
        r = results[k]
        agree = (f"{np.mean(r['_pred'] == exact_pred)*100:6.0f}%"
                 if "_pred" in r else "     —")
        f = lambda key, fmt: (fmt.format(r[key]) if r.get(key) is not None else "—")
        print(f"  {LABEL[k]:<26}{r['accuracy']*100:7.1f}{r['macro_f1']:7.3f}"
              f"{f('auc', '{:.3f}'):>7}{agree:>8}{f('mae_test', '{:.4f}'):>8}"
              f"{f('max_abs_err', '{:.3f}'):>8}{f('pearson_r', '{:.3f}'):>7}"
              f"{f('diag_mean_train', '{:.3f}'):>8}{f('offdiag_mean', '{:.3f}'):>8}"
              f"{f('alignment', '{:.3f}'):>7}{f('min_eigenvalue', '{:.2e}'):>10}")
    print("  MAE = mean |K − K_exact| over test–train entries; r = Pearson correlation with the "
          "exact kernel;\n  K(x,x) = mean measured diagonal (ideal 1); =exact = test "
          "predictions identical to the exact QSVM.")


def plot(results, kernels, P, out_dir, title):
    srcs = [k for k in ORDER if k in kernels]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    order = np.argsort(P["y_tr"], kind="stable")
    show = [s for s in ["exact", "noisy_sim", "hardware", "hardware_mitigated"] if s in kernels]
    if len(show) < 4:
        show += [s for s in srcs if s not in show][:4 - len(show)]
    for ax, s in zip(axes[0], show):
        im = ax.imshow(kernels[s][0][np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{LABEL[s]}\nK_train (sorted by class)")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[0][len(show):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)

    E_te = kernels["exact"][1].ravel()
    ax = axes[1, 0]
    for s in ["shots", "noisy_sim", "hardware", "hardware_mitigated"]:
        c = COLOR[s]
        if s in kernels:
            ax.scatter(E_te, kernels[s][1].ravel(), s=4, alpha=0.4, color=c, label=LABEL[s])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="exact kernel value", ylabel="estimated kernel value",
           title="Test–train kernel entries vs exact")
    ax.legend(markerscale=3, fontsize=8)

    ax = axes[1, 1]
    for s in ["noisy_sim", "hardware"]:
        c = COLOR[s]
        if s in kernels:
            d = np.concatenate([np.diag(kernels[s][0]), kernels[s][2]])
            ax.hist(d, bins=20, range=(0, 1), alpha=0.6, color=c, label=LABEL[s])
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set(xlabel="measured K(x, x)  (ideal = 1)", ylabel="circuits",
           title="Diagonal: fidelity of U†(x)U(x)")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    names = [k for k in ORDER + ["rbf"] if k in results]
    vals = [results[k]["accuracy"] * 100 for k in names]
    bars = ax.bar(range(len(names)), vals, color=[COLOR[k] for k in names], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9,
                color="#374151")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABEL[k] for k in names], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="test accuracy (%)", ylim=[max(0, min(vals) - 15), 102],
           title="Classifier accuracy per kernel source")

    ax = axes[1, 3]
    errs = [results[k]["mae_test"] for k in srcs if k != "exact"]
    ax.bar(range(len(errs)), errs, color=[COLOR[k] for k in srcs if k != "exact"], width=0.6)
    ax.set_xticks(range(len(errs)))
    ax.set_xticklabels([LABEL[k] for k in srcs if k != "exact"], rotation=30, ha="right",
                       fontsize=8)
    ax.set(ylabel="mean |K − K_exact|", title="Kernel estimation error")
    path = out_dir / "hardware_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT / "output.csv"))
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--entanglement", default="linear", choices=["linear", "circular", "full"])
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=None, help="default: CV-tuned")
    ap.add_argument("--C", type=float, default=None, help="default: CV-tuned")
    ap.add_argument("--backend", default=None, help="IBM backend (default: least busy)")
    ap.add_argument("--fake-backend", default="fez",
                    help="noise-model snapshot used when TOKEN is not set (fez, torino, …)")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--dry-run", action="store_true", help="never submit to hardware")
    ap.add_argument("--job-id", default=None, help="fetch an already submitted job")
    ap.add_argument("--twirl", action="store_true", help="enable gate twirling")
    ap.add_argument("--max-seconds", type=int, default=600,
                    help="QPU-time cap of the hardware job (max_execution_time)")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # show progress live when piped to tee

    print("=" * 100)
    print(f"  QSVM ON IBM QUANTUM HARDWARE — ZZ map, {args.n_qubits} qubits, reps={args.reps}, "
          f"{args.entanglement}")
    print("=" * 100)
    P = build_problem(args)
    n_tr, n_te, nq = len(P["y_tr"]), len(P["y_te"]), args.n_qubits
    names = P["target_names"]

    service = None if args.dry_run else get_service(args)
    if service is None and not args.dry_run:
        print("  TOKEN is not set → dry run (exact, shot noise and noisy simulation only)")
    backend = pick_backend(args, service)
    on_hw = service is not None
    tag = (f"{backend.name}_{nq}q_r{args.reps}_{n_tr}x{n_te}_s{args.shots}")
    out_dir = Path(args.out_dir or ROOT / "outputs" / "hardware" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = circuit_index(n_tr, n_te)
    circuits = [kernel_circuit(*circuit_points(k, i, j, P), nq, args.reps, args.entanglement,
                               P["alpha"]) for k, i, j in index]
    print(f"  {len(circuits)} kernel circuits: {n_tr*(n_tr-1)//2} train pairs + "
          f"{n_tr + n_te} diagonal + {n_te*n_tr} test–train, {args.shots} shots each")
    tqc, layout = transpile_all(circuits, backend, args.seed)
    stats = circuit_stats(tqc, backend, layout)
    print(f"  Backend {backend.name}: depth {stats['depth_mean']:.0f} (max {stats['depth_max']}), "
          f"{stats['two_qubit_gates_mean']:.0f} two-qubit gates per circuit"
          + (f", median {stats['two_qubit_gate']} error {stats['median_2q_error_on_layout']:.2e}, "
             f"readout error {stats['median_readout_error_on_layout']:.2e}"
             if "two_qubit_gate" in stats else ""))
    est = len(tqc) * args.shots * 250e-6
    print(f"  Rough QPU-time estimate: {est/60:.1f} min "
          f"(circuits × shots × 250 µs repetition delay)")

    # ── kernel sources ────────────────────────────────────────────────────
    kernels = {}
    qk = QuantumKernel(nq, args.reps, entanglement=args.entanglement, alpha=P["alpha"])
    E_tr = qk.compute_kernel_matrix(P["A_tr"], verbose=False)
    E_te = qk.compute_kernel_matrix(P["A_te"], P["A_tr"], verbose=False)
    kernels["exact"] = (E_tr, E_te, np.ones(n_te))
    sk = QuantumKernel(nq, args.reps, shots=args.shots, use_sampling=True,
                       entanglement=args.entanglement, alpha=P["alpha"], seed=args.seed)
    kernels["shots"] = (sk.compute_kernel_matrix(P["A_tr"], verbose=False),
                        sk._shot_noise(E_te, False), np.ones(n_te))
    p0_sim = run_noisy_sim(tqc, backend, args.shots, args.seed, nq)
    kernels["noisy_sim"] = assemble(p0_sim, index, n_tr, n_te)
    kernels["noisy_sim_mitigated"] = (*mitigate(*kernels["noisy_sim"]), kernels["noisy_sim"][2])

    hw_meta = None
    if on_hw:
        meta = {"args": {k: v for k, v in vars(args).items()}, "alpha": P["alpha"], "C": P["C"],
                "layout": [int(q) for q in layout], "n_circuits": len(tqc)}
        if args.job_id:
            job = service.job(args.job_id)
            print(f"  Fetching job {args.job_id} on {job.backend().name}")
        else:
            job = run_hardware(tqc, backend, args, out_dir, meta)
        result = wait_job(job)
        p0_hw = np.array([counts_p0(result[i].data.meas.get_counts(), nq, args.shots)
                          for i in range(len(tqc))])
        kernels["hardware"] = assemble(p0_hw, index, n_tr, n_te)
        kernels["hardware_mitigated"] = (*mitigate(*kernels["hardware"]),
                                         kernels["hardware"][2])
        hw_meta = {"job_id": job.job_id(), "backend": job.backend().name}
        try:
            hw_meta["usage_seconds"] = job.usage()
            hw_meta["metrics"] = job.metrics()
        except Exception as e:
            hw_meta["usage_note"] = repr(e)

    # ── evaluate + print ─────────────────────────────────────────────────
    results = {}
    for s in ORDER:
        if s in kernels:
            results[s] = evaluate(s, kernels[s][0], kernels[s][1], P, (E_tr, E_te),
                                  kernels[s][2])
    rbf, params, _ = tune_rbf(P["Z_tr"], P["y_tr"], [0.1, 1.0, 10.0, 100.0], RBF_GAMMAS,
                              seed=args.seed)
    rp = rbf.predict(P["Z_te"])
    results["rbf"] = {**{k: v for k, v in classification_metrics(
        P["y_te"], rp, rbf.decision_function(P["Z_te"]), 0).items()}, "params": params}

    print("\n  Intermediate results: kernel excerpts (first "
          f"{args.show} training flows) from each source")
    for s in ORDER:
        if s in kernels:
            print_kernel(kernels[s][0][:args.show, :args.show], P["y_tr"][:args.show],
                         P["y_tr"][:args.show], f"[{LABEL[s]}] K_train", args.show, names)
    for s in ["noisy_sim", "hardware"]:
        if s in kernels:
            d = np.concatenate([np.diag(kernels[s][0]), kernels[s][2]])
            print(f"\n      [{LABEL[s]}] measured diagonal K(x,x) (ideal 1): mean {d.mean():.4f}, "
                  f"min {d.min():.4f}, max {d.max():.4f}")

    exact_pred = results["exact"]["_pred"]
    print_table(results, P, exact_pred)
    for s in ORDER:
        if s in results:
            results[s]["agreement_with_exact_predictions"] = float(
                np.mean(results[s].pop("_pred") == exact_pred))

    title = (f"QSVM kernel on {backend.name if on_hw else backend.name + ' (simulated)'} — ZZ, "
             f"{nq} qubits, reps={args.reps}, α={P['alpha']}, {args.shots} shots, "
             f"{n_tr} train / {n_te} test flows")
    fig_path = plot(results, kernels, P, out_dir, title)
    np.savez_compressed(out_dir / "kernels.npz",
                        **{f"{s}_train": kernels[s][0] for s in kernels},
                        **{f"{s}_test": kernels[s][1] for s in kernels},
                        **{f"{s}_test_diag": kernels[s][2] for s in kernels},
                        y_train=P["y_tr"], y_test=P["y_te"], A_train=P["A_tr"], A_test=P["A_te"])
    out = {"args": vars(args), "backend": backend.name, "real_hardware": on_hw,
           "hardware_job": hw_meta, "alpha": P["alpha"], "C": P["C"],
           "n_train": n_tr, "n_test": n_te, "n_circuits": len(tqc),
           "full_split_exact_accuracy": P["full_split_exact_accuracy"],
           "circuits": stats, "results": results}
    (out_dir / "results.json").write_text(json.dumps(out, indent=2, default=lambda o: (
        o.item() if hasattr(o, "item") else str(o))))
    print(f"\n  Saved: {out_dir/'results.json'}, {out_dir/'kernels.npz'}, {fig_path}")
    return out


if __name__ == "__main__":
    main()
