"""
Benchmark suite — produces every number and figure used in the report.
=====================================================================

  python benchmark.py                 # everything (≈1–2 h on 8 cores)
  python benchmark.py --only speed headline
  python benchmark.py --quick         # 1 seed, reduced grids (≈10 min)

Each configuration is a call to main.main(...) and writes
outputs/benchmark/<experiment>/<config>/results.json. Finished
configurations are skipped on re-runs, so the suite is resumable.
Afterwards run `python make_figures.py` to build the plots and tables.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import time
from pathlib import Path

import numpy as np

import main as pipeline
from qsvm import QuantumKernel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "benchmark"

S5 = [0, 1, 2, 3, 4]
S3 = [0, 1, 2]

# Circuit settings reported in the paper (Table 3)
PAPER_ZZ = {"binary": ["--n-qubits", "8", "--n-reps", "2"],
            "multiclass": ["--n-qubits", "16", "--n-reps", "1"]}
RX16 = ["--kernel", "rx", "--n-qubits", "16"]


def run(exp: str, name: str, argv, seeds, quick=False):
    out_dir = OUT / exp / name
    if (out_dir / "results.json").exists():
        print(f"  [skip] {exp}/{name}")
        return
    seeds = seeds[:1] if quick else seeds
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline.main(list(map(str, argv)) + ["--quiet", "--extra-baselines",
                      "--seeds", *map(str, seeds), "--out-dir", str(out_dir)])
    (out_dir / "log.txt").write_text(buf.getvalue())
    s = json.load(open(out_dir / "results.json"))["summary"]
    print(f"  [done] {exp}/{name:<40} QSVM {s['qsvm']['accuracy']['mean']*100:6.2f}%  "
          f"RBF {s['rbf_svm']['accuracy']['mean']*100:6.2f}%  ({time.time()-t0:.0f}s)")


# ── Experiments ───────────────────────────────────────────────────────────────

def exp_headline(q):
    """Paper's three methods, original vs improved protocol, both modes."""
    for mode in ["binary", "multiclass"]:
        m = ["--mode", mode]
        run("headline", f"{mode}_zz_legacy", m + PAPER_ZZ[mode] + ["--legacy"], S5, q)
        run("headline", f"{mode}_zz_improved", m + PAPER_ZZ[mode], S5, q)
        run("headline", f"{mode}_rx_legacy", m + RX16 + ["--legacy"], S5, q)
        run("headline", f"{mode}_rx_improved", m + RX16, S5, q)
        run("headline", f"{mode}_rxshots_legacy", m + RX16 + ["--legacy", "--sampling"], S5, q)
        run("headline", f"{mode}_rxshots_improved", m + RX16 + ["--sampling"], S5, q)
        run("headline", f"{mode}_zz16r2_improved",
            m + ["--n-qubits", "16", "--n-reps", "2"], S5, q)


def exp_ablation(q):
    """Add the fixes one at a time to the ZZ pipeline."""
    steps = [
        ("0_legacy", ["--legacy"]),
        ("1_dedup", ["--legacy-scaling", "--no-tune"]),
        ("2_trainonly_nolog", ["--no-tune", "--no-log"]),
        ("3_trainonly_log", ["--no-tune"]),
        ("4_tuned", []),
    ]
    for mode in ["binary", "multiclass"]:
        for name, flags in steps:
            run("ablation", f"{mode}_{name}", ["--mode", mode] + PAPER_ZZ[mode] + flags, S5, q)


def exp_bandwidth(q):
    alphas = [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    for mode in ["binary", "multiclass"]:
        for nq in [8, 16]:
            for a in (alphas[::2] if q else alphas):
                run("bandwidth", f"{mode}_{nq}q_a{a}",
                    ["--mode", mode, "--n-qubits", nq, "--alphas", a], S3, q)


def exp_qubits(q):
    for mode in ["binary", "multiclass"]:
        for nq in ([4, 8, 16] if q else [2, 4, 6, 8, 10, 12, 14, 16]):
            run("qubits", f"{mode}_zz_{nq}q", ["--mode", mode, "--n-qubits", nq], S3, q)
            run("qubits", f"{mode}_rx_{nq}q", ["--mode", mode, "--kernel", "rx",
                                                "--n-qubits", nq], S3, q)


def exp_shots(q):
    for mode in ["binary", "multiclass"]:
        for shots in [16, 64, 256, 1024, 4096]:
            run("shots", f"{mode}_8q_s{shots}", ["--mode", mode, "--n-qubits", 8,
                                                 "--sampling", "--shots", shots], S3, q)
        run("shots", f"{mode}_8q_exact", ["--mode", mode, "--n-qubits", 8], S3, q)


def exp_architecture(q):
    for mode in ["binary", "multiclass"]:
        for reps in [1, 2, 3]:
            for ent in ["linear", "circular", "full"]:
                run("architecture", f"{mode}_r{reps}_{ent}",
                    ["--mode", mode, "--n-qubits", 8, "--n-reps", reps,
                     "--entanglement", ent], S3, q)


def exp_features(q):
    for mode in ["binary", "multiclass"]:
        for fs in ["paper", "all"]:
            run("features", f"{mode}_{fs}", ["--mode", mode, "--n-qubits", 8,
                                              "--features", fs], S3, q)


def exp_samples(q):
    for mode in ["binary", "multiclass"]:
        for n in [250, 500, 1000, 2000, 4000]:
            run("samples", f"{mode}_n{n}", ["--mode", mode, "--n-qubits", 8,
                                             "--samples", n], S3, q)


def exp_speed(q):
    """Wall-clock cost of the original vs new kernel computation."""
    path = OUT / "speed" / "speed.json"
    if path.exists():
        print("  [skip] speed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("legacy_qsvm", ROOT / "legacy" / "qsvm.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    rng = np.random.default_rng(0)
    res = {"n_train": 1600, "n_test": 400, "rows": []}
    # entries the original code evaluated: symmetric train + test twice
    # (predict and predict_proba each rebuilt the test kernel)
    n_tr, n_te = 1600, 400
    legacy_entries = n_tr * (n_tr + 1) // 2 + 2 * n_te * n_tr
    for nq in [4, 8, 12, 16]:
        X = rng.uniform(0, np.pi, (20, nq))
        row = {"n_qubits": nq}
        lk = legacy.QuantumKernel(nq, 2)
        t = time.time()
        for i in range(10):
            lk.compute_kernel_value(X[i], X[i + 10])
        row["legacy_per_entry_s"] = (time.time() - t) / 10
        row["legacy_total_s_est"] = row["legacy_per_entry_s"] * legacy_entries
        if nq <= 8:
            lks = legacy.QuantumKernel(nq, 2, shots=1024, use_sampling=True)
            t = time.time()
            for i in range(5):
                lks.compute_kernel_value(X[i], X[i + 10])
            row["legacy_sampling_per_entry_s"] = (time.time() - t) / 5
            row["legacy_sampling_total_s_est"] = row["legacy_sampling_per_entry_s"] * legacy_entries
        ck = QuantumKernel(nq, 2, backend="cirq")
        t = time.time()
        ck.states(X)
        per_state = (time.time() - t) / len(X)
        A_tr, A_te = rng.uniform(0, np.pi, (n_tr, nq)), rng.uniform(0, np.pi, (n_te, nq))
        fk = QuantumKernel(nq, 2)
        t = time.time()
        fk.states(A_tr), fk.states(A_te)
        row["fast_states_s"] = time.time() - t
        fk.compute_kernel_matrix(A_tr, verbose=False)
        fk.compute_kernel_matrix(A_te, A_tr, verbose=False)
        row["fast_total_s"] = time.time() - t
        row["gram_s"] = row["fast_total_s"] - row["fast_states_s"]
        # same Gram products, but statevectors from Cirq (one per sample)
        row["cirq_cached_total_s_est"] = per_state * (n_tr + n_te) + row["gram_s"]
        res["rows"].append(row)
        print(f"  [speed] {nq:2d}q legacy≈{row['legacy_total_s_est']/3600:8.2f} h | "
              f"cirq-cached≈{row['cirq_cached_total_s_est']:7.1f} s | fast {row['fast_total_s']:6.2f} s")
    res["legacy_entries"] = legacy_entries
    json.dump(res, open(path, "w"), indent=2)


EXPERIMENTS = {
    "speed": exp_speed, "headline": exp_headline, "ablation": exp_ablation,
    "bandwidth": exp_bandwidth, "qubits": exp_qubits, "shots": exp_shots,
    "architecture": exp_architecture, "features": exp_features, "samples": exp_samples,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        OUT = ROOT / "outputs" / "benchmark_quick"
    os.chdir(ROOT)
    for name in a.only:
        print(f"\n=== {name} ===")
        EXPERIMENTS[name](a.quick)
