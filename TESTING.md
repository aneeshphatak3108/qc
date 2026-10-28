# Testing guide — Quantum Network Defense (improved pipeline)

Every command below runs from the project root, using the project's virtualenv.
Approximate runtimes were measured on the development machine (8 cores, 15 GB RAM).

```bash
cd ~/Desktop/qcnn
source venv/bin/activate          # or prefix every command with ./venv/bin/
```

## Project layout after the improvements

| File | Purpose |
|---|---|
| `data_loader.py` | Loads `output.csv`: removes duplicates, uses seeded sampling, returns **unscaled** features |
| `preprocessing.py` | Preprocessing fitted on the training split only (signed log, then StandardScaler, then PCA, then [0, π]), plus the original "legacy" preprocessing |
| `qsvm.py` | ZZ / Rx feature maps (Cirq), fast exact statevector engine, quantum kernel, QSVM |
| `evaluation.py` | Metrics, McNemar test, cross-validated tuning, kernel diagnostics |
| `main.py` | CLI pipeline (all original flags still work) |
| `benchmark.py` | Every experiment used in the report (resumable) |
| `make_figures.py` | Builds the report's figures and tables from `outputs/benchmark/` |
| `tests/test_core.py` | Unit tests |
| `legacy/` | The original, untouched `data_loader.py`, `qsvm.py`, `main.py` |
| `report/` | LaTeX source of the PDF report |

---

## 1. Unit tests (≈15 s)

```bash
python -m unittest discover -s tests -v
```

Expected: `Ran 19 tests ... OK`. The tests check that:

* the fast engine reproduces Cirq statevectors and kernels (linear, circular and full entanglement; 2–3 reps; α ≠ 1; Rx map)
* the new kernel equals the **original** `legacy/qsvm.py` kernel value for α = 1
* kernels are symmetric and PSD, with a unit diagonal
* shot-noise emulation is unbiased, and `--psd` projection makes a noisy kernel PSD
* the bandwidth α controls kernel concentration
* preprocessing statistics come only from the training data
* the loader is reproducible, removes duplicates (keyed on the task's label), gives balanced classes and finds 6 multiclass labels
* alignment, geometric difference and McNemar behave correctly on known cases
* QSVM learns a toy problem end to end

## 2. Run the original code, unchanged (≈30 s)

The original pipeline is kept in `legacy/`. It simulates two circuits per kernel entry, so use a tiny run:

```bash
python legacy/main.py --samples 100 --n-qubits 4 --n-reps 2
```

It reports `N: 42`, not 100. The original loader draws 3 × `--samples` random rows *before* balancing, and BENIGN is only 6.8% of the file. With `--samples 2000` the same effect gives the paper's N = 822. The improved loader fixes this.

Note: this overwrites `outputs/qnd_results.png`, as before.

## 3. Original protocol on the new engine (≈5 s)

`--legacy` reproduces the original methodology: duplicates kept, global MinMax before the split, α = 1, C = 1, untuned RBF baseline. It runs on the fast engine, which gives numerically identical kernels.

```bash
python main.py --legacy --n-qubits 8 --n-reps 2
```

Look for the line `test rows with an exact copy in train: ~30%`. This is the duplicate leakage.

## 4. Improved pipeline

### Binary, paper's ZZ circuit (8 qubits, 2 reps) (≈15 s)
```bash
python main.py --n-qubits 8 --n-reps 2
```

### Binary, default 16 qubits, 2 reps (≈2–3 min)
```bash
python main.py
```

### Multiclass, paper's multiclass circuit (16 qubits, 1 rep) (≈2–3 min)
```bash
python main.py --mode multiclass --n-qubits 16 --n-reps 1
```

### Multiclass, 8 qubits (≈20 s)
```bash
python main.py --mode multiclass --n-qubits 8
```

### Mean ± std over 5 seeds, with a Random Forest reference (≈1–2 min at 8 qubits)
```bash
python main.py --n-qubits 8 --seeds 0 1 2 3 4 --extra-baselines
python main.py --mode multiclass --n-qubits 8 --seeds 0 1 2 3 4 --extra-baselines
```

Each run writes `outputs/<mode>_<kernel>_<n>q/results.json` (all metrics, per seed) and `results_seed<k>.png` (confusion matrices, kernel heatmap, ROC, tuning curve, F1 bars). `--out-dir DIR` overrides the location.

## 5. The paper's three methods through one pipeline

```bash
# Method 1 — Analytical (separable Rx) kernel
python main.py --kernel rx --n-qubits 16
# Method 2 — Sampling kernel (1024 shots; fast binomial emulation of the circuits)
python main.py --kernel rx --n-qubits 16 --sampling --shots 1024
# Method 3 — ZZ entangling kernel
python main.py --kernel zz --n-qubits 8 --n-reps 2
```

Add `--mode multiclass` to any of these.

## 6. Cross-check the fast engine against Cirq

The following two commands must print **identical** accuracies and kernel statistics:

```bash
python main.py --n-qubits 8 --samples 500 --no-tune --alpha 0.4 --C 10 --backend fast  --out-dir outputs/check_fast
python main.py --n-qubits 8 --samples 500 --no-tune --alpha 0.4 --C 10 --backend cirq  --out-dir outputs/check_cirq
```

Real circuit sampling with Cirq (the original method 2: each kernel entry is a measured circuit) works too, but it is slow, so keep it tiny (≈2 min):

```bash
python main.py --backend cirq --sampling --shots 256 --samples 100 --n-qubits 4 --no-tune --alpha 0.4
```

## 7. Ablation: turn each fix on one at a time (8 qubits, binary)

```bash
python main.py --n-qubits 8 --seeds 0 1 2 --legacy                         --out-dir outputs/abl0
python main.py --n-qubits 8 --seeds 0 1 2 --legacy-scaling --no-tune       --out-dir outputs/abl1   # + dedup
python main.py --n-qubits 8 --seeds 0 1 2 --no-tune --no-log               --out-dir outputs/abl2   # + train-only scaling
python main.py --n-qubits 8 --seeds 0 1 2 --no-tune                        --out-dir outputs/abl3   # + log transform
python main.py --n-qubits 8 --seeds 0 1 2                                  --out-dir outputs/abl4   # + CV tuning
```

## 8. Other useful knobs

```bash
python main.py --n-qubits 8 --entanglement full --n-reps 3       # different circuit
python main.py --n-qubits 8 --alphas 0.05 0.1 0.2 --Cs 1 10      # custom tuning grid
python main.py --n-qubits 8 --no-tune --alpha 1.0 --C 1.0        # fixed hyperparameters
python main.py --n-qubits 8 --features all                       # all CICFlowMeter columns
python main.py --n-qubits 8 --keep-duplicates                    # keep duplicate rows
python main.py --help                                            # full option list
```

## 9. Regenerate all report results, figures and the PDF

```bash
python benchmark.py                 # every experiment (≈1.5–2 h; resumable — re-run to continue)
python benchmark.py --quick         # 1 seed, reduced grids, to outputs/benchmark_quick (≈10 min)
python benchmark.py --only speed headline ablation   # selected experiments
python make_figures.py              # → report/figures/*.pdf|png, report/tables/*.tex
cd report && pdflatex report.tex && pdflatex report.tex && cd ..   # → report/report.pdf
```

To rerun one experiment from scratch, delete its folder first, e.g. `rm -rf outputs/benchmark/shots`.
