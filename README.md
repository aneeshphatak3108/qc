# Quantum-Kernel SVM for DDoS Detection (CIC-DDoS2019)

Network intrusion detection with **quantum-kernel support vector machines (QSVMs)**, simulated with
Google Cirq. Network flows from CIC-DDoS2019 are classified as **BENIGN vs DDoS** (binary) or into
**6 classes**: BENIGN, LDAP, MSSQL, Portmap, Syn and UDP.

This repository re-evaluates an earlier QSVM project that reported a quantum advantage over a
classical RBF-SVM. It fixes the evaluation and repeats the comparison.

## Key finding

| Task | Protocol | QSVM (ZZ) | RBF-SVM | Significant seeds |
|---|---|---:|---:|---:|
| binary | original | 99.50 ± 0.32 | 97.35 ± 0.25 | 5/5 |
| binary | **corrected** | 99.60 ± 0.41 | 99.55 ± 0.29 | 0/5 |
| multiclass | original | 95.70 ± 0.56 | 88.85 ± 1.12 | 5/5 |
| multiclass | **corrected** | 90.85 ± 0.85 | 91.05 ± 0.95 | 0/5 |

Accuracy in %, mean ± std over 5 seeds. "Significant seeds" counts the seeds in which McNemar's
test gives p < 0.05.

The original protocol had four problems: duplicate rows leaking between train and test (66% of
the file is exact duplicates), a scaler fitted before the split, a linear encoding of heavy-tailed
features, and an untuned classical baseline. Once they are fixed, **the quantum kernel ties the
classical one**, and a Random Forest on all features is slightly better than both.

## Quick start

```bash
python3.8 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./run_tests.sh                       # 27 checks, ~3 min
python main.py --n-qubits 8          # corrected pipeline, binary
python main.py --legacy --n-qubits 8 # original protocol, for comparison
python main.py --mode multiclass --n-qubits 8
```

Each run prints its intermediate results as well as the summary table: the rotation angles fed to
the circuit, excerpts and statistics of the training and test kernel matrices (class-by-class mean
similarity, eigenvalue spectrum), the support vectors, the SVM decision values of the first test
flows, and the RBF kernel on the same flows for comparison (`--show N` sets the excerpt size). It
writes `outputs/<run>/results.json`, a figure, and the full matrices in `kernels_seed<seed>.npz`.

### Other feature maps

```bash
python main.py --n-qubits 8 --kernel pauli --paulis Y ZZ          # any Pauli strings
python main.py --n-qubits 8 --kernel pauli --paulis Z ZZ ZZZ --entanglement full
python main.py --n-qubits 8 --n-reps 3 --data-map cosine           # ZZ with another data map
python feature_map_study.py      # ZZ settings + 8 other maps, both tasks, 3 seeds (~1 h)
```

`--kernel pauli` is a general Pauli feature map (the same circuits as Qiskit's `PauliFeatureMap`;
`tests/test_qiskit.py` checks this). `--data-map` sets the function φ applied to the multi-qubit
terms: `havlicek` (π−xᵢ)(π−xⱼ) (the default), `product`, `cosine` or `exp`.

### Real IBM quantum hardware (8 qubits)

The Qiskit stack needs Python ≥ 3.10, so it lives in a second environment:

```bash
python3.12 -m venv venv-qiskit
./venv-qiskit/bin/pip install -r requirements-qiskit.txt
./venv-qiskit/bin/python hardware.py --dry-run          # exact / shot-noise / noisy-simulator only
export TOKEN=<IBM Cloud API key> INSTANCE=<instance CRN>     # from quantum.cloud.ibm.com
./venv-qiskit/bin/python hardware.py                   # least-busy QPU
./venv-qiskit/bin/python hardware.py --job-id <id>     # fetch a finished job
```

`hardware.py` estimates the kernel with compute–uncompute circuits on the QPU and compares it with
the exact kernel, the exact kernel with shot noise, and an Aer simulation using the device's noise
model. Both noisy kernels are also shown after device-noise mitigation. The script prints kernel
excerpts from every source and a comparison table (accuracy, agreement with the exact QSVM,
kernel error, measured K(x,x), alignment), and saves `outputs/hardware/<run>/`. The job ID is saved
in `job.json` as soon as the job is submitted, so a long queue can be resumed later.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Pipeline and CLI (`python main.py --help`) |
| `data_loader.py` | Reads `output.csv`, cleans it, deduplicates, takes a balanced sample |
| `preprocessing.py` | Train-only log transform, scaling, PCA, conversion to angles |
| `qsvm.py` | ZZ / Rx / general Pauli feature maps (Cirq), fast exact statevector engines, QSVM |
| `evaluation.py` | Metrics, CV tuning, McNemar test, kernel diagnostics |
| `benchmark.py` | Every experiment in the report (resumable) |
| `feature_map_study.py` | ZZ settings and other feature maps compared (resumable) |
| `hardware.py` | IBM quantum hardware run and comparison with simulation |
| `make_figures.py` | Builds report figures and tables from `outputs/benchmark/` |
| `tests/`, `run_tests.sh` | Unit tests (incl. Qiskit equivalence) and end-to-end checks |
| `legacy/` | The original, unmodified implementation |
| `outputs/benchmark/` | Results (`results.json`) of every benchmark run |
| `report/` | LaTeX report and compiled `report.pdf` |
| `output.csv` | Data: 74,995 flows from CIC-DDoS2019 (11 March 2019 capture) |

## Documentation

- [`TESTING.md`](TESTING.md): copy-paste commands for every test and experiment.
- [`report/report.pdf`](report/report.pdf): the formal report, with derivations and every table
  and figure.

## Data

`output.csv` is a subset of **CIC-DDoS2019** (Canadian Institute for Cybersecurity, University of
New Brunswick). If you use it, please cite:

> I. Sharafaldin, A. H. Lashkari, S. Hakak, A. A. Ghorbani. *Developing Realistic Distributed
> Denial of Service (DDoS) Attack Dataset and Taxonomy.* IEEE ICCST, 2019.
