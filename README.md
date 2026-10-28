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

Each run prints a summary table and writes `outputs/<run>/results.json` plus a figure.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Pipeline and CLI (`python main.py --help`) |
| `data_loader.py` | Reads `output.csv`, cleans it, deduplicates, takes a balanced sample |
| `preprocessing.py` | Train-only log transform, scaling, PCA, conversion to angles |
| `qsvm.py` | ZZ / Rx feature maps (Cirq), fast exact statevector engine, QSVM |
| `evaluation.py` | Metrics, CV tuning, McNemar test, kernel diagnostics |
| `benchmark.py` | Every experiment in the report (resumable) |
| `make_figures.py` | Builds report figures and tables from `outputs/benchmark/` |
| `tests/`, `run_tests.sh` | Unit tests and end-to-end checks |
| `legacy/` | The original, unmodified implementation |
| `outputs/benchmark/` | Results (`results.json`) of every benchmark run |
| `report/` | LaTeX report and compiled `report.pdf` |
| `output.csv` | Data: 74,995 flows from CIC-DDoS2019 (11 March 2019 capture) |

## Documentation

- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md): the full explanation, covering the theory, the data,
  the audit, every change, the results and an FAQ.
- [`TESTING.md`](TESTING.md): copy-paste commands for every test and experiment.
- [`report/report.pdf`](report/report.pdf): the formal report, with derivations and every table
  and figure.

The original paper (`qc_comsis_2.pdf`), referenced in the guide, is not included in this
repository.

## Data

`output.csv` is a subset of **CIC-DDoS2019** (Canadian Institute for Cybersecurity, University of
New Brunswick). If you use it, please cite:

> I. Sharafaldin, A. H. Lashkari, S. Hakak, A. A. Ghorbani. *Developing Realistic Distributed
> Denial of Service (DDoS) Attack Dataset and Taxonomy.* IEEE ICCST, 2019.
