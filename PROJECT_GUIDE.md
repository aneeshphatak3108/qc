# Quantum Network Defense — The Complete Project Guide

> **Network intrusion detection on CIC-DDoS2019 with quantum-kernel support vector machines (QSVMs), simulated with Google Cirq.**
>
> This guide explains the project from first principles to the last line of code: the problem, the data, the classical and quantum theory, the original implementation and what was wrong with it, every improvement and why it was made, how to run everything, how to read the outputs, and what the results mean.
>
> Companion files:
> - [`TESTING.md`](TESTING.md): copy-paste commands to test everything.
> - `report/report.pdf`: the ~37-page formal report with derivations, figures and all tables.
> - `qc_comsis_2.pdf`: the original paper.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Quick start](#2-quick-start)
3. [The problem: DDoS detection from network flows](#3-the-problem-ddos-detection-from-network-flows)
4. [The data: what is actually in `output.csv`](#4-the-data-what-is-actually-in-outputcsv)
5. [Classical machine-learning background](#5-classical-machine-learning-background)
6. [Quantum computing background](#6-quantum-computing-background)
7. [Quantum kernels in depth](#7-quantum-kernels-in-depth)
8. [The original project](#8-the-original-project)
9. [Audit: what was wrong](#9-audit-what-was-wrong)
10. [The improvements, one by one](#10-the-improvements-one-by-one)
11. [Code walkthrough (improved version)](#11-code-walkthrough-improved-version)
12. [Anatomy of a run: what happens when you type `python main.py`](#12-anatomy-of-a-run)
13. [Results](#13-results)
14. [Reading the outputs](#14-reading-the-outputs)
15. [Command-line reference](#15-command-line-reference)
16. [Tests, benchmark suite and report generation](#16-tests-benchmark-suite-and-report-generation)
17. [Limitations and future work](#17-limitations-and-future-work)
18. [FAQ](#18-faq)
19. [Glossary](#19-glossary)
20. [References](#20-references)

---

## 1. TL;DR

- **What it does.** It classifies network flows as **BENIGN** or **DDoS** (binary task), or as one of **six classes**: BENIGN, LDAP, MSSQL, Portmap, Syn, UDP (multiclass task). The classifier is a **support vector machine whose kernel is computed by a quantum circuit**: the overlap between quantum states that encode two flows.
- **Where the data is.** In `output.csv` (33 MB, 74,999 rows). There is no separate dataset folder. `main.py` reads this file by default.
- **The original claim.** The entangling ZZ quantum kernel beats a classical RBF-SVM: 98.79% vs 96.97% (binary) and 92.00% vs 80.00% (multiclass).
- **What we found.** Five main problems:
  - 66% of the rows are exact duplicates, so about 30–49% of test flows had an identical copy in training.
  - The scaler was fitted on all the data before the train/test split.
  - Heavy-tailed features were encoded linearly, wasting most of the rotation range.
  - Nothing was tuned, and the classical baseline in particular was left at its default settings.
  - The kernel computation was very slow: an estimated **44 hours** for one 16-qubit run.

  We also found that the paper's binary ZZ result was measured on only **165 test flows**, so "98.79% vs 96.97%" is **163 vs 160 correct**.
- **What we changed.** Deduplication, leakage-free train-only preprocessing with a log transform, a kernel bandwidth α, cross-validated tuning with an equal budget for both models, a fast exact statevector engine (≈26 s instead of ≈44 h, verified identical to Cirq), multi-seed runs with significance tests, richer metrics, kernel diagnostics, unit tests and a benchmark suite. The original code is preserved in `legacy/`.
- **What the corrected numbers say.** Under the original protocol we reproduce the paper's pattern: the QSVM wins, significantly, in 5 of 5 seeds. Under the corrected protocol, the **QSVM and the classical RBF-SVM are statistically tied** (no seed significant), and a Random Forest on all features is slightly better than both. The earlier "quantum advantage" came from leakage and an untuned baseline, not from entanglement.

---

## 2. Quick start

```bash
cd ~/Desktop/qcnn
source venv/bin/activate                       # Python 3.8 venv with cirq 1.1, sklearn 1.3

python -m unittest discover -s tests -v        # 19 tests, ~15 s — everything should say "ok"
python main.py --n-qubits 8                    # improved pipeline, binary, ~15 s
python main.py --legacy --n-qubits 8           # same circuit, ORIGINAL protocol, ~5 s
python main.py --mode multiclass --n-qubits 8  # 6-class, ~20 s
```

Each run prints a summary table and writes `outputs/<mode>_<kernel>_<n>q/results.json` plus a 6-panel figure. See [`TESTING.md`](TESTING.md) for every other command.

---

## 3. The problem: DDoS detection from network flows

### 3.1 DDoS in one paragraph
A **Distributed Denial-of-Service** attack makes a service unavailable by flooding it with traffic from many sources. Many modern DDoS attacks are **reflection/amplification** attacks. The attacker sends small requests to public servers with the *victim's* IP address forged as the source. The servers reply to the victim with responses that can be 10–100× larger. Each packet looks legitimate on its own, so simple signature rules do not catch these attacks. What gives them away is the **statistical shape of the traffic**.

### 3.2 The classes in this project

| Label | Kind | How it works | What it looks like as a flow |
|---|---|---|---|
| **MSSQL** | reflection | Abuses the SQL Server Resolution Protocol (UDP 1434) | One-directional UDP, characteristic response sizes |
| **LDAP** | reflection | Connectionless LDAP (CLDAP, UDP 389), huge amplification | Very repetitive identical flows |
| **Portmap** | reflection | RPC port mapper / rpcbind (UDP 111) | Very repetitive identical flows |
| **UDP** | exploitation | High-rate UDP flood to random ports | Short UDP flows, high packet rates |
| **Syn** | exploitation | TCP SYN flood, exhausts half-open connection table | Tiny TCP flows, SYN flag set, little/no backward traffic |
| **BENIGN** | — | Normal traffic from simulated users | Highly varied |

The three reflection attacks (MSSQL, LDAP, Portmap) resemble each other. That makes them the hard part of the 6-class task.

### 3.3 Flows and CICFlowMeter
A **flow** is the group of packets sharing a 5-tuple (source IP, destination IP, source port, destination port, protocol) within a time window. **CICFlowMeter** turns packet captures into one row per bidirectional flow, with ~80 statistics:

- **Duration and counts:** `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`
- **Byte totals and packet-length statistics:** `Total Length of Fwd Packets`, `Fwd Packet Length Max/Min/Mean/Std`, the same for backward, `Packet Length Mean/Std/Variance`, `Average Packet Size`
- **Rates:** `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`
- **Inter-arrival times (IAT):** `Flow IAT Mean/Std/Max/Min`, `Fwd IAT ...`, `Bwd IAT ...`
- **Flags:** `SYN/ACK/RST/URG/... Flag Count`, `Fwd PSH Flags`, ...
- **Headers, bulk, subflow, window and active/idle statistics**

These numbers are what the classifier learns from.

---

## 4. The data: what is actually in `output.csv`

### 4.1 Where it is and how it is read
- The **only** data file is `output.csv` in the project root. That is how the code runs "without a dataset folder".
- `main.py --csv` defaults to `./output.csv`, which is passed to `data_loader.load_csv()`, which calls `pd.read_csv()`.

### 4.2 Shape and columns
- 74,999 lines and 88 columns:
  - `Unnamed: 0`: an index pointing back into the original, much larger CSVs (values up to 482,780)
  - 7 identifiers: `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Protocol`, `Timestamp`
  - 77 numeric flow features
  - `SimillarHTTP`, `Inbound`
  - `Label`
- 4 of the lines are **repeated header rows** (Label = "Label"), which shows the file was built by concatenating several CSVs. Removing them leaves 74,995 data rows.

### 4.3 Where it came from
- The labels lack the `DrDoS_` prefix of the January files, and `Portmap` is present. Both match the **11 March 2019 capture day** of CIC-DDoS2019, whose timestamps read `2018-11-03`.
- The paper attributes these classes to the 12 January "training day". That should be corrected.
- Each class is a **contiguous slice** of its capture: all 14,999 Syn flows fall within **1.1 seconds**, all UDP flows within **2 seconds**, and all LDAP flows within about **2 minutes**. The file is not a random sample of the full dataset. This partly explains why the task is relatively easy.

### 4.4 Class counts and the duplicate problem

| Class | Rows | Exact duplicates | Unique rows | Duplicate share |
|---|---:|---:|---:|---:|
| BENIGN | 5,126 | 786 | 4,340 | 15.3% |
| LDAP | 9,931 | 9,353 | **578** | 94.2% |
| MSSQL | 19,657 | 13,697 | 5,960 | 69.7% |
| Portmap | 10,284 | 9,521 | **763** | 92.6% |
| Syn | 14,999 | 8,706 | 6,293 | 58.0% |
| UDP | 14,998 | 7,330 | 7,668 | 48.9% |
| **Total** | **74,995** | **49,393** | **25,602** | **65.9%** |

"Duplicate" means all 61 feature columns and the label are identical to an earlier row. Amplification attacks replay the same request millions of times, so identical flows are natural. For **evaluation**, though, duplicates are dangerous: if you split at random, a test flow very often has an identical twin in the training set, and the model only has to "remember" it (see §9).

### 4.5 Features used
- The original loader lists **61** numeric columns (`FEATURE_COLS` in `data_loader.py`).
- **12 of them are always zero** in this file: `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `FIN Flag Count`, `PSH Flag Count`, `ECE Flag Count`, and all six `Bulk` statistics.
- That leaves **49 informative features**, not the 58 the paper states. The original code padded the matrix with zero columns to reach exactly 58.
- `--features all` uses all 77 numeric columns (adds Subflow, `Init_Win_bytes_*`, `act_data_pkt_fwd`, `min_seg_size_forward`, Active/Idle).
- **Identifiers are never used.** In this testbed, IPs, ports and timestamps would identify the attack hosts and time windows. That gives a perfect but meaningless classifier.

### 4.6 Dirty values
- `Flow Bytes/s` and `Flow Packets/s` are **±∞ in about 2,650 rows**, because CICFlowMeter divides by a zero duration. 5 rows are NaN. All of these are set to 0.
- `Fwd Header Length` is **negative** in 12,592 rows (a CICFlowMeter artefact). The new log transform handles negatives.

### 4.7 Heavy tails
`Flow Bytes/s` goes up to 2.9 × 10⁹. With plain min–max scaling to [0, π]:
- **35%** of flows get an angle below 0.01 rad;
- for **69% of the features**, 90% of flows land in the bottom 5% of [0, π].

A qubit that receives practically the same angle for nearly every flow carries no information. This is why the improved pipeline applies `sign(x)·log(1+|x|)` first. After the transform, the median angle of `Flow Bytes/s` moves from 0.45 to 2.86 rad.

### 4.8 The two tasks
- **Binary:** BENIGN = 0, any attack = 1 ("DDoS"). Balanced by undersampling.
- **Multiclass:** the 6 labels, coded alphabetically: 0 BENIGN, 1 LDAP, 2 MSSQL, 3 Portmap, 4 Syn, 5 UDP. Balanced.
- `--samples 2000` (default) gives 1000 + 1000 (binary) or 6 × 333 (multiclass), split 80/20 stratified into **1600/1598 training and 400 test** flows.
- With deduplication, LDAP's 578 unique flows cap a balanced multiclass sample at 6 × 578 = 3,468.

---

## 5. Classical machine-learning background

### 5.1 Support vector machines
An SVM finds the separating boundary with the **largest margin** between classes. The soft-margin dual problem is

$$\max_{\beta}\ \sum_i \beta_i - \tfrac12\sum_{i,j}\beta_i\beta_j y_i y_j K(x_i,x_j)\quad\text{s.t. } 0\le\beta_i\le C,\ \sum_i\beta_i y_i=0,$$

and the prediction is $f(x)=\mathrm{sign}\big(\sum_i \beta_i y_i K(x_i,x)+b\big)$.

**Key fact:** the data only ever appear through the **kernel** $K(x,x')$, an inner product in some (possibly enormous) feature space. Any valid (positive semi-definite) kernel can be used, including one computed by a quantum circuit. That is the whole basis of a QSVM.

- `C` (regularisation): a large C means fewer training errors and a narrower margin, so a higher risk of overfitting.
- `SVC(kernel="precomputed")`: you give scikit-learn the **N×N Gram matrix** $K_{ij}=K(x_i,x_j)$ for training and the **M×N** test–train matrix for prediction. This is how the quantum kernel is plugged in.
- **Multiclass:** scikit-learn trains one-vs-one SVMs for every pair of classes ($k(k-1)/2$ of them) and takes a vote.
- **Probabilities:** `predict_proba` uses **Platt scaling**, a logistic fit on internal cross-validation. It is random, so it must be seeded.

### 5.2 The RBF kernel (the classical baseline)
$K(x,x')=\exp(-\gamma\|x-x'\|^2)$. **γ is the bandwidth:**
- γ too large: $K\approx I$ (every point is only similar to itself), so the model memorises.
- γ too small: $K\approx$ all ones (everything looks the same), so the model underfits.

`gamma="scale"` is scikit-learn's heuristic default, $1/(d\cdot\mathrm{Var}(X))$. It is a reasonable guess, not an optimum. The same bandwidth idea is central to quantum kernels (§7.4).

### 5.3 PCA
**Principal Component Analysis** projects data onto the directions of greatest variance. Here it compresses ~49 features into **n numbers, one per qubit** (n = 2…16). It must be fitted on training data only.

### 5.4 Cross-validation
To choose hyperparameters (C, γ, α) without touching the test set, the training set is split into k = 5 folds. Each candidate is trained on 4 folds and scored on the 5th, rotating through all 5. The candidate with the best average score wins, is refitted on the whole training set, and is evaluated **once** on the test set.

### 5.5 Metrics used

| Metric | Meaning | Why it matters here |
|---|---|---|
| Accuracy | fraction correct | the paper's only metric |
| Balanced accuracy | mean recall per class | robust to imbalance |
| Macro-F1 | unweighted mean F1 over classes | punishes ignoring any class; **our CV criterion** |
| MCC | Matthews correlation | single robust summary |
| ROC-AUC | ranking quality (binary: from the SVM margin; multiclass: one-vs-rest macro) | threshold-free |
| **False-alarm rate (FAR)** | BENIGN flows flagged as attacks | operators drown in false alarms |
| **Detection rate (DR)** | attack flows flagged as *some* attack | missed attacks are the real risk |

### 5.6 Significance: McNemar's test
Two classifiers are tested on the **same** flows. Let $n_{01}$ be the number of flows only model A gets right, and $n_{10}$ the number only model B gets right. If the models were equally good, $n_{01}$ would follow Binomial($n_{01}+n_{10}$, ½). The exact two-sided p-value tells you whether the difference is real. **Rule of thumb:** with 400 test flows, one mistake is 0.25 points, so a 1-point difference is only 4 flows and is usually not significant.

---

## 6. Quantum computing background

### 6.1 Qubits and statevectors
- One qubit: $|\psi\rangle = a|0\rangle + b|1\rangle$ with $|a|^2+|b|^2=1$.
- n qubits: a vector of **2ⁿ complex amplitudes**, one for each bit-string $z\in\{0,1\}^n$.
- Measuring gives bit-string z with probability $|\langle z|\psi\rangle|^2$ (Born rule).
- Simulating on a classical computer means storing all 2ⁿ amplitudes:

| Qubits | Amplitudes | Memory (complex64) |
|---:|---:|---:|
| 8 | 256 | 2 KB |
| 16 | 65,536 | 512 KB |
| 20 | 1,048,576 | 8 MB |
| 58 | 2.9 × 10¹⁷ | ≈ 2.3 **exabytes** |

That is why the 49 features are compressed with PCA to ≤16 numbers. (The paper says 128 GB for 58 qubits; the correct figure is ≈2.3 × 10¹⁸ bytes.)

### 6.2 Gates used

| Gate | Matrix / action | Role |
|---|---|---|
| **H** (Hadamard) | $\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | creates superposition |
| **Rz(θ)** | $\mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})$ | encodes a number as a phase (diagonal) |
| **Rx(θ)** | $\begin{pmatrix}\cos\frac\theta2&-i\sin\frac\theta2\\-i\sin\frac\theta2&\cos\frac\theta2\end{pmatrix}$ | encodes a number as a rotation |
| **CNOT** | $|a,b\rangle\mapsto|a,a\oplus b\rangle$ | entangles two qubits |

### 6.3 Two identities that matter
Write $s(z)=1-2z\in\{+1,-1\}$.
1. $R_z(\theta)|z\rangle = e^{-i\theta s(z)/2}|z\rangle$: **Rz is diagonal**, it only multiplies by a phase.
2. $\mathrm{CNOT}_{ij}\,R_z^{(j)}(\theta)\,\mathrm{CNOT}_{ij} = e^{-i\frac\theta2 Z_iZ_j}$: the **"ZZ" gate is also diagonal**, multiplying $|z\rangle$ by $e^{-i\frac\theta2 s(z_i)s(z_j)}$.

So in the ZZ feature map, **everything except H is diagonal**. That observation is what allowed the fast simulator (§10.6).

---

## 7. Quantum kernels in depth

### 7.1 Feature map and fidelity kernel
A **feature map** is a circuit $U(x)$ that turns a data vector x into a quantum state $|\phi(x)\rangle = U(x)|0\cdots0\rangle$. The **quantum kernel** is the squared overlap:

$$K(a,b)=|\langle\phi(a)|\phi(b)\rangle|^2=|\langle 0|U^\dagger(a)U(b)|0\rangle|^2 .$$

- The **second form** is how hardware does it: run $U(b)$, then $U^\dagger(a)$, measure, and count how often you get all zeros.
- A **simulator** can instead compute both states and take their inner product.
- Properties: $K(x,x)=1$, $0\le K\le1$, and K is a valid (PSD) kernel.

### 7.2 Method 1: separable Rx ("Analytical") kernel
Encoding: $|\phi(x)\rangle=\bigotimes_i R_x(x_i)|0\rangle$, a product state with **no entanglement**. The overlap factorises:

$$K_{R_x}(a,b)=\prod_{i}\cos^2\!\Big(\frac{b_i-a_i}{2}\Big).$$

This is a cheap classical formula, essentially a periodic RBF-like kernel. **No quantum computer is needed to evaluate it.**

### 7.3 Method 3: the ZZ entangling feature map (Havlíček et al. 2019)
One repetition on n qubits, with $x$ the (scaled) angles:

```
|0⟩ ─H─Rz(2x₀)──●──────────────────●──────────────────────────────────── ...
|0⟩ ─H─Rz(2x₁)──X─Rz(2x̄₀x̄₁)─X──●──────────────────●─────────────────── ...
|0⟩ ─H─Rz(2x₂)───────────────────X─Rz(2x̄₁x̄₂)─X──●──────────────────●── ...
|0⟩ ─H─Rz(2x₃)──────────────────────────────────────X─Rz(2x̄₂x̄₃)─X── ...
                                         where x̄ᵢ = π − xᵢ
```

As a formula:

$$U_{ZZ}(x)=\underbrace{\prod_{(i,j)\in E} e^{-i(\pi-x_i)(\pi-x_j)Z_iZ_j}\ \prod_i e^{-ix_iZ_i}}_{D(x)\text{, diagonal}}\ H^{\otimes n},$$

repeated `n_reps` times. **E** is the set of entangled pairs: adjacent pairs (`linear`) in the original. With 2+ repetitions, the alternation of H layers and diagonal phase layers produces interference that is **conjectured to be hard to simulate classically** for large n. That conjecture is the theoretical case for a quantum advantage.

### 7.4 Kernel concentration and the bandwidth α
For expressive maps on many qubits, two different inputs produce **nearly orthogonal** states, so $K(a,b)\approx 0$ for every pair and the Gram matrix is ≈ the identity. Consequences:
- the SVM can separate the training set perfectly, but each test prediction then rests on the few closest training points, and the kernel's geometry says little;
- on hardware, estimating values ≈ 2⁻ⁿ needs exponentially many shots.

This is **exponential concentration** (Thanasilp et al. 2024). The fix is the quantum equivalent of tuning γ: **scale the inputs by α < 1** (Shaydulin & Wild 2022; Canatar et al. 2023). The states move closer together and the kernel regains useful structure. The original project always used α = 1.

For the ZZ map, $(\pi-\alpha\theta_i)(\pi-\alpha\theta_j)=\pi^2-\alpha\pi(\theta_i+\theta_j)+\alpha^2\theta_i\theta_j$. So α also controls how strongly the **entangling (product) terms** depend on the data.

### 7.5 Can this kernel beat classical ones? The geometric difference
Huang et al. (2021) defined

$$g(K_C\|K_Q)=\sqrt{\big\|\sqrt{K_Q}\,(K_C+\lambda I)^{-1}\sqrt{K_Q}\big\|_\infty},$$

computed on the training data, with both kernels normalised to trace N.
- **g ≈ 1:** the classical kernel can do everything the quantum one can, on this data. No advantage is possible.
- **g ≳ √N:** there is *room* for a quantum advantage. It is necessary, not sufficient: the labels must also follow the quantum kernel's geometry.

The improved pipeline computes g for every run (against the tuned RBF kernel, on 400 training points, so √N = 20).

---

## 8. The original project

### 8.1 Files (preserved unchanged in `legacy/`)
- **`legacy/data_loader.py`:** reads the CSV, takes 3×`max_samples` random rows, makes labels, picks columns, replaces inf/NaN with 0, drops constant columns, pads to 58, balances with **unseeded** `np.random`, then **MinMax-scales everything to [0, π] before the split**.
- **`legacy/qsvm.py`:**
  - `ZZFeatureMap` (Cirq circuit builder);
  - `QuantumKernel`, which for **every pair** simulates **both** circuits and returns $|\langle\psi_a|\psi_b\rangle|^2$ (or runs the $U^\dagger U$ circuit 1024 times in sampling mode);
  - `QSVM` (PCA on train, then min–max [0, π], then the kernel, then `SVC(precomputed, C=1)`; `predict` and `predict_proba` each rebuild the test kernel).
- **`legacy/main.py`:** split 80/20, train the QSVM, train `SVC(rbf, C=1, gamma='scale')` on the same PCA features, print the results, save `outputs/qnd_results.png`.

### 8.2 The paper's results

| Method | Mode | QSVM | RBF-SVM | N |
|---|---|---:|---:|---:|
| Analytical Rx (16q) | binary | 97.57% | 98.54% | 1028 test |
| Analytical Rx (16q) | multiclass | 89.63% | 87.68% | 3084 test |
| Sampling 1024 shots | binary | 98.54% | 98.54% | 1028 |
| Sampling 1024 shots | multiclass | 89.63% | 87.68% | 3084 |
| **ZZ (8q, 2 reps)** | binary | **98.79%** | 96.97% | 822 |
| **ZZ (16q, 1 rep)** | multiclass | **92.00%** | 80.00% | 996 |

Only the ZZ method was in the code. The other two came from earlier versions.

---

## 9. Audit: what was wrong

| # | Issue | Severity | Evidence |
|---|---|---|---|
| I1 | **Duplicates kept.** Test flows have identical training twins | High | 28.4% (binary) and **48.6% (multiclass)** of test rows have an exact copy in train under the original protocol |
| I2 | **Untuned, unequal baselines.** C = 1 for both; RBF γ left at its default | High | Tuning the RBF alone closes the gap (§13) |
| I3 | **Kernel concentration.** No bandwidth control | High at 16 qubits | Off-diagonal entries collapse towards 0 as α grows (bandwidth sweep in the report) |
| I4 | **2N² circuit simulations.** Test kernel built twice | High (practical) | Estimated **43.8 h** for a 1600/400 run at 16 qubits |
| I5 | **Global MinMax before the split.** Test statistics leak into training | Medium | Code inspection |
| I6 | **Linear encoding of heavy tails.** Most angles ≈ 0 | Medium | §4.7 |
| I7 | **Single runs, no error bars or tests** | Medium | — |
| I8 | **Unseeded sampling and Platt scaling.** Results not reproducible | Medium | — |
| I9 | **Methods compared at different N** (822 vs 1028) | Medium | — |
| I10 | **Two of the three paper methods missing** from the code | Medium | — |
| I11 | **Accuracy only.** No FAR/DR/F1/MCC, no multiclass AUC | Low | — |
| I12 | **Cosmetic.** Zero-padding to 58, hard-coded "58 →" message, silent truncation to 3000 | Low | — |
| I13 | **Pre-sampling shrinks the data.** 3×`max_samples` rows drawn *before* balancing | Medium | `--samples 2000` gives **822** binary samples, exactly the paper's N, so only **165 test flows** |

**I13 explains the paper's binary numbers.** 98.79% = 163/165 and 96.97% = 160/165. The headline binary "advantage" is **3 flows**. Even if all three went the QSVM's way, McNemar gives p = 0.25: not significant. The multiclass N = 996 = 6 × 166 corresponds to a 200-flow test set.

None of this means the idea was bad. These are the most common traps in IDS benchmarking (duplicate leakage) and in quantum-ML benchmarking (untuned baselines). The quantum code itself was correct: the new engine reproduces its kernel values exactly.

---

## 10. The improvements, one by one

Every change keeps the original architecture: same three modules, same Cirq feature-map class, same CLI flags, same data file. Each fix can be switched off for comparison (`--legacy`, `--legacy-scaling`, `--keep-duplicates`, `--no-log`, `--no-tune`).

### 10.1 Duplicate removal (I1) — `data_loader.py`
- `load_csv(dedup=True)` drops rows whose features **and** label equal an earlier row, *before* sampling. The label is the one *of the task*: in binary mode an LDAP flow and an identical Portmap flow are both "DDoS", so the second is a duplicate (175 extra rows removed in binary mode compared with keying on the 6-class label).
- Rows with identical features but **different** task labels are kept: 177 in multiclass mode (mostly reflection-attack flows that look the same under two labels), 2 in binary mode. That is genuine ambiguity, not leakage.
- `duplicate_overlap(X_train, X_test)` hashes training rows and reports the fraction of test rows found among them. It prints in every run as `test rows with an exact copy in train`. After dedup it is 0–0.8%; before, 28–49%.

### 10.2 Leakage-free preprocessing (I5) — `preprocessing.py`
- The loader now returns **raw** features.
- `QuantumPreprocessor` is fitted **on the training split only**, inside `QSVM.fit()`:
  1. drop zero-variance columns,
  2. signed log,
  3. `StandardScaler`,
  4. `PCA(n_qubits)`,
  5. per-component min–max to [0, π].
- Test data are transformed with the stored statistics and **clipped** to [0, π].
- The classical baselines use `transform_pca()` of the *same* fitted object, so both models see exactly the same information.

### 10.3 Signed log transform (I6) — `preprocessing.py`
- $x\mapsto \mathrm{sign}(x)\log(1+|x|)$ is monotone, works for zeros and negatives, and squeezes 9 orders of magnitude into ~22 units.
- `--no-log` disables it.

### 10.4 Reproducible, honest sampling (I8, I12, I13) — `data_loader.py`
- All randomness comes from `np.random.default_rng(seed)` and `random_state=seed` (split, PCA, CV folds, Platt scaling, Random Forest, shot noise).
- Classes are balanced from the **whole** deduplicated pool, so `--samples 2000` really means 2000.
- No zero padding.
- `--legacy` uses this sampler too, so old and new protocols are always compared on the same number of flows.

### 10.5 Statevector caching (I4) — `qsvm.py`
- **Problem:** $|\phi(x)\rangle$ depends only on x, yet it was re-simulated for every pair: 5.1 million simulations for 1600 train + 400 test.
- **Change:** `QuantumKernel.states(X)` simulates each sample **once**, then

  $$K(X,Y)=\big|\overline{\Phi_X}\,\Phi_Y^\top\big|^{2}\quad(\text{element-wise}),$$

  which is a single complex matrix product (BLAS `cgemm`), done in blocks of 512 rows.
- The last two state matrices are memoised, so building the test kernel reuses the training states. `predict`, `predict_proba` and `decision_function` share one cached test kernel.

### 10.6 The fast exact ZZ engine (I4) — `qsvm.FastZZStatevector`
Because everything except H is diagonal (§6.3), one repetition is $D_x H^{\otimes n}$ with $D_x|z\rangle=e^{-i\Phi_x(z)}|z\rangle$ and

$$\Phi_x(z)=\sum_i x_i s_i(z)+\sum_{(i,j)\in E}(\pi-x_i)(\pi-x_j)s_i(z)s_j(z).$$

Implementation:
1. **Phase table:** precompute the ±1 sign matrix S (2ⁿ × n) and the pair-product matrix S_P (2ⁿ × |E|). For a batch of inputs, $\Phi = XS^\top + WS_P^\top$: two real matrix products.
2. **First repetition is free:** $\psi = 2^{-n/2}e^{-i\Phi}$.
3. **Hadamard layers:** $H^{\otimes n}=H^{\otimes n_1}\otimes H^{\otimes n_2}$. Reshape ψ into a $2^{n_1}\times2^{n_2}$ matrix M; then $(A\otimes B)\,\mathrm{vec}(M)=\mathrm{vec}(AMB^\top)$, i.e. two 256×256 matmuls at 16 qubits.
4. Multiply by $D_x$ again and repeat.

Qubit 0 is the most significant bit, which matches Cirq's ordering, so the vectors are **directly comparable to Cirq's**. The maximum difference is 2.4 × 10⁻⁷ (float32 rounding).

**Speed (train 1600² + test 400×1600, 2 reps, linear):**

| Qubits | Original (est.) | Cirq, cached states (est.) | **Fast (measured)** | Speed-up vs original |
|---:|---:|---:|---:|---:|
| 4 | 6.1 h | 8.5 s | **0.06 s** | ~370,000× |
| 8 | 17.6 h | 17.3 s | **0.16 s** | ~400,000× |
| 12 | 24.4 h | 27.7 s | **0.88 s** | ~100,000× |
| 16 | 43.8 h | 67.5 s | **26.4 s** | ~6,000× |

This is what made the multi-seed, cross-validated experiments possible.

### 10.7 Shot noise without running millions of circuits (I10) — `qsvm.py`
- Measuring $U^\dagger(a)U(b)$ `s` times and counting all-zeros outcomes gives a **Binomial(s, K(a,b))** count. Each shot is an independent trial with success probability K.
- So with `--sampling`, the fast backend draws $\hat K = \mathrm{Binomial}(s,K)/s$ for each entry. This has *exactly the same distribution* as the original sampling method, at negligible cost.
- `--backend cirq --sampling` still runs the real circuits if you want to see them.

### 10.8 Kernel bandwidth α (I3) — `qsvm.py`
- Every feature map (Cirq and fast, ZZ and Rx) multiplies angles by `alpha` before encoding.
- α = 1 reproduces the original circuit exactly.

### 10.9 Cross-validated tuning with equal budgets (I2) — `evaluation.py`
- **QSVM:** α ∈ {0.05, 0.1, 0.2, 0.4, 1.0} × C ∈ {0.1, 1, 10, 100}, macro-F1 by 5-fold stratified CV **on the training set**.
  - The kernel is computed **once per α**. Every C and every fold reuses sub-blocks of that matrix, so the search costs only 5 kernel computations.
- **RBF-SVM:** the same C grid × γ ∈ {scale, 0.01, 0.03, 0.1, 0.3, 1}, same folds, metric and seed.
- Both get about 20–24 candidates: a fair fight.

### 10.10 More circuits (paper's future work) — `qsvm.entangling_pairs`
- `--entanglement linear | circular | full` and `--n-reps r`.
- Both Cirq and the fast engine use the same pair list.

### 10.11 All three methods in one pipeline (I9, I10)
- `--kernel rx` gives Method 1 and `--sampling` gives Method 2; the ZZ map is the default.
- Same loader, same split, same preprocessing and same tuning for every method and seed.

### 10.12 Better evaluation (I7, I11) — `evaluation.py`, `main.py`
- Accuracy, balanced accuracy, macro-F1, MCC, AUC (binary and OvR), false-alarm rate and detection rate.
- `--seeds 0 1 2 3 4` repeats **everything** per seed and reports mean ± std.
- An exact McNemar test (QSVM vs RBF) for each seed.
- Kernel diagnostics: off-diagonal mean/std (concentration), centred kernel–target alignment, and the geometric difference g.
- `--extra-baselines` adds a Random Forest (300 trees) on all raw features, as a practical IDS reference.

### 10.13 Engineering
- `legacy/` keeps the original code runnable.
- `tests/test_core.py` has 19 tests, including "new kernel equals the original `legacy/qsvm.py` kernel".
- `benchmark.py` is a resumable experiment suite; `make_figures.py` turns its results into the report's figures and tables.
- `results.json` and a 6-panel PNG are saved per run.

---

## 11. Code walkthrough (improved version)

### 11.1 `data_loader.py`
| Name | What it does |
|---|---|
| `FEATURE_COLS` | the 61 feature names from the original loader |
| `META_COLS` | identifier columns that must never be features |
| `_read_clean(path)` | read the CSV, strip names, drop repeated header rows |
| `_feature_frame(df, feature_set)` | select `paper` (61) or `all` (77) columns, coerce to numbers, ±inf/NaN → 0 |
| `load_csv(csv_path, mode, max_samples, balance, dedup, feature_set, seed, verbose)` | dedup, make labels, balance with seeded rng; returns `X_raw, y, label_names, info` |
| `duplicate_overlap(X_train, X_test)` | fraction of test rows that also appear in train |

`info` contains `rows_raw`, `duplicates_removed`, `rows_unique`, `n_samples`, `columns`.

### 11.2 `preprocessing.py`
| Name | What it does |
|---|---|
| `signed_log1p(X)` | $\mathrm{sign}(x)\log(1+|x|)$ |
| `QuantumPreprocessor(n_components, log_transform, clip, seed)` | `fit` (train only), `transform` (angles in [0, π]), `transform_pca` (PCA scores for classical models), `explained_variance`, `n_input_features` |
| `legacy_global_scale(X)` | the original global MinMax + pad to 58 |
| `LegacyPreprocessor` | the original PCA + min–max without clipping |

### 11.3 `qsvm.py`
| Name | What it does |
|---|---|
| `entangling_pairs(n, mode)` | `linear` / `circular` / `full` pair lists |
| `ZZFeatureMap(n_qubits, n_reps, entanglement, alpha)` | the Cirq circuit: `build_encoding_circuit(angles)`, `build_kernel_circuit(a, b)` |
| `RxFeatureMap(n_qubits, alpha)` | separable Rx circuit (Method 1) |
| `_hadamard_matrix(n)` | dense normalised $H^{\otimes n}$ |
| `FastZZStatevector` | `phases(X)`, `states(X)`: the exact fast engine |
| `QuantumKernel(n_qubits, n_reps, shots, use_sampling, feature_map, entanglement, alpha, backend, seed)` | `states`, `compute_kernel_value`, `compute_kernel_matrix` (exact / binomial shots / real Cirq sampling), state memo |
| `QSVM(...)` | `fit(X_raw, y, K_train=None)`, `predict`, `predict_proba`, `decision_function`, `score`, `test_kernel`, `.pca` |

### 11.4 `evaluation.py`
| Name | What it does |
|---|---|
| `classification_metrics(y, pred, scores, benign_label)` | all metrics + confusion matrix |
| `mcnemar(y, pred_a, pred_b)` | exact McNemar |
| `cv_score_precomputed(K, y, C)` | 5-fold CV macro-F1 on a precomputed kernel |
| `tune_quantum(kernel_fn, y, alphas, Cs)` | α × C grid; returns best α, C, K_train, table |
| `tune_rbf(Z, y, Cs, gammas)` | `GridSearchCV` for the RBF-SVM |
| `offdiag_mean/std`, `centered_alignment`, `geometric_difference`, `rbf_kernel_matrix`, `kernel_diagnostics` | diagnostics |
| `standardize(train, test)` | `StandardScaler` fitted on train |

### 11.5 `main.py`
| Name | What it does |
|---|---|
| `run_once(args, seed)` | one full experiment (6 steps, §12) → dict |
| `plot_run(res, args, out_dir)` | the 6-panel figure |
| `summarize(runs)` | mean ± std across seeds |
| `main(argv)` | CLI parsing, loop over seeds, save `results.json`, print the summary |

### 11.6 Support files
- `benchmark.py`: experiments `speed, headline, ablation, bandwidth, qubits, shots, architecture, features, samples`.
- `make_figures.py`: builds `report/figures/*` and `report/tables/*`.
- `tests/test_core.py`: unit tests.
- `report/`: LaTeX sources (`report.tex`, `sections/*.tex`).
- `TESTING.md`: commands.
- `PROJECT_GUIDE.md`: this file.

---

## 12. Anatomy of a run

`python main.py --n-qubits 8 --seeds 0` does the following:

1. **Load** (`load_csv`). Read 74,995 rows, remove 49,393 duplicates (25,602 unique), sample 1000 BENIGN + 1000 DDoS with `default_rng(0)`, and return raw features (2000 × 61).
2. **Split.** 80/20 stratified (`random_state=0`) gives 1600 train / 400 test. Print the duplicate overlap (≈0%).
3. **Preprocess** (train only). Drop the 12 constant columns → 49 features → signed log → z-score → PCA(8) → [0, π]. Print the variance retained.
4. **QSVM.**
   - For each α in the grid: compute the 1600 training states (fast engine) and the 1600×1600 kernel.
   - For each C: 5-fold CV macro-F1.
   - Choose the best (α, C), fit the SVC on the full training kernel, compute the 400×1600 test kernel (training states reused), predict.
5. **RBF-SVM.** z-scored PCA features, then `GridSearchCV` over (C, γ), then predict. Run McNemar against the QSVM.
6. **Extra (optional).** Random Forest on all raw features.
7. **Diagnostics.** Concentration, alignment and geometric difference on the training kernels.
8. **Save.** `outputs/binary_zz_8q/results.json` and `results_seed0.png`; print the summary table.

Typical console summary:
```
  Method                  Accuracy        Macro-F1             AUC    FalseAlarm
  QSVM (zz)           99.25 ± 0.00    99.25 ± 0.00    99.91 ± 0.00   0.50 ± 0.00
  RBF-SVM             99.25 ± 0.00    99.25 ± 0.00    99.80 ± 0.00   0.50 ± 0.00
  McNemar p-value (QSVM vs RBF) per seed: 1
```
(Exact numbers depend on the seed.)

---

## 13. Results

All numbers are mean ± std over **5 seeds**. Each seed has its own sample, split, preprocessing and tuning; test set = 400 flows. "Sig." is the number of seeds where McNemar p < 0.05 for QSVM vs RBF. The full set of experiments (bandwidth, qubit count, shots, architecture, feature set, learning curves) is in `report/report.pdf`.

### 13.1 Headline: same circuits, original vs improved protocol

| Task | Method | Protocol | Test rows with twin in train | QSVM acc. | RBF-SVM acc. | Random Forest | Sig. |
|---|---|---|---:|---:|---:|---:|---:|
| binary | ZZ, 8q, 2 reps (paper) | original | 28.4% | 99.50 ± 0.32 | 97.35 ± 0.25 | 99.90 ± 0.20 | **5/5** |
| binary | ZZ, 8q, 2 reps (paper) | **improved** | 0.0% | 99.60 ± 0.41 | 99.55 ± 0.29 | 99.70 ± 0.29 | 0/5 |
| binary | ZZ, 16q, 2 reps | improved | 0.0% | 99.60 ± 0.25 | 99.50 ± 0.27 | 99.70 ± 0.29 | 0/5 |
| binary | Rx analytical, 16q | original | 28.4% | 98.35 ± 0.12 | 98.60 ± 0.41 | — | 0/5 |
| binary | Rx analytical, 16q | improved | 0.0% | 99.50 ± 0.32 | 99.50 ± 0.27 | — | 0/5 |
| binary | Rx + 1024 shots, 16q | original | 28.4% | 98.30 ± 0.10 | 98.60 ± 0.41 | — | 0/5 |
| binary | Rx + 1024 shots, 16q | improved | 0.0% | 99.30 ± 0.62 | 99.50 ± 0.27 | — | 0/5 |
| multiclass | ZZ, 16q, 1 rep (paper) | original | **48.6%** | 95.70 ± 0.56 | 88.85 ± 1.12 | 98.05 ± 0.60 | **5/5** |
| multiclass | ZZ, 16q, 1 rep (paper) | **improved** | 0.8% | 90.85 ± 0.85 | 91.05 ± 0.95 | 93.10 ± 0.49 | 0/5 |
| multiclass | Rx analytical, 16q | original | 48.6% | 91.70 ± 2.54 | 88.85 ± 1.12 | — | 2/5 |
| multiclass | Rx analytical, 16q | improved | 0.8% | 91.05 ± 0.24 | 91.05 ± 0.95 | — | 0/5 |
| multiclass | Rx + 1024 shots, 16q | original | 48.6% | 91.35 ± 1.30 | 88.85 ± 1.12 | — | 2/5 |
| multiclass | Rx + 1024 shots, 16q | improved | 0.8% | 88.10 ± 0.41 | 91.05 ± 0.95 | — | 2/5 (RBF better) |

### 13.2 What this means
1. **The original protocol reproduces the paper's story.** The ZZ QSVM beats the RBF-SVM by +2.2 points (binary) and +6.9 points (multiclass), significant in every seed. So the paper's observation was real *under its protocol*.
2. **Under the corrected protocol, the gap disappears.** The differences shrink to ±0.2 points and no seed is significant. The QSVM didn't get worse; the **RBF-SVM got much better** (97.35 → 99.55 binary; 88.85 → 91.05 multiclass) once it had good preprocessing and tuning.
3. **Duplicates inflated multiclass accuracy a lot.** The QSVM dropped from 95.70% to 90.85% once almost half the test set no longer had training twins. The reflection classes (LDAP, Portmap, MSSQL) are the hard ones, and they are precisely the ones that are 70–94% duplicates.
4. **Entanglement did not help here.** On identical data, the separable Rx kernel (99.50%) and the entangling ZZ kernel (99.60%) are within noise of each other and of the RBF.
5. **Shots matter in the harder task.** In binary, 1024-shot estimation is within noise of the exact kernel (99.30 vs 99.50). In multiclass it costs about 3 points (88.10 vs 91.05 exact), and the RBF-SVM becomes significantly better in 2 of 5 seeds. Distinguishing the similar reflection classes depends on small kernel differences, which 1024 shots blur. On real hardware this would be the first thing to hurt.
6. **Compressing to PCA costs some accuracy.** A Random Forest on all 49 raw features is the best model in every setting (99.70% binary, 93.10% multiclass), above every kernel model on ≤16 PCA components.
7. **Fewer false alarms.** In binary mode the improved QSVM's false-alarm rate is 0.3 ± 0.4%; for the Rx kernel under the original protocol it is 3.0%.
8. **Diagnostics agree** (mean over 5 seeds). With tuned α, the geometric difference g against the tuned RBF kernel is 2.9–3.1 for Rx, 4.0–9.2 for the 16-qubit ZZ circuit and 10.9–13.8 for the paper's ZZ settings, all below √N = 20. On this data a classical kernel can express what the quantum kernel expresses, which is why no advantage appears. The legacy runs show very large g (65.7 binary, 112.0 multiclass); that reflects the badly scaled RBF baseline and the near-identity α = 1 quantum kernel, not quantum power.
9. **PSD projection of shot-noise kernels hurts.** A finite-shot Gram matrix is not PSD (negative eigenvalues summing to −83…−319 on 1,600 flows). Clipping them (`--psd`, Hubregtsen et al. 2022) made the 16-qubit multiclass shot result *worse* in 5/5 seeds (88.10% → 84.65%), because only the training kernel is fixed and the test kernel keeps its noise. It is therefore off by default.
10. **The Rx bandwidth grid is truncated.** The tuned α of the Rx kernel was 1.0, the top of the default grid, in every headline run. With α up to 3 it rises slightly (binary 99.60%, multiclass 91.65%), still with no significant seed against the RBF-SVM (p ≥ 0.125). The reported Rx numbers are slightly conservative; the conclusion does not change. Pass e.g. `--alphas 0.05 0.1 0.2 0.4 1 1.5 2 3` for Rx.

### 13.3 The one-sentence conclusion
> On this dataset, once leakage is removed and both models get the same tuning budget, **the quantum kernel SVM matches but does not beat a classical RBF-SVM**. The previously reported advantages come from duplicate leakage, an untuned classical baseline, and (for binary) a 165-flow test set.

That is still a valuable result. The pipeline is now fast, correct, reproducible and statistically sound, and it is ready for harder datasets where a quantum kernel *might* matter (see §17).

---

## 14. Reading the outputs

### 14.1 Console
- `[2/6] ... test rows with an exact copy in train: X%`: the leakage check.
- `Preprocessing: 49 features → PCA 8 (xx% variance, fitted on train only)`.
- `α=..., C=... | accuracy ... | macro-F1 ...`: the chosen hyperparameters and test performance.
- `[6/6] Quantum kernel: off-diag mean ... | alignment ... | geometric difference g=... (√N=20.0)`.
- The final table: mean ± std over seeds, and McNemar p per seed.

### 14.2 `results.json`
```jsonc
{
  "args":    { ...every CLI argument... },
  "summary": { "qsvm": {"accuracy": {"mean":..., "std":...}, "macro_f1": ..., "auc": ..., "false_alarm_rate": ..., ...},
               "rbf_svm": {...}, "random_forest": {...} },
  "runs": [ {
      "seed": 0,
      "data": {"rows_raw", "duplicates_removed", "rows_unique", "n_samples", "n_train", "n_test",
               "test_rows_with_exact_copy_in_train"},
      "pca_variance": 0.9...,
      "qsvm_tuning": [ {"alpha", "C", "cv_macro_f1", "offdiag_mean"}, ... ],
      "qsvm":    {"accuracy", "balanced_accuracy", "macro_f1", "mcc", "auc", "false_alarm_rate",
                  "detection_rate", "confusion_matrix", "alpha", "C", "train_s", "infer_s"},
      "rbf_svm": {..., "C", "gamma"},
      "random_forest": {...},
      "mcnemar_qsvm_vs_rbf": {"a_right_b_wrong", "a_wrong_b_right", "p_value"},
      "kernel_diagnostics": {"quantum": {"offdiag_mean", "offdiag_std", "alignment",
                                         "geometric_difference", "geometric_difference_n"},
                             "rbf": {...}}
  }, ... ]
}
```

### 14.3 The 6-panel figure (`results_seed<k>.png`)
| Panel | Shows |
|---|---|
| top-left | QSVM confusion matrix (with chosen α, C) |
| top-middle | RBF-SVM confusion matrix |
| top-right | training quantum kernel sorted by class: bright diagonal blocks mean the kernel sees same-class flows as similar |
| bottom-left | ROC curves (binary) or OvR AUC values (multiclass) |
| bottom-middle | CV macro-F1 vs α for each C (the tuning curve) |
| bottom-right | macro-F1 bars for all models |

---

## 15. Command-line reference

| Option | Default | Meaning |
|---|---|---|
| `--csv PATH` | `./output.csv` | data file |
| `--samples N` | 2000 | total balanced sample size |
| `--mode` | `binary` | `binary` / `multiclass` |
| `--n-qubits n` | 16 | qubits = PCA components |
| `--n-reps r` | 2 | feature-map repetitions |
| `--kernel` | `zz` | `zz` (entangling) / `rx` (separable) |
| `--entanglement` | `linear` | `linear` / `circular` / `full` |
| `--backend` | `fast` | `fast` (NumPy, exact) / `cirq` |
| `--sampling`, `--shots s` | off, 1024 | finite-shot kernel |
| `--psd` | off | with `--sampling`: clip negative eigenvalues of the noisy training kernel (hurts; see §13.2) |
| `--alphas …`, `--Cs …` | 0.05 0.1 0.2 0.4 1 / 0.1 1 10 100 | tuning grids |
| `--no-tune`, `--alpha a`, `--C c` | off, 1, 1 | fixed hyperparameters (and an untuned RBF) |
| `--seeds k …` | 42 | repeat per seed, report mean ± std |
| `--features` | `paper` | `paper` (61) / `all` (77) |
| `--keep-duplicates` | off | do not deduplicate |
| `--no-log` | off | no signed log transform |
| `--legacy-scaling` | off | original global MinMax before the split |
| `--legacy` | off | = legacy scaling + keep duplicates + no tuning + α = 1, C = 1 |
| `--extra-baselines` | off | add a Random Forest |
| `--out-dir DIR` | `outputs/<tag>` | output folder |
| `--quiet` | off | less console output |

---

## 16. Tests, benchmark suite and report generation

```bash
python -m unittest discover -s tests -v       # 19 tests
python benchmark.py                           # all experiments (~1.5–2 h; resumable)
python benchmark.py --quick                   # 1 seed, small grids → outputs/benchmark_quick
python benchmark.py --only headline ablation  # selected experiments
python make_figures.py                        # → report/figures, report/tables
cd report && pdflatex report.tex && pdflatex report.tex   # → report/report.pdf
```

**What the tests check:**
- the fast engine equals Cirq (linear, circular and full entanglement; 2–3 reps; α ≠ 1; the Rx map);
- the new kernel equals the original `legacy/qsvm.py` value;
- kernels are PSD and symmetric with a unit diagonal;
- shot noise is unbiased, `--psd` makes a noisy kernel PSD, and α controls concentration;
- preprocessing never refits on test data;
- the loader is reproducible, removes duplicates (on the task's label, so in binary mode identical LDAP/Portmap flows count as duplicates), balances classes and finds 6 classes;
- the McNemar, alignment and geometric-difference implementations behave correctly on known cases;
- the QSVM learns a toy problem.

**Benchmark experiments:**

| Name | Question it answers |
|---|---|
| `speed` | How much faster is the new kernel computation? |
| `headline` | The paper's 3 methods, original vs improved protocol, 5 seeds |
| `ablation` | Which fix changes what? (legacy → +dedup → +train-only scaling → +log → +tuning) |
| `bandwidth` | Accuracy, concentration and alignment vs α at 8 and 16 qubits |
| `qubits` | 2…16 qubits, ZZ vs Rx vs RBF, PCA variance |
| `shots` | 16…4096 shots vs the exact kernel |
| `architecture` | reps 1–3 × linear/circular/full entanglement |
| `features` | paper vs all features |
| `samples` | learning curve, 250…4000 samples |

---

## 17. Limitations and future work

**Limitations**
- All results are from **classical simulation**. Real hardware adds gate errors, decoherence and readout noise; the shot experiment models only statistical noise.
- The data file holds contiguous slices from one capture day. Real-world generalisation (other days, other networks) is untested.
- Balanced undersampling does not reflect real traffic, where attacks and benign flows are heavily imbalanced.
- PCA to ≤16 components discards information; the Random Forest on all features shows the cost.

**Future work that builds on this code**
1. **Harder data:** the 12 January day (more attack types, including near-identical DNS/NTP/SSDP reflection), cross-day train/test, CICIDS2017, UNSW-NB15. Use `--csv`.
2. **Trainable kernels:** optimise feature-map parameters to maximise kernel–target alignment (`evaluation.centered_alignment` already exists).
3. **Projected quantum kernels** (Huang et al. 2021), which resist concentration better.
4. **Noise models** in Cirq (`cirq.depolarize`) on the Cirq backend, plus error mitigation.
5. **Hardware runs** of a few kernel entries via `build_kernel_circuit()`.
6. **Imbalanced evaluation:** train balanced, test on the natural class ratio, and report precision at fixed FAR.

---

## 18. FAQ

**Q: How does the code run if there's no dataset in the directory?**
The dataset *is* in the directory: `output.csv` (33 MB) is a merged CIC-DDoS2019 extract, read by default via `--csv ./output.csv`.

**Q: Is this "real" quantum computing?**
The circuits are real quantum circuits (Cirq objects you could send to hardware), but they are **simulated** on a CPU. The fast engine computes exactly the same statevectors, faster.

**Q: If the fast engine doesn't use Cirq, is it still a quantum kernel?**
Yes. It computes the *same* quantum states (verified to 10⁻⁷ against Cirq and against the original code) by exploiting the circuit's structure. Cirq remains the reference and can be selected with `--backend cirq`.

**Q: Why did accuracy go *down* in multiclass after the improvements?**
Because almost half of the old test set had identical copies in training (48.6%). The old number measured memorisation; the new number measures generalisation to unseen flows.

**Q: So is the quantum approach useless?**
On *this* data, under a fair protocol, it ties a well-tuned classical kernel. That is a common and honest outcome in quantum ML today. The geometric-difference diagnostic shows *why*: the data's structure is already captured by classical kernels. The improved pipeline is the right tool for checking harder datasets.

**Q: What is α and why should I care?**
The kernel's bandwidth. With α = 1 and many qubits, the quantum kernel becomes ≈ the identity matrix (concentration) and stops encoding useful similarity. CV picks α automatically; it often chooses 0.05–0.4 at 16 qubits.

**Q: Why is the RBF baseline "suddenly" so good?**
It always could be. It was just never tuned (C = 1, γ = default), and it received poorly scaled features. Comparing a tuned model against an untuned one measures the tuning, not the model.

**Q: Can I still run the old code?**
Yes: `python legacy/main.py ...` runs the untouched original. `python main.py --legacy ...` runs the original *protocol* on the fast engine.

**Q: How long do runs take?**
8 qubits: seconds. 16 qubits with tuning: about 2 minutes per seed. The full benchmark: about 1.5–2 hours.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Alignment** | how well kernel similarities match label similarities (cosine between centred matrices) |
| **Amplification / reflection** | DDoS via third-party servers sending large replies to a spoofed victim |
| **Bandwidth (α, γ)** | the scale at which a kernel considers points similar |
| **CICFlowMeter** | packet-capture → flow-statistics tool used to build the dataset |
| **Concentration** | off-diagonal kernel values collapsing to a constant (here ≈ 0) |
| **Cross-validation** | estimating performance by rotating held-out folds of the training data |
| **DR / FAR** | detection rate / false-alarm rate |
| **Entanglement** | correlation between qubits that cannot be described qubit by qubit |
| **Feature map** | circuit that encodes a data vector into a quantum state |
| **Fidelity kernel** | $K(a,b)=|\langle\phi(a)|\phi(b)\rangle|^2$ |
| **Geometric difference g** | test of whether a classical kernel can match a quantum one on given data |
| **Leakage** | test information influencing training |
| **Macro-F1** | mean of per-class F1 scores |
| **McNemar test** | paired significance test for two classifiers on the same test set |
| **PCA** | projection onto the directions of greatest variance |
| **QSVM** | SVM with a quantum-computed kernel |
| **Shots** | repetitions of a circuit to estimate measurement probabilities |
| **Statevector** | the 2ⁿ complex amplitudes of an n-qubit state |
| **ZZ feature map** | Havlíček et al.'s entangling encoding with $Z_iZ_j$ interaction terms |

---

## 20. References

1. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212 (2019).
2. Sharafaldin, Lashkari, Hakak, Ghorbani, *Developing realistic DDoS attack dataset and taxonomy*, IEEE ICCST (2019).
3. Sharafaldin, Lashkari, Ghorbani, *Toward generating a new intrusion detection dataset*, ICISSP (2018).
4. Cortes & Vapnik, *Support-vector networks*, Machine Learning 20 (1995).
5. Platt, *Probabilistic outputs for SVMs* (1999).
6. Pedregosa et al., *Scikit-learn*, JMLR 12 (2011).
7. Cirq Developers, *Cirq*, https://quantumai.google/cirq.
8. Schuld & Killoran, *Quantum ML in feature Hilbert spaces*, PRL 122, 040504 (2019).
9. Schuld, *Supervised quantum ML models are kernel methods*, arXiv:2101.11020 (2021).
10. Thanasilp, Wang, Cerezo, Holmes, *Exponential concentration in quantum kernel methods*, Nat. Commun. 15, 5200 (2024).
11. Shaydulin & Wild, *Importance of kernel bandwidth in quantum ML*, PRA 106, 042407 (2022).
12. Canatar et al., *Bandwidth enables generalization in quantum kernel models*, TMLR (2023).
13. Huang et al., *Power of data in quantum machine learning*, Nat. Commun. 12, 2631 (2021).
14. Bowles, Ahmed, Schuld, *Better than classical? The subtle art of benchmarking QML models*, arXiv:2403.07059 (2024).
15. Cortes, Mohri, Rostamizadeh, *Algorithms for learning kernels based on centered alignment*, JMLR 13 (2012).
16. McNemar, Psychometrika 12(2) (1947); Dietterich, Neural Computation 10(7) (1998).
17. Kaufman et al., *Leakage in data mining*, ACM TKDD 6(4) (2012).
18. Engelen, Rimmer, Joosen, *Troubleshooting an intrusion detection dataset: CICIDS2017*, IEEE SPW (2021).
19. Breiman, *Random forests*, Machine Learning 45 (2001).
