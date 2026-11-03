# QSVM on real IBM quantum hardware — comparative analysis

**Run:** job `dbkn0ftgg49c73dj8k20` on **ibm_kingston** (Heron r2, 156 qubits), 3 Nov 2026.
Queued 29.5 min, ran 6.6 min wall-clock, **174 s of QPU time** (estimate was 2.6 min).

## Setup

| | |
|---|---|
| Task | binary BENIGN vs DDoS, CIC-DDoS2019, duplicates removed |
| Feature map | ZZ, 8 qubits, reps = 2, linear entanglement, α = 0.4, C = 10 (tuned by 5-fold CV on the full 1,600-flow training split) |
| Hardware subset | 40 training flows, 40 test flows (balanced), preprocessing fitted on the full training split |
| Circuits | 2,460 kernel circuits U(b)·U(a)† (780 train pairs + 80 diagonals + 1,600 test–train), 256 shots each; K(a,b) = P(00000000) |
| Compiled circuit | physical qubits [91, 98, 111, 110, 109, 118, 129, 128] (an adjacent chain, no SWAPs); depth ≈ 149; 56 CZ gates |
| Device errors on that layout | median CZ error 1.59×10⁻³, median readout error 4.88×10⁻³ |
| Error suppression | dynamical decoupling (XY4); no twirling |
| Mitigation | divide each entry by √(K(a,a)·K(b,b)) using the measured diagonals, then project the training kernel onto the PSD cone (Hubregtsen et al. 2022) |

## Results

| Kernel source | Test acc. % | AUC | Same predictions as exact | MAE vs exact | max \|Δ\| | Pearson r | mean K(x,x) | λmin(K_train) |
|---|---|---|---|---|---|---|---|---|
| Exact statevector | 97.5 | 0.997 | — | 0 | 0 | 1.000 | 1.000 | ≈ 0 |
| Ideal + 256-shot noise | 97.5 | 1.000 | 100 % | 0.0062 | 0.054 | 0.999 | 1.000 | −0.050 |
| Noisy simulator (Kingston noise model) | 97.5 | 0.997 | 100 % | 0.0179 | 0.225 | 0.998 | 0.822 | −0.096 |
| Noisy simulator, mitigated | 97.5 | 0.997 | 100 % | 0.0081 | 0.100 | 0.998 | 1.005 | ≈ 0 |
| **IBM hardware, raw** | **97.5** | 0.997 | **100 %** | 0.0318 | 0.416 | 0.996 | **0.651** | −0.134 |
| **IBM hardware, mitigated** | **97.5** | 0.997 | **100 %** | **0.0124** | 0.179 | 0.996 | 1.009 | ≈ 0 |
| Classical RBF-SVM (same 40/40 flows) | 92.5 | 1.000 | — | — | — | — | — | — |

Reference: the exact QSVM on the full 1,600 / 400 split scores 99.50 %.

Figure: `ibm_kingston_8q_r2_40x40_s256/hardware_comparison.png`. Raw numbers:
`results.json`. Kernel matrices: `kernels.npz`.

## Findings

1. **The hardware classifier is exactly the simulated one.** All six quantum kernel
   versions, including raw hardware, give the same 40 test predictions as the exact
   statevector kernel (97.5 %, one misclassified flow). Noise changes the kernel
   values, not the decision.

2. **The hardware noise is mostly a uniform shrinkage, and the SVM tolerates it.**
   On the hardware, the circuit U(x)U(x)† should return to |0…0⟩ with probability 1.
   It did so only **65 %** of the time (range 59–72 %). Kernel values near 1 read as
   ≈ 0.6–0.7 (the pink band in the scatter plot). Small values are barely affected.
   The correlation with the exact kernel stays at r = 0.996. A near-uniform scale
   factor is equivalent to rescaling C, and the support vectors do not change.

3. **Mitigation removes 61 % of the error.** Normalising by the measured diagonal
   reduces the mean absolute error from 0.0318 to 0.0124 and the largest error from
   0.42 to 0.18. It also restores a unit diagonal. PSD projection fixes the negative
   eigenvalues (λmin = −0.13 raw), so the kernel is valid again for the SVM.
   The remaining error is 1.5× that of the mitigated simulator (0.0081) and
   about 2× the pure shot-noise floor (0.0062).

4. **The noise model underestimates the real device.** The simulator, built from the
   backend's published error rates, predicted a diagonal fidelity of 0.82. The chip
   delivered 0.65. Calibration data captures gate and readout errors but misses
   crosstalk, coherent errors and drift over the 7-minute run, which all accumulate
   over 56 CZs at depth ≈ 149. The simulator therefore gives an optimistic upper
   bound on hardware quality.

5. **Noise makes the kernel more concentrated.** The mean off-diagonal value drops
   from 0.082 (exact) to 0.057 (raw hardware), so flows look less alike than they
   should. After mitigation it is 0.086, close to exact. At larger α, or with more
   reps or full entanglement, the fidelity would fall further. The signal in small
   kernel entries would then disappear below the shot-noise floor (≈ 1/256).

6. **No quantum advantage.** On these 40 test flows the QSVM beats RBF by
   5 points (39 vs 37 correct), but with n = 40 that difference is two flows and
   not statistically meaningful. On the full split, the feature-map study found no
   configuration significantly better than a tuned RBF-SVM. The hardware run shows
   the method is *executable* on today's devices with no loss of accuracy at
   8 qubits. It does not show that the method is *better*.

## Limits of this experiment

- One job, one seed, one subset. The 40 × 40 size was set by the QPU-time budget:
  the full 1,600 × 400 problem would need about 1.9 million circuits.
- Compute–uncompute needs one circuit per kernel entry, so cost grows as N². The
  exact simulator handles the full problem in seconds at 8 qubits.
- Twirling (`--twirl`) and readout-error mitigation were not used. Both should
  reduce the remaining error further.

## Reproduce

```bash
export TOKEN=<IBM Cloud API key> INSTANCE=<instance CRN>
./venv-qiskit/bin/python hardware.py --job-id dbkn0ftgg49c73dj8k20   # re-download this job (no QPU time)
./venv-qiskit/bin/python hardware.py --dry-run                      # same pipeline, simulated only
```
