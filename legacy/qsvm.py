"""
Quantum SVM (QSVM) — Quantum Network Defense
==============================================
Implements a quantum kernel SVM using Google Cirq.

Feature Map — ZZ Entangling Encoding (Havlíček et al., Nature 2019):
  Layer 1: Rx(θᵢ) on each qubit  — single-qubit encoding
  Layer 2: exp(i·θᵢ·θⱼ·ZZ)       — two-qubit entangling interactions

  This creates entanglement between features, so the kernel:
    K(a,b) = |⟨φ(a)|φ(b)⟩|²
  CANNOT be written as a product over individual qubits.
  No closed-form shortcut exists — full circuit simulation is required.
  This is what makes it genuinely quantum.

  Compare to the old Rx-only encoding:
    K(a,b) = ∏ᵢ cos²((bᵢ-aᵢ)/2)   ← classical formula, no quantum needed
  The ZZ map breaks this factorization.

Architecture:
  PCA: 58 → N_QUBITS features (classical preprocessing)
  ZZ Feature Map (2 repetitions):
    [H gates]  put all qubits in superposition
    [Rz(2θᵢ)] encode feature i
    [ZZ pairs] exp(i·(π-θᵢ)(π-θⱼ)·Z⊗Z) for adjacent pairs
  Quantum kernel: K(x,x') = |⟨0|U†(x)U(x')|0⟩|²  via circuit simulation
  Classical SVM with precomputed kernel matrix

Reference:
  Havlíček et al. (2019) "Supervised learning with quantum-enhanced
  feature spaces", Nature 567, 209–212.
  Kalinin & Krundyshev (2022), J. Computer Virology & Hacking Techniques.
"""

import numpy as np
import cirq
from typing import Optional
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ── ZZ Feature Map ─────────────────────────────────────────────────────────────

class ZZFeatureMap:
    """
    Havlíček-style ZZ entangling feature map.

    For n qubits and feature vector θ = [θ₀, ..., θₙ₋₁]:

    One repetition:
      1. H on all qubits
      2. Rz(2·θᵢ) on qubit i          — encodes feature
      3. For adjacent pairs (i, i+1):
           CNOT(i, i+1)
           Rz(2·(π - θᵢ)(π - θᵢ₊₁)) on qubit i+1   — ZZ interaction
           CNOT(i, i+1)

    The ZZ interaction term creates entanglement between adjacent features,
    making the kernel non-separable — genuinely quantum.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_reps : int
        Number of times to repeat the encoding layer (default 2).
        More reps = more expressive, slower simulation.
    """

    def __init__(self, n_qubits: int = 16, n_reps: int = 2):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.qubits = cirq.LineQubit.range(n_qubits)

    def build_encoding_circuit(self, angles: np.ndarray) -> cirq.Circuit:
        """
        Build ZZ feature map circuit for one sample.

        Parameters
        ----------
        angles : np.ndarray, shape (n_qubits,), values in [0, π]
        """
        assert len(angles) == self.n_qubits
        ops = []

        for _ in range(self.n_reps):
            # Layer 1: Hadamard on all qubits
            ops += [cirq.H(q) for q in self.qubits]

            # Layer 2: Single-qubit Rz encoding
            for i, q in enumerate(self.qubits):
                ops.append(cirq.rz(2.0 * float(angles[i]))(q))

            # Layer 3: ZZ entangling pairs (adjacent qubits)
            for i in range(self.n_qubits - 1):
                q_a = self.qubits[i]
                q_b = self.qubits[i + 1]
                # ZZ rotation: CNOT · Rz(2·(π-θᵢ)(π-θⱼ)) · CNOT
                zz_angle = 2.0 * (np.pi - float(angles[i])) * (np.pi - float(angles[i+1]))
                ops.append(cirq.CNOT(q_a, q_b))
                ops.append(cirq.rz(zz_angle)(q_b))
                ops.append(cirq.CNOT(q_a, q_b))

        return cirq.Circuit(ops)

    def build_kernel_circuit(
        self, angles_a: np.ndarray, angles_b: np.ndarray
    ) -> cirq.Circuit:
        """
        Build U†(a)·U(b) kernel circuit.
        K(a,b) = P(|00...0⟩) after measuring all qubits.
        """
        circuit = cirq.Circuit()
        circuit += self.build_encoding_circuit(angles_b)
        # U†(a) = inverse of U(a): negate all rotation angles
        circuit += cirq.inverse(self.build_encoding_circuit(angles_a))
        circuit += cirq.measure(*self.qubits, key="result")
        return circuit


# ── Quantum Kernel ─────────────────────────────────────────────────────────────

class QuantumKernel:
    """
    Computes quantum kernel matrix K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²

    Uses full statevector simulation — required because the ZZ feature map
    creates entanglement that cannot be computed analytically.

    Memory: 2^n_qubits complex64 amplitudes
      16 qubits → 512 KB  ✅
      20 qubits → 8 MB    ✅
      58 qubits → 128 GB  ❌  (why PCA to 16 is needed)

    Parameters
    ----------
    n_qubits : int
    n_reps : int
        ZZ layer repetitions.
    shots : int
        Used only if use_sampling=True.
    use_sampling : bool
        False (default): exact statevector inner product — fast and exact.
        True: circuit sampling — illustrates real QPU execution, slower.
    """

    def __init__(
        self,
        n_qubits: int = 16,
        n_reps: int = 2,
        shots: int = 1024,
        use_sampling: bool = False,
    ):
        self.n_qubits = n_qubits
        self.shots = shots
        self.use_sampling = use_sampling
        self.feature_map = ZZFeatureMap(n_qubits, n_reps)
        self.simulator = cirq.Simulator()

    def _kernel_statevector(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Exact kernel via statevector inner product.
        K(a,b) = |⟨ψ(a)|ψ(b)⟩|²
        """
        circuit_a = self.feature_map.build_encoding_circuit(a)
        circuit_b = self.feature_map.build_encoding_circuit(b)

        sv_a = self.simulator.simulate(circuit_a).final_state_vector.flatten()
        sv_b = self.simulator.simulate(circuit_b).final_state_vector.flatten()

        inner = np.dot(sv_a.conj(), sv_b)
        return float(np.abs(inner) ** 2)

    def _kernel_sampled(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Sampled kernel via circuit measurement.
        K(a,b) ≈ P(|00...0⟩) from shots repetitions.
        """
        circuit = self.feature_map.build_kernel_circuit(a, b)
        result = self.simulator.run(circuit, repetitions=self.shots)
        all_zero = np.all(result.measurements["result"] == 0, axis=1)
        return float(np.sum(all_zero) / self.shots)

    def compute_kernel_value(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.use_sampling:
            return self._kernel_sampled(a, b)
        return self._kernel_statevector(a, b)

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = K(Xᵢ, Yⱼ).
        If Y is None, computes symmetric training kernel K(X, X).
        """
        symmetric = Y is None
        if symmetric:
            Y = X
        m, n = len(X), len(Y)
        K = np.zeros((m, n), dtype=np.float64)

        total = m * (m + 1) // 2 if symmetric else m * n
        done = 0

        for i in range(m):
            j_start = i if symmetric else 0
            for j in range(j_start, n):
                k_val = self.compute_kernel_value(X[i], Y[j])
                K[i, j] = k_val
                if symmetric:
                    K[j, i] = k_val
                done += 1
                if verbose and done % 100 == 0:
                    print(f"  Kernel: {done}/{total} ({100*done/total:.1f}%)", end="\r")

        if verbose:
            print(f"  Kernel: {total}/{total} (100.0%) ✓   ")
        return K


# ── QSVM Classifier ───────────────────────────────────────────────────────────

class QSVM:
    """
    Quantum SVM with ZZ entangling feature map.

    Pipeline:
      1. PCA: 58 features → n_qubits (classical, fit on train only)
      2. Rescale to [0, π]
      3. ZZ feature map encoding → quantum kernel computation
      4. Classical SVC with precomputed kernel

    Parameters
    ----------
    n_qubits : int
        Qubit count (= PCA target dim). Default 16.
    n_reps : int
        ZZ encoding repetitions. Default 2.
    C : float
        SVM regularization.
    shots : int
        Shots for sampled kernel (if use_sampling=True).
    use_sampling : bool
        Use sampling vs exact statevector kernel.
    """

    def __init__(
        self,
        n_qubits: int = 16,
        n_reps: int = 2,
        C: float = 1.0,
        shots: int = 1024,
        use_sampling: bool = False,
    ):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.C = C
        self.quantum_kernel = QuantumKernel(n_qubits, n_reps, shots, use_sampling)
        self.svc = SVC(kernel="precomputed", C=C, probability=True)
        self.X_train_ = None
        self.pca = None
        self.pca_min = None
        self.pca_max = None

    def _fit_pca(self, X: np.ndarray) -> np.ndarray:
        self.pca = PCA(n_components=self.n_qubits, random_state=42)
        X_pca = self.pca.fit_transform(X)
        self.pca_min = X_pca.min(axis=0)
        self.pca_max = X_pca.max(axis=0)
        var = self.pca.explained_variance_ratio_.sum() * 100
        print(f"[QSVM] PCA: 58 → {self.n_qubits} components, "
              f"variance retained: {var:.1f}%")
        return self._scale(X_pca)

    def _scale(self, X_pca: np.ndarray) -> np.ndarray:
        return (X_pca - self.pca_min) / (self.pca_max - self.pca_min + 1e-8) * np.pi

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._scale(self.pca.transform(X))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QSVM":
        X_enc = self._fit_pca(X_train)
        self.X_train_ = X_enc
        print(f"[QSVM] ZZ feature map: {self.n_qubits} qubits, "
              f"{self.n_reps} reps (entangled — no closed-form kernel)")
        print(f"[QSVM] Computing training kernel ({len(X_train)}×{len(X_train)})...")
        K_train = self.quantum_kernel.compute_kernel_matrix(X_enc, verbose=True)
        self.svc.fit(K_train, y_train)
        print("[QSVM] Training complete.")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_enc = self._transform(X_test)
        K_test = self.quantum_kernel.compute_kernel_matrix(
            X_enc, self.X_train_, verbose=True
        )
        return self.svc.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_enc = self._transform(X_test)
        K_test = self.quantum_kernel.compute_kernel_matrix(
            X_enc, self.X_train_, verbose=True
        )
        return self.svc.predict_proba(K_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return accuracy_score(y_test, self.predict(X_test))
