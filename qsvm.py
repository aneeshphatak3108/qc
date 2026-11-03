"""
Quantum SVM (QSVM) — Quantum Network Defense
==============================================
Implements a quantum kernel SVM. Google Cirq is the reference simulator;
an exact NumPy statevector engine specialised to the ZZ feature map is the
default backend (identical numbers, orders of magnitude faster).

Feature Map — ZZ Entangling Encoding (Havlíček et al., Nature 2019):
  One repetition, for angles x = α·θ  (θ ∈ [0, π] from preprocessing):
    [H gates]   H on every qubit
    [Rz(2xᵢ)]   encode feature i on qubit i
    [ZZ pairs]  CNOT · Rz(2(π-xᵢ)(π-xⱼ)) · CNOT  for each entangled pair (i, j)
  Repeated n_reps times.

  Kernel: K(a, b) = |⟨φ(a)|φ(b)⟩|²

What changed vs. the original implementation (legacy/qsvm.py)
--------------------------------------------------------------
  1. Statevector caching. The original simulated BOTH circuits for EVERY
     kernel entry: 2·N² circuit simulations for an N×N matrix. A feature
     state |φ(x)⟩ only depends on x, so we now simulate each sample ONCE
     and obtain the whole Gram matrix with one complex matrix product:
         K = |Φ_X^* Φ_Yᵀ|²            (N simulations instead of 2·N²)
  2. Fast exact engine ('fast' backend). Every gate in the ZZ map except H
     is diagonal in the computational basis. One repetition is therefore
         U(x) = D(x) · H^{⊗n},   D(x)|z⟩ = exp(-i Φ_x(z)) |z⟩
         Φ_x(z) = Σᵢ xᵢ sᵢ(z) + Σ_(i,j) (π-xᵢ)(π-xⱼ) sᵢ(z) sⱼ(z),  sᵢ = 1-2zᵢ
     so a state is computed with a vectorised phase table plus a fast
     Walsh–Hadamard transform. Verified against Cirq to ~1e-6
     (tests/test_core.py).
  3. Kernel bandwidth α. Angles are multiplied by α before encoding. With
     16 qubits and α = 1 the kernel concentrates (off-diagonal values → 0,
     K ≈ I), which makes the SVM memorise rather than generalise.
     α < 1 restores a useful kernel (Shaydulin & Wild, 2022).
  4. Entanglement topology: 'linear' (original), 'circular', 'full'.
  5. The paper's separable Rx kernel (Method 1) is available as
     feature_map='rx' — K(a,b) = Πᵢ cos²((aᵢ-bᵢ)/2) — so all methods run
     through the same, identical pipeline.
  9. General Pauli feature maps (feature_map='pauli'): any list of Pauli
     strings (['Z'], ['Y', 'ZZ'], ['Z', 'XX'], ['Z', 'ZZ', 'ZZZ'], …),
     k-body entanglement and a choice of data map φ_S
     (havlicek | product | cosine | exp, after Suzuki et al. 2020). Exact
     batched engine: FastPauliStatevector. The ZZ map also accepts data_map.
  6. Fast shot-noise emulation. With use_sampling=True and the fast backend
     each entry is drawn as Binomial(shots, K)/shots. This is exactly the
     distribution of "fraction of all-zero outcomes" obtained by running
     U†(a)U(b) for `shots` repetitions, without running N² circuits.
     backend='cirq' + use_sampling=True still runs the real circuits.
  7. QSVM.predict / predict_proba / decision_function share one cached
     test kernel (the original recomputed it for predict AND predict_proba).
  8. PSD projection of finite-shot training kernels. Shot noise makes the
     estimated Gram matrix indefinite (on 1600 flows the negative
     eigenvalues sum to −80 … −320), so the SVM dual is no longer convex.
     psd_project=True clips negative eigenvalues to 0 (Hubregtsen et al.,
     2022). It is OFF by default: it only fixes the training kernel, and
     the resulting train/test mismatch cost 3.5 points in the 16-qubit
     multiclass shot experiment (88.1% → 84.7%, 5/5 seeds worse).

Reference:
  Havlíček et al. (2019) Nature 567, 209–212.
  Shaydulin & Wild (2022) Phys. Rev. A 106, 042407 (kernel bandwidth).
  Huang et al. (2021) Nature Communications 12, 2631 (power of data).
  Hubregtsen et al. (2022) Phys. Rev. A 106, 042431 (kernel mitigation).
  Suzuki et al. (2020) Quantum Mach. Intell. 2, 9 (feature-map data maps).
"""

import time
from itertools import combinations
from typing import List, Optional, Tuple

import cirq
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from preprocessing import QuantumPreprocessor

INV_SQRT2 = 1.0 / np.sqrt(2.0)


def entangling_pairs(n_qubits: int, entanglement: str = "linear") -> List[Tuple[int, int]]:
    return entangling_subsets(n_qubits, 2, entanglement)


def entangling_subsets(n_qubits: int, k: int,
                       entanglement: str = "linear") -> List[Tuple[int, ...]]:
    """Qubit subsets a k-body Pauli term acts on (k=1: every qubit)."""
    if k == 1:
        return [(i,) for i in range(n_qubits)]
    if k > n_qubits:
        return []
    if entanglement == "linear":
        return [tuple(range(i, i + k)) for i in range(n_qubits - k + 1)]
    if entanglement == "circular":
        subsets = [tuple(range(i, i + k)) for i in range(n_qubits - k + 1)]
        if n_qubits > k:
            subsets += [tuple((i + j) % n_qubits for j in range(k))
                        for i in range(n_qubits - k + 1, n_qubits)]
        return subsets
    if entanglement == "full":
        return list(combinations(range(n_qubits), k))
    raise ValueError(f"unknown entanglement {entanglement!r}")


# ── Data maps φ_S(x) for multi-qubit terms ───────────────────────────────────
# A k-body term on qubit subset S is exp(-i φ_S(x) P_S). One-body terms always
# use φ_i(x) = xᵢ. The variants follow Suzuki et al. (2020), who compared
# several choices of φ for the ZZ map. Input: x_S of shape (..., k), the
# map reduces over the last axis.

DATA_MAPS = {
    "havlicek": lambda xs: np.prod(np.pi - xs, axis=-1),        # Havlíček 2019 / Qiskit default
    "product": lambda xs: np.prod(xs, axis=-1),
    "cosine": lambda xs: np.pi * np.prod(np.cos(xs), axis=-1),
    "exp": lambda xs: np.exp((xs.max(axis=-1) - xs.min(axis=-1)) ** 2 / (8 / np.log(np.pi))),
}


def data_map_fn(name: str):
    if name not in DATA_MAPS:
        raise ValueError(f"unknown data map {name!r} (choose from {sorted(DATA_MAPS)})")
    return DATA_MAPS[name]


# ── Cirq reference feature maps ───────────────────────────────────────────────

class ZZFeatureMap:
    """
    Havlíček-style ZZ entangling feature map built as a Cirq circuit.

    For n qubits and feature vector θ = [θ₀, ..., θₙ₋₁], x = α·θ:

    One repetition:
      1. H on all qubits
      2. Rz(2·xᵢ) on qubit i          — encodes feature
      3. For each entangled pair (i, j):
           CNOT(i, j)
           Rz(2·(π - xᵢ)(π - xⱼ)) on qubit j   — ZZ interaction
           CNOT(i, j)

    n_reps=2, entanglement='linear', alpha=1.0 is exactly the original map.
    """

    def __init__(self, n_qubits: int = 16, n_reps: int = 2,
                 entanglement: str = "linear", alpha: float = 1.0,
                 data_map: str = "havlicek"):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.alpha = alpha
        self.pairs = entangling_pairs(n_qubits, entanglement)
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.data_map = data_map_fn(data_map)

    def build_encoding_circuit(self, angles: np.ndarray) -> cirq.Circuit:
        assert len(angles) == self.n_qubits
        x = self.alpha * np.asarray(angles, dtype=np.float64)
        ops = []
        for _ in range(self.n_reps):
            ops += [cirq.H(q) for q in self.qubits]
            for i, q in enumerate(self.qubits):
                ops.append(cirq.rz(2.0 * x[i])(q))
            for i, j in self.pairs:
                zz_angle = 2.0 * self.data_map(x[None, [i, j]])[0]
                ops.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                ops.append(cirq.rz(zz_angle)(self.qubits[j]))
                ops.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
        return cirq.Circuit(ops)

    def build_kernel_circuit(self, angles_a: np.ndarray,
                             angles_b: np.ndarray) -> cirq.Circuit:
        """U†(a)·U(b); K(a,b) = P(|00...0⟩) after measuring all qubits."""
        circuit = cirq.Circuit()
        circuit += self.build_encoding_circuit(angles_b)
        circuit += cirq.inverse(self.build_encoding_circuit(angles_a))
        circuit += cirq.measure(*self.qubits, key="result")
        return circuit


class RxFeatureMap(ZZFeatureMap):
    """Separable Rx(xᵢ) product encoding — the paper's 'Analytical' method."""

    def __init__(self, n_qubits: int = 16, alpha: float = 1.0):
        super().__init__(n_qubits, n_reps=1, entanglement="linear", alpha=alpha)
        self.pairs = []

    def build_encoding_circuit(self, angles: np.ndarray) -> cirq.Circuit:
        x = self.alpha * np.asarray(angles, dtype=np.float64)
        return cirq.Circuit(cirq.rx(x[i])(q) for i, q in enumerate(self.qubits))


class PauliFeatureMap(ZZFeatureMap):
    """
    General Pauli feature map (Havlíček et al. 2019; Qiskit's PauliFeatureMap).

    One repetition: H on every qubit, then for every Pauli string P in
    `paulis` (e.g. ['Z', 'ZZ'], ['Y', 'XX'], ['Z', 'ZZ', 'ZZZ']) and every
    qubit subset S given by `entanglement`:
        exp(-i φ_S(x) P_S),   φ_i(x) = xᵢ,   φ_S(x) = data_map(x_S) for |S| > 1
    Letter m of P acts on qubit S[m]. paulis=['Z', 'ZZ'] is the ZZ map.

    Each term is built the textbook way: rotate every qubit of S into the
    Z basis (X: H, Y: H·S†), CNOT ladder, Rz(2φ) on the last qubit, undo.
    """

    def __init__(self, n_qubits: int = 16, n_reps: int = 2, paulis=("Z", "ZZ"),
                 entanglement: str = "linear", alpha: float = 1.0,
                 data_map: str = "havlicek"):
        super().__init__(n_qubits, n_reps, entanglement, alpha, data_map)
        self.terms = pauli_terms(n_qubits, paulis, entanglement)

    def build_encoding_circuit(self, angles: np.ndarray) -> cirq.Circuit:
        x = self.alpha * np.asarray(angles, dtype=np.float64)
        ops = []
        for _ in range(self.n_reps):
            ops += [cirq.H(q) for q in self.qubits]
            for letters, S in self.terms:
                phi = x[S[0]] if len(S) == 1 else self.data_map(x[None, list(S)])[0]
                qs = [self.qubits[i] for i in S]
                pre, post = [], []
                for p, q in zip(letters, qs):
                    if p == "X":
                        pre.append(cirq.H(q)); post.append(cirq.H(q))
                    elif p == "Y":
                        pre += [cirq.S(q) ** -1, cirq.H(q)]; post += [cirq.H(q), cirq.S(q)]
                ladder = [cirq.CNOT(qs[m], qs[m + 1]) for m in range(len(qs) - 1)]
                ops += pre + ladder + [cirq.rz(2.0 * phi)(qs[-1])] + ladder[::-1] + post
        return cirq.Circuit(ops)


def pauli_terms(n_qubits: int, paulis, entanglement: str = "linear"):
    """[(letters, subset), ...] in application order for one repetition."""
    terms = []
    for p in paulis:
        p = p.upper()
        if not p or set(p) - set("XYZ"):
            raise ValueError(f"invalid Pauli string {p!r}")
        terms += [(p, S) for S in entangling_subsets(n_qubits, len(p), entanglement)]
    return terms


# ── Fast exact statevector engine ────────────────────────────────────────────

def _hadamard_matrix(n: int) -> np.ndarray:
    """Normalised H^{⊗n} as a dense (2ⁿ, 2ⁿ) complex64 matrix."""
    H = np.ones((1, 1))
    for _ in range(n):
        H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]]) * INV_SQRT2)
    return H.astype(np.complex64)


class FastZZStatevector:
    """
    Exact statevectors of the ZZ feature map using its diagonal structure.

    Qubit 0 is the most significant bit, matching Cirq's ordering, so the
    vectors are directly comparable with cirq.Simulator output.
    """

    def __init__(self, n_qubits: int, n_reps: int = 2,
                 entanglement: str = "linear", alpha: float = 1.0,
                 data_map: str = "havlicek"):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.alpha = alpha
        self.data_map = data_map_fn(data_map)
        self.dim = 2 ** n_qubits
        self.pairs = entangling_pairs(n_qubits, entanglement)
        idx = np.arange(self.dim)
        bits = (idx[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
        self.S = (1 - 2 * bits).astype(np.float64)                 # (dim, n)
        if self.pairs:
            a, b = np.array(self.pairs).T
            self.SP = self.S[:, a] * self.S[:, b]                  # (dim, P)
            self._pa, self._pb = a, b
        n1 = n_qubits // 2
        self._H1 = _hadamard_matrix(n1)
        self._H2 = _hadamard_matrix(n_qubits - n1)

    def phases(self, angles: np.ndarray) -> np.ndarray:
        x = self.alpha * np.atleast_2d(angles)
        phi = x @ self.S.T
        if self.pairs:
            w = self.data_map(np.stack([x[:, self._pa], x[:, self._pb]], axis=-1))
            phi += w @ self.SP.T
        return phi

    def _hadamard_all(self, psi: np.ndarray) -> np.ndarray:
        # H^{⊗n} = H^{⊗n1} ⊗ H^{⊗n2}: with psi reshaped to a (2^n1, 2^n2)
        # matrix M, (A ⊗ B)·vec(M) = vec(A · M · Bᵀ) — two small matmuls.
        B = psi.shape[0]
        M = psi.reshape(B, self._H1.shape[0], self._H2.shape[0])
        return (self._H1 @ M @ self._H2.T).reshape(B, -1)

    def states(self, angles: np.ndarray, batch_size: int = 128) -> np.ndarray:
        angles = np.atleast_2d(angles)
        out = np.empty((len(angles), self.dim), dtype=np.complex64)
        for s in range(0, len(angles), batch_size):
            phi = self.phases(angles[s:s + batch_size])
            D = (np.cos(phi) - 1j * np.sin(phi)).astype(np.complex64)
            psi = D * self.dim ** -0.5                 # D · H^{⊗n} |0…0⟩
            for _ in range(self.n_reps - 1):
                psi = self._hadamard_all(psi) * D
            out[s:s + batch_size] = psi
        return out


class FastPauliStatevector(FastZZStatevector):
    """
    Exact statevectors of a general PauliFeatureMap, batched over samples.

    P_S is a Pauli string, so P_S² = I and
        exp(-i φ P_S) ψ = cos φ · ψ − i sin φ · P_S ψ.
    P_S|z⟩ = c(z)|z ⊕ m⟩, where m flips the bits of the X/Y qubits and
    c(z) = Πₖ cₖ(zₖ), with c_X = 1, c_Y = i(−1)^z and c_Z = (−1)^z. So
    P_S ψ is a fixed index permutation times a fixed phase vector. Runs of
    consecutive Z-only terms commute and are merged into one diagonal phase,
    as in FastZZStatevector. One-qubit X/Y terms skip the permutation and
    apply their 2×2 rotation on a reshaped view of ψ.
    """

    def __init__(self, n_qubits: int, n_reps: int = 2, paulis=("Z", "ZZ"),
                 entanglement: str = "linear", alpha: float = 1.0,
                 data_map: str = "havlicek"):
        super().__init__(n_qubits, n_reps, "linear", alpha, data_map)
        bits = (1 - self.S) / 2                                    # (dim, n) z-bits
        idx = np.arange(self.dim)
        # blocks: ("diag", [subsets], (dim, T) signs), ("1q", subset, letter, None)
        #         or ("pauli", subset, src, coef)
        self.blocks = []
        for letters, S in pauli_terms(n_qubits, paulis, entanglement):
            if set(letters) == {"Z"}:
                sign = np.prod(self.S[:, list(S)], axis=1)
                if self.blocks and self.blocks[-1][0] == "diag":
                    self.blocks[-1][1].append(S)
                    self.blocks[-1][2].append(sign)
                else:
                    self.blocks.append(("diag", [S], [sign]))
                continue
            if len(S) == 1:
                self.blocks.append(("1q", S, letters, None))
                continue
            mask = sum(1 << (n_qubits - 1 - q) for p, q in zip(letters, S) if p in "XY")
            src = idx ^ mask
            coef = np.ones(self.dim, dtype=np.complex128)
            for p, q in zip(letters, S):
                z = bits[src, q]                                   # bit of the source state
                if p == "Y":
                    coef *= 1j * (1 - 2 * z)
                elif p == "Z":
                    coef *= 1 - 2 * z
            self.blocks.append(("pauli", S, src, coef.astype(np.complex64)))
        self.blocks = [(b[0], b[1], np.stack(b[2], axis=1), None) if b[0] == "diag" else b
                       for b in self.blocks]

    def _phi(self, x: np.ndarray, S) -> np.ndarray:
        return x[:, S[0]] if len(S) == 1 else self.data_map(x[:, list(S)])

    def states(self, angles: np.ndarray, batch_size: int = 128) -> np.ndarray:
        angles = np.atleast_2d(angles)
        out = np.empty((len(angles), self.dim), dtype=np.complex64)
        for s in range(0, len(angles), batch_size):
            x = self.alpha * angles[s:s + batch_size]
            psi = np.full((len(x), self.dim), self.dim ** -0.5, dtype=np.complex64)
            for r in range(self.n_reps):
                if r:
                    psi = self._hadamard_all(psi)
                for kind, S, a, b in self.blocks:
                    if kind == "diag":
                        phi = np.stack([self._phi(x, T) for T in S], axis=1) @ a.T
                        psi *= (np.cos(phi) - 1j * np.sin(phi)).astype(np.complex64)
                        continue
                    phi = self._phi(x, S).astype(np.float32)
                    c, sn = np.cos(phi), np.sin(phi)
                    if kind == "1q":
                        # exp(-iφX) = [[c, -is], [-is, c]],  exp(-iφY) = [[c, -s], [s, c]]
                        q = S[0]
                        v = psi.reshape(len(x), 2 ** q, 2, -1)
                        c, sn = c[:, None, None], sn[:, None, None]
                        p0, p1 = v[:, :, 0, :].copy(), v[:, :, 1, :]
                        off = -1j * sn if a == "X" else sn
                        v[:, :, 0, :] = c * p0 + (-off if a == "Y" else off) * p1
                        v[:, :, 1, :] = off * p0 + c * p1
                    else:
                        psi = c[:, None] * psi - 1j * sn[:, None] * (b[None, :] * psi[:, a])
            out[s:s + batch_size] = psi
        return out


def nearest_psd(K: np.ndarray) -> np.ndarray:
    """Closest PSD matrix in Frobenius norm: clip negative eigenvalues to 0."""
    w, V = np.linalg.eigh(K)
    return (V * np.maximum(w, 0.0)) @ V.T


# ── Quantum Kernel ─────────────────────────────────────────────────────────────

class QuantumKernel:
    """
    Computes the quantum kernel matrix K[i,j] = |⟨φ(xᵢ)|φ(yⱼ)⟩|².

    Parameters
    ----------
    n_qubits, n_reps : circuit size (n_reps ignored for feature_map='rx')
    shots : int        shots per kernel entry when use_sampling=True
    use_sampling : bool
        False: exact kernel.
        True + backend='fast': Binomial(shots, K)/shots shot-noise emulation.
        True + backend='cirq': run U†(a)U(b) circuits (original method).
    feature_map : 'zz', 'rx' or 'pauli'
    entanglement : 'linear' | 'circular' | 'full'
    alpha : float      bandwidth — angles are multiplied by alpha
    backend : 'fast' (NumPy, exact) or 'cirq' (Cirq statevector simulator)
    seed : int         RNG seed for shot-noise emulation
    psd_project : bool clip negative eigenvalues of sampled training kernels
    paulis : list      Pauli strings for feature_map='pauli', e.g. ['Z', 'ZZ']
    data_map : str     φ_S for multi-qubit terms (zz / pauli), see DATA_MAPS
    """

    def __init__(
        self,
        n_qubits: int = 16,
        n_reps: int = 2,
        shots: int = 1024,
        use_sampling: bool = False,
        feature_map: str = "zz",
        entanglement: str = "linear",
        alpha: float = 1.0,
        backend: str = "fast",
        seed: Optional[int] = 42,
        psd_project: bool = False,
        paulis=("Z", "ZZ"),
        data_map: str = "havlicek",
    ):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.shots = shots
        self.psd_project = psd_project
        self.use_sampling = use_sampling
        self.feature_map_name = feature_map
        self.alpha = alpha
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        if feature_map == "zz":
            self.feature_map = ZZFeatureMap(n_qubits, n_reps, entanglement, alpha, data_map)
            self.engine = FastZZStatevector(n_qubits, n_reps, entanglement, alpha, data_map)
        elif feature_map == "pauli":
            self.feature_map = PauliFeatureMap(n_qubits, n_reps, paulis, entanglement,
                                               alpha, data_map)
            self.engine = FastPauliStatevector(n_qubits, n_reps, paulis, entanglement,
                                               alpha, data_map)
        elif feature_map == "rx":
            self.feature_map = RxFeatureMap(n_qubits, alpha)
            self.engine = None
        else:
            raise ValueError(f"unknown feature_map {feature_map!r}")
        self.simulator = cirq.Simulator(dtype=np.complex64)
        self.n_state_simulations = 0
        self._state_cache = {}

    # -- states ---------------------------------------------------------------
    def states(self, X: np.ndarray) -> np.ndarray:
        """
        Feature statevectors |φ(x)⟩ for every row of X, shape (N, 2ⁿ).
        The most recent results are memoised so the training states are not
        re-simulated when the test kernel K(X_test, X_train) is built.
        """
        key = (X.shape, hash(np.ascontiguousarray(X).tobytes()))
        if key not in self._state_cache:
            if len(self._state_cache) >= 2:
                self._state_cache.pop(next(iter(self._state_cache)))
            self._state_cache[key] = self._simulate(X)
        return self._state_cache[key]

    def _simulate(self, X: np.ndarray) -> np.ndarray:
        self.n_state_simulations += len(X)
        if self.backend == "fast":
            if self.engine is not None:
                return self.engine.states(X)
            return self._rx_states(X)
        if self.backend == "cirq":
            return np.stack([
                self.simulator.simulate(self.feature_map.build_encoding_circuit(x))
                .final_state_vector.reshape(-1) for x in X
            ]).astype(np.complex64)
        raise ValueError(f"unknown backend {self.backend!r}")

    def _rx_states(self, X: np.ndarray) -> np.ndarray:
        # ⊗ᵢ Rx(xᵢ)|0⟩ = ⊗ᵢ (cos(xᵢ/2), -i sin(xᵢ/2))
        x = self.alpha * np.atleast_2d(X)
        psi = np.ones((len(x), 1), dtype=np.complex64)
        for i in range(self.n_qubits):
            q = np.stack([np.cos(x[:, i] / 2), -1j * np.sin(x[:, i] / 2)], axis=1)
            psi = (psi[:, :, None] * q[:, None, :]).reshape(len(x), -1)
        return psi.astype(np.complex64)

    # -- single entries (API kept from the original) --------------------------
    def compute_kernel_value(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.use_sampling and self.backend == "cirq":
            return self._kernel_sampled(a, b)
        return float(self.compute_kernel_matrix(a[None], b[None], verbose=False)[0, 0])

    def _kernel_sampled(self, a: np.ndarray, b: np.ndarray) -> float:
        circuit = self.feature_map.build_kernel_circuit(a, b)
        result = self.simulator.run(circuit, repetitions=self.shots)
        all_zero = np.all(result.measurements["result"] == 0, axis=1)
        return float(np.sum(all_zero) / self.shots)

    # -- matrices -------------------------------------------------------------
    def _exact_matrix(self, X: np.ndarray, Y: Optional[np.ndarray]) -> np.ndarray:
        if self.feature_map_name == "rx" and self.backend == "fast":
            # closed form: Πᵢ cos²(α(bᵢ-aᵢ)/2), evaluated in row blocks
            Yr = X if Y is None else Y
            K = np.empty((len(X), len(Yr)))
            for s in range(0, len(X), 256):
                d = X[s:s + 256, None, :] - Yr[None, :, :]
                K[s:s + 256] = np.prod(np.cos(self.alpha * d / 2.0) ** 2, axis=2)
            return K
        SX = self.states(X)
        SY = SX if Y is None else self.states(Y)
        K = np.empty((len(X), len(SY)))
        for s in range(0, len(SX), 512):
            K[s:s + 512] = np.abs(SX[s:s + 512].conj() @ SY.T) ** 2
        return K

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        K[i,j] = K(Xᵢ, Yⱼ). If Y is None the symmetric training kernel
        K(X, X) is returned with an exact unit diagonal.
        """
        symmetric = Y is None
        t0 = time.time()

        if self.use_sampling and self.backend == "cirq":
            K = self._sampled_matrix_cirq(X, Y, verbose)
        else:
            K = self._exact_matrix(X, Y)
            if self.use_sampling:
                K = self._shot_noise(K, symmetric)

        if symmetric:
            K = (K + K.T) / 2.0
            np.fill_diagonal(K, 1.0)
        K = np.clip(K, 0.0, 1.0)
        if symmetric and self.use_sampling and self.psd_project:
            K = nearest_psd(K)
        if verbose:
            print(f"  Kernel {K.shape[0]}×{K.shape[1]} computed in {time.time()-t0:.2f}s "
                  f"[{self.feature_map_name}, backend={self.backend}"
                  f"{', shots=%d' % self.shots if self.use_sampling else ''}]")
        return K

    def _shot_noise(self, K: np.ndarray, symmetric: bool) -> np.ndarray:
        noisy = self.rng.binomial(self.shots, np.clip(K, 0.0, 1.0)) / self.shots
        if symmetric:
            iu = np.triu_indices_from(noisy, k=1)
            noisy.T[iu] = noisy[iu]
        return noisy

    def _sampled_matrix_cirq(self, X, Y, verbose) -> np.ndarray:
        symmetric = Y is None
        Y = X if symmetric else Y
        m, n = len(X), len(Y)
        K = np.zeros((m, n))
        total = m * (m + 1) // 2 if symmetric else m * n
        done = 0
        for i in range(m):
            for j in range(i if symmetric else 0, n):
                K[i, j] = self._kernel_sampled(X[i], Y[j])
                if symmetric:
                    K[j, i] = K[i, j]
                done += 1
                if verbose and done % 100 == 0:
                    print(f"  Kernel: {done}/{total} ({100*done/total:.1f}%)", end="\r")
        return K


# ── QSVM Classifier ───────────────────────────────────────────────────────────

class QSVM:
    """
    Quantum SVM: preprocessing → quantum kernel → SVC(kernel='precomputed').

    Parameters
    ----------
    n_qubits : int         qubit count (= PCA target dimension)
    n_reps : int           ZZ repetitions
    C : float              SVM regularisation
    shots, use_sampling    see QuantumKernel
    feature_map : 'zz' | 'rx' | 'pauli'
    entanglement : 'linear' | 'circular' | 'full'
    alpha : float          kernel bandwidth
    backend : 'fast' | 'cirq'
    preprocessor           object with fit_transform/transform; defaults to a
                           leakage-free QuantumPreprocessor(n_qubits)
    seed : int
    psd_project : bool     see QuantumKernel
    paulis, data_map       see QuantumKernel
    """

    def __init__(
        self,
        n_qubits: int = 16,
        n_reps: int = 2,
        C: float = 1.0,
        shots: int = 1024,
        use_sampling: bool = False,
        feature_map: str = "zz",
        entanglement: str = "linear",
        alpha: float = 1.0,
        backend: str = "fast",
        preprocessor=None,
        seed: int = 42,
        psd_project: bool = False,
        paulis=("Z", "ZZ"),
        data_map: str = "havlicek",
    ):
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.C = C
        self.seed = seed
        self.quantum_kernel = QuantumKernel(
            n_qubits, n_reps, shots, use_sampling, feature_map,
            entanglement, alpha, backend, seed, psd_project, paulis, data_map,
        )
        self.prep = preprocessor or QuantumPreprocessor(n_qubits, seed=seed)
        self.svc = SVC(kernel="precomputed", C=C, probability=True, random_state=seed)
        self.X_train_ = None
        self.K_train_ = None
        self._cache = (None, None)

    @property
    def pca(self):
        return self.prep.pca_

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            K_train: Optional[np.ndarray] = None) -> "QSVM":
        """Fit on raw features. Pass K_train to reuse an already computed kernel."""
        self.X_train_ = self.prep.fit_transform(X_train)
        if K_train is None:
            K_train = self.quantum_kernel.compute_kernel_matrix(self.X_train_)
        self.K_train_ = K_train
        self.svc.fit(K_train, y_train)
        return self

    def test_kernel(self, X_test: np.ndarray) -> np.ndarray:
        key = (X_test.shape, hash(np.ascontiguousarray(X_test).tobytes()))
        if self._cache[0] != key:
            A = self.prep.transform(X_test)
            K = self.quantum_kernel.compute_kernel_matrix(A, self.X_train_)
            self._cache = (key, K)
        return self._cache[1]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.svc.predict(self.test_kernel(X_test))

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.svc.predict_proba(self.test_kernel(X_test))

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        return self.svc.decision_function(self.test_kernel(X_test))

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return accuracy_score(y_test, self.predict(X_test))
