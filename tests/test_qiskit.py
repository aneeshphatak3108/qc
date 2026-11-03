"""
Our feature maps against Qiskit's circuit library (needs the Qiskit env).

Run:  ./venv-qiskit/bin/python -m unittest tests.test_qiskit -v
Skipped when Qiskit is not installed (e.g. in the Python 3.8 venv).
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from qiskit.circuit.library import pauli_feature_map
    from qiskit.quantum_info import Statevector
except ImportError:                                           # pragma: no cover
    pauli_feature_map = None

from qsvm import QuantumKernel                                # noqa: E402


@unittest.skipIf(pauli_feature_map is None, "qiskit not installed")
class TestQiskitEquivalence(unittest.TestCase):

    def test_pauli_maps_match_qiskit(self):
        n = 5
        X = np.random.default_rng(0).uniform(0, np.pi, (3, n))
        for paulis in [["Z"], ["Z", "ZZ"], ["Y", "ZZ"], ["Z", "XX"], ["Z", "YY"],
                       ["Z", "ZZ", "ZZZ"]]:
            for ent in ["linear", "circular", "full"]:
                fm = pauli_feature_map(n, reps=2, paulis=paulis, entanglement=ent)
                # Qiskit is little-endian (qubit 0 = least significant bit)
                q = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
                q = q.reshape(3, *[2] * n).transpose(0, *range(n, 0, -1)).reshape(3, -1)
                ours = QuantumKernel(n, 2, feature_map="pauli", paulis=paulis,
                                     entanglement=ent).states(X)
                np.testing.assert_allclose(np.abs(np.sum(q.conj() * ours, axis=1)), 1.0,
                                           atol=1e-5, err_msg=f"{paulis} {ent}")

    def test_hardware_kernel_circuits_match_exact_kernel(self):
        from hardware import kernel_circuit
        n, reps, alpha = 4, 2, 0.4
        X = np.random.default_rng(1).uniform(0, np.pi, (3, n))
        exact = QuantumKernel(n, reps, alpha=alpha).compute_kernel_matrix(X, verbose=False)
        for i in range(3):
            for j in range(3):
                qc = kernel_circuit(X[i], X[j], n, reps, "linear", alpha, measure=False)
                p0 = abs(Statevector(qc).data[0]) ** 2
                self.assertAlmostEqual(p0, exact[i, j], places=5)


if __name__ == "__main__":
    unittest.main()
