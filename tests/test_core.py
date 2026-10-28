"""
Correctness tests for the improved pipeline.

Run:  ./venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader import duplicate_overlap, load_csv          # noqa: E402
from evaluation import centered_alignment, geometric_difference, mcnemar  # noqa: E402
from preprocessing import QuantumPreprocessor                # noqa: E402
from qsvm import QSVM, QuantumKernel, entangling_pairs       # noqa: E402

CSV = os.path.join(ROOT, "output.csv")


class TestFastEngineMatchesCirq(unittest.TestCase):
    """The NumPy engine must reproduce Cirq's statevectors exactly."""

    def _compare(self, feature_map, n=6, reps=2, ent="linear", alpha=1.0):
        X = np.random.default_rng(0).uniform(0, np.pi, (5, n))
        fast = QuantumKernel(n, reps, feature_map=feature_map, entanglement=ent,
                             alpha=alpha, backend="fast")
        ref = QuantumKernel(n, reps, feature_map=feature_map, entanglement=ent,
                            alpha=alpha, backend="cirq")
        np.testing.assert_allclose(fast.states(X), ref.states(X), atol=1e-5)
        np.testing.assert_allclose(fast.compute_kernel_matrix(X, verbose=False),
                                   ref.compute_kernel_matrix(X, verbose=False), atol=1e-5)

    def test_zz_linear(self):
        self._compare("zz")

    def test_zz_full_three_reps(self):
        self._compare("zz", n=5, reps=3, ent="full")

    def test_zz_circular_bandwidth(self):
        self._compare("zz", ent="circular", alpha=0.3)

    def test_rx_closed_form(self):
        self._compare("rx", alpha=0.7)

    def test_original_kernel_value_unchanged(self):
        """α=1, linear, 2 reps must equal the original per-pair Cirq kernel."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "legacy_qsvm", os.path.join(ROOT, "legacy", "qsvm.py"))
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        a, b = np.random.default_rng(1).uniform(0, np.pi, (2, 6))
        old = legacy.QuantumKernel(6, 2).compute_kernel_value(a, b)
        new = QuantumKernel(6, 2).compute_kernel_value(a, b)
        self.assertAlmostEqual(new, old, places=5)


class TestKernelProperties(unittest.TestCase):

    def test_psd_unit_diagonal_symmetric(self):
        X = np.random.default_rng(2).uniform(0, np.pi, (40, 8))
        K = QuantumKernel(8, 2, alpha=0.5).compute_kernel_matrix(X, verbose=False)
        np.testing.assert_allclose(np.diag(K), 1.0)
        np.testing.assert_allclose(K, K.T)
        self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-5)

    def test_shot_noise_is_unbiased(self):
        X = np.random.default_rng(3).uniform(0, np.pi, (30, 4))
        exact = QuantumKernel(4, 2, alpha=0.3).compute_kernel_matrix(X, verbose=False)
        noisy = np.mean([
            QuantumKernel(4, 2, shots=256, use_sampling=True, alpha=0.3, seed=s)
            .compute_kernel_matrix(X, verbose=False) for s in range(40)], axis=0)
        self.assertLess(np.abs(noisy - exact).max(), 0.03)

    def test_psd_projection_of_shot_noise_kernel(self):
        X = np.random.default_rng(9).uniform(0, np.pi, (60, 6))
        kw = dict(shots=64, use_sampling=True, alpha=0.05, seed=0)
        raw = QuantumKernel(6, 2, **kw).compute_kernel_matrix(X, verbose=False)
        psd = QuantumKernel(6, 2, psd_project=True, **kw).compute_kernel_matrix(
            X, verbose=False)
        self.assertLess(np.linalg.eigvalsh(raw).min(), -1e-3)
        self.assertGreater(np.linalg.eigvalsh(psd).min(), -1e-8)

    def test_bandwidth_controls_concentration(self):
        X = np.random.default_rng(4).uniform(0, np.pi, (40, 12))
        off = []
        for a in (0.05, 1.0):
            K = QuantumKernel(12, 2, alpha=a).compute_kernel_matrix(X, verbose=False)
            off.append((K.sum() - 40) / (40 * 39))
        self.assertGreater(off[0], 10 * off[1])

    def test_entangling_pairs(self):
        self.assertEqual(len(entangling_pairs(5, "linear")), 4)
        self.assertEqual(len(entangling_pairs(5, "circular")), 5)
        self.assertEqual(len(entangling_pairs(5, "full")), 10)


class TestPreprocessingNoLeakage(unittest.TestCase):

    def test_statistics_come_from_train_only(self):
        rng = np.random.default_rng(5)
        X_train = rng.lognormal(size=(200, 10))
        X_test = rng.lognormal(size=(50, 10)) * 100        # very different scale
        p = QuantumPreprocessor(4).fit(X_train)
        before = p.transform(X_train).copy()
        p.transform(X_test)                                  # must not refit
        np.testing.assert_allclose(p.transform(X_train), before)
        A = p.transform(X_test)
        self.assertTrue((A >= 0).all() and (A <= np.pi).all())


class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CSV):
            raise unittest.SkipTest("output.csv not found")

    def test_dedup_and_reproducible(self):
        X1, y1, names, info = load_csv(CSV, max_samples=300, seed=7, verbose=False)
        X2, y2, _, _ = load_csv(CSV, max_samples=300, seed=7, verbose=False)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        self.assertGreater(info["duplicates_removed"], 40000)
        self.assertEqual(len(np.unique(X1, axis=0)), len(X1))
        self.assertEqual(names, {0: "BENIGN", 1: "DDoS"})
        self.assertEqual(int((y1 == 0).sum()), int((y1 == 1).sum()))

    def test_binary_dedup_ignores_attack_type(self):
        # Identical flows labelled e.g. LDAP and Portmap are both "DDoS" in
        # binary mode, so they must count as duplicates there.
        _, _, _, b = load_csv(CSV, mode="binary", max_samples=300, verbose=False)
        _, _, _, m = load_csv(CSV, mode="multiclass", max_samples=300, verbose=False)
        self.assertGreater(b["duplicates_removed"], m["duplicates_removed"])

    def test_multiclass_has_six_classes(self):
        _, y, names, _ = load_csv(CSV, mode="multiclass", max_samples=600, verbose=False)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(np.unique(y)), 6)

    def test_duplicate_overlap(self):
        a = np.arange(12.0).reshape(6, 2)
        self.assertAlmostEqual(duplicate_overlap(a[:4], a[2:]), 0.5)


class TestEvaluation(unittest.TestCase):

    def test_alignment_perfect_kernel(self):
        y = np.array([0, 0, 1, 1])
        K = (y[:, None] == y[None, :]).astype(float)
        self.assertAlmostEqual(centered_alignment(K, y), 1.0, places=6)

    def test_geometric_difference_identical_kernels(self):
        X = np.random.default_rng(6).normal(size=(30, 3))
        K = np.exp(-np.sum((X[:, None] - X[None]) ** 2, axis=2))
        self.assertLess(geometric_difference(K, K), 1.1)

    def test_mcnemar(self):
        y = np.zeros(10, dtype=int)
        r = mcnemar(y, y, np.ones(10, dtype=int))
        self.assertEqual(r["a_right_b_wrong"], 10)
        self.assertLess(r["p_value"], 0.01)


class TestEndToEnd(unittest.TestCase):

    def test_qsvm_learns_toy_problem(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(120, 6))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        m = QSVM(n_qubits=4, alpha=0.3, C=10).fit(X[:90], y[:90])
        self.assertGreater(m.score(X[90:], y[90:]), 0.8)


if __name__ == "__main__":
    unittest.main()
