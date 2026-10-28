"""
Evaluation, model selection and kernel diagnostics
===================================================

Metrics      accuracy, balanced accuracy, macro-F1, MCC, ROC-AUC
             (binary: decision function; multiclass: one-vs-rest macro),
             and IDS-specific rates: false-alarm rate (BENIGN flagged as
             attack) and detection rate (attacks flagged as attack).
Significance exact McNemar test between two classifiers on one test set.
Tuning       stratified k-fold CV on the TRAINING kernel only, so the test
             set never influences hyperparameters (C, α, RBF γ).
Diagnostics  kernel concentration (off-diagonal mean / std), centred
             kernel–target alignment, and the geometric difference
             g(K_C ‖ K_Q) of Huang et al. (2021), which upper-bounds how much
             better a quantum kernel can be than a classical one on this data.
"""

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.linalg import eigh, sqrtm
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, scores=None, benign_label: int = 0) -> Dict:
    labels = np.unique(y_true)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    benign = y_true == benign_label
    m["false_alarm_rate"] = float(np.mean(y_pred[benign] != benign_label)) if benign.any() else None
    m["detection_rate"] = float(np.mean(y_pred[~benign] != benign_label)) if (~benign).any() else None
    m["auc"] = None
    if scores is not None:
        try:
            if len(labels) == 2:
                m["auc"] = roc_auc_score(y_true, scores)
            else:
                m["auc"] = roc_auc_score(y_true, scores, multi_class="ovr", average="macro")
        except ValueError:
            pass
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def mcnemar(y_true, pred_a, pred_b) -> Dict:
    """Exact McNemar test: do classifiers A and B err on different samples?"""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))      # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))      # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}


# ── Model selection ───────────────────────────────────────────────────────────

def cv_score_precomputed(K: np.ndarray, y: np.ndarray, C: float,
                         folds: int = 5, seed: int = 42) -> float:
    """Mean macro-F1 of SVC(precomputed) over stratified folds of K_train."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(K, y):
        svc = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(f1_score(y[va], svc.predict(K[np.ix_(va, tr)]), average="macro"))
    return float(np.mean(scores))


def tune_quantum(
    kernel_fn: Callable[[float], np.ndarray],
    y: np.ndarray,
    alphas: Iterable[float],
    Cs: Iterable[float],
    folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[float, float, np.ndarray, list]:
    """
    Grid search over bandwidth α and C.

    kernel_fn(alpha) must return the training kernel for that α. Each α
    needs one kernel computation; every C is then evaluated on the same
    matrix, so the search costs len(alphas) kernels, not len(alphas)·len(Cs).

    Returns best_alpha, best_C, best_K_train, table of (alpha, C, cv_f1).
    """
    best = (-1.0, None, None, None)
    table = []
    for a in alphas:
        K = kernel_fn(a)
        for C in Cs:
            s = cv_score_precomputed(K, y, C, folds, seed)
            table.append({"alpha": float(a), "C": float(C), "cv_macro_f1": s,
                          "offdiag_mean": offdiag_mean(K)})
            if verbose:
                print(f"    α={a:<5g} C={C:<6g} CV macro-F1={s:.4f}")
            if s > best[0]:
                best = (s, a, C, K)
    return best[1], best[2], best[3], table


def tune_rbf(Z: np.ndarray, y: np.ndarray, Cs, gammas, folds: int = 5, seed: int = 42):
    """Grid-searched RBF-SVM on standardised features (same CV protocol)."""
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=seed),
        {"C": list(Cs), "gamma": list(gammas)},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=-1,
    ).fit(Z, y)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


# ── Kernel diagnostics ────────────────────────────────────────────────────────

def offdiag_mean(K: np.ndarray) -> float:
    n = len(K)
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def offdiag_std(K: np.ndarray) -> float:
    return float(K[~np.eye(len(K), dtype=bool)].std())


def centered_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centred kernel–target alignment (Cortes, Mohri & Rostamizadeh 2012)."""
    Y = (y[:, None] == np.unique(y)[None, :]).astype(float)
    T = Y @ Y.T
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Tc = H @ K @ H, H @ T @ H
    return float(np.sum(Kc * Tc) / (np.linalg.norm(Kc) * np.linalg.norm(Tc) + 1e-12))


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray, reg: float = 1e-3) -> float:
    """
    g(K_C ‖ K_Q) = sqrt(‖ √K_Q · (K_C + λI)⁻¹ · √K_Q ‖_∞)   (Huang et al. 2021)

    Both kernels are normalised to trace N. Small g (≈1) means the classical
    kernel can match the quantum one on this data; a quantum advantage
    requires g ≳ √N.
    """
    n = len(K_c)
    K_c = K_c * n / np.trace(K_c)
    K_q = K_q * n / np.trace(K_q)
    sq = np.real(sqrtm(K_q))
    M = sq @ np.linalg.inv(K_c + reg * np.eye(n)) @ sq
    return float(np.sqrt(np.max(eigh(M, eigvals_only=True))))


def rbf_kernel_matrix(Z: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum(Z ** 2, axis=1)
    return np.exp(-gamma * np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))


def kernel_diagnostics(K: np.ndarray, y: np.ndarray,
                       K_classical: Optional[np.ndarray] = None,
                       max_n: int = 400, seed: int = 0) -> Dict:
    d = {
        "offdiag_mean": offdiag_mean(K),
        "offdiag_std": offdiag_std(K),
        "alignment": centered_alignment(K, y),
    }
    if K_classical is not None:
        idx = np.random.default_rng(seed).permutation(len(K))[:max_n]
        sub = np.ix_(idx, idx)
        d["geometric_difference"] = geometric_difference(K_classical[sub], K[sub])
        d["geometric_difference_n"] = int(len(idx))
    return d


def standardize(train: np.ndarray, test: np.ndarray):
    s = StandardScaler().fit(train)
    return s.transform(train), s.transform(test)
