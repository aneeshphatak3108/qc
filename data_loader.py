"""
Direct loader for output.csv (CIC-DDoS2019 format)
==================================================

Changes vs. the original loader (kept in legacy/data_loader.py):

  1. NO global scaling. The original loader fitted a MinMaxScaler on the
     whole dataset *before* the train/test split, so test-set statistics
     leaked into training. This loader returns cleaned, *unscaled* features;
     every scaler / PCA is now fitted on the training split only
     (see preprocessing.py).
  2. Duplicate removal. ~65% of rows in output.csv are exact duplicates
     (same flow features + same task label). With duplicates, a test flow very
     often has an identical twin in the training set, which inflates
     accuracy. dedup=True (default) removes them before sampling.
  3. Reproducible sampling. The original used the unseeded global
     np.random, so every run drew a different subset. All sampling now goes
     through a seeded np.random.default_rng(seed).
  4. Robust numeric parsing. Repeated header rows inside the CSV force
     pandas to read columns as strings; values are now converted with
     pd.to_numeric(errors="coerce") and ±inf / NaN are set to 0.
  5. Feature sets. feature_set="paper" reproduces the 58-feature list used
     in the paper; feature_set="all" uses every numeric CICFlowMeter column
     (adds Subflow, Init_Win_bytes, Active/Idle statistics).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# The 61 CICFlowMeter numeric columns listed in the original loader
# (zero-variance columns are dropped later, leaving 49 informative ones
# on this file).
FEATURE_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
]

# Identifiers / metadata that must never be used as features (they encode
# the testbed's addressing and time, not the behaviour of the flow).
META_COLS = {
    'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
    'Destination Port', 'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound',
}

LABEL_COL = 'Label'


def _read_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df[LABEL_COL].notna()]
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL] != LABEL_COL]          # repeated header rows
    return df.reset_index(drop=True)


def _feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "paper":
        cols = [c for c in FEATURE_COLS if c in df.columns]
    elif feature_set == "all":
        cols = [c for c in df.columns if c not in META_COLS and c != LABEL_COL]
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    return X


def load_csv(
    csv_path: str = "./output.csv",
    mode: str = "binary",
    max_samples: int = 2000,
    balance: bool = True,
    dedup: bool = True,
    feature_set: str = "paper",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], dict]:
    """
    Load output.csv and return *unscaled* features.

    Parameters
    ----------
    csv_path : str
    mode : 'binary' (BENIGN vs DDoS) or 'multiclass' (6 classes)
    max_samples : int
        Total number of samples returned (split equally across classes
        when balance=True).
    balance : bool
        Undersample every class to the same size.
    dedup : bool
        Drop exact duplicate (features, label) rows before sampling.
    feature_set : 'paper' or 'all'
    seed : int
        Seed for all random sampling.
    verbose : bool

    Returns
    -------
    X : np.ndarray, shape (n, d), raw float64 features (NOT scaled)
    y : np.ndarray, shape (n,)
    label_names : dict {int: str}
    info : dict with bookkeeping (row counts, duplicates removed, columns)
    """
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"[Loader] Reading {csv_path}...")

    df = _read_clean(csv_path)
    X_df = _feature_frame(df, feature_set)
    labels = df[LABEL_COL].values
    info = {"rows_raw": int(len(df)), "feature_set": feature_set}

    if verbose:
        print(f"[Loader] Raw rows: {len(df):,} | candidate features: {X_df.shape[1]}")

    # ── Binary / multiclass labels ─────────────────────────────────────────
    if mode == "binary":
        y_all = (pd.Series(labels).str.upper() != "BENIGN").astype(int).values
        label_names = {0: "BENIGN", 1: "DDoS"}
    elif mode == "multiclass":
        cats = pd.Categorical(labels)
        y_all = cats.codes.astype(int)
        label_names = dict(enumerate(cats.categories))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # ── Duplicate removal ──────────────────────────────────────────────────
    if dedup:
        # Key on the label of the *task*: in binary mode an LDAP flow and an
        # identical Portmap flow both become "DDoS", so they are duplicates.
        key = X_df.copy()
        key["__label"] = y_all
        keep = ~key.duplicated().values
        info["duplicates_removed"] = int((~keep).sum())
        X_df, labels, y_all = X_df[keep].reset_index(drop=True), labels[keep], y_all[keep]
        if verbose:
            print(f"[Loader] Removed {info['duplicates_removed']:,} exact duplicates "
                  f"→ {len(X_df):,} unique rows")
    else:
        info["duplicates_removed"] = 0

    if verbose:
        vc = pd.Series(labels).value_counts()
        print("[Loader] Label distribution:\n" + vc.to_string())

    # ── Balanced, seeded sampling ──────────────────────────────────────────
    classes, counts = np.unique(y_all, return_counts=True)
    if balance:
        per_class = min(counts.min(), max_samples // len(classes))
        idx = np.concatenate([
            rng.choice(np.where(y_all == c)[0], size=per_class, replace=False)
            for c in classes
        ])
    else:
        idx = rng.choice(len(y_all), size=min(max_samples, len(y_all)), replace=False)
    rng.shuffle(idx)

    X = X_df.values[idx]
    y = y_all[idx]
    info.update({
        "rows_unique": int(len(X_df)),
        "n_samples": int(len(y)),
        "columns": list(X_df.columns),
    })

    if verbose:
        per = ", ".join(f"{label_names[c]}={int((y == c).sum())}" for c in classes)
        print(f"[Loader] Sampled {len(y):,} rows ({per}) | X={X.shape} (unscaled)")

    return X, y, label_names, info


def duplicate_overlap(X_train: np.ndarray, X_test: np.ndarray) -> float:
    """Fraction of test rows that have an exact copy in the training set."""
    train_rows = {r.tobytes() for r in np.ascontiguousarray(X_train)}
    hits = sum(r.tobytes() in train_rows for r in np.ascontiguousarray(X_test))
    return hits / max(len(X_test), 1)
