"""
Direct loader for output.csv (CIC-DDoS2019 format)
Replaces cic_loader.py dependency.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# The 58 CICFlowMeter numeric features used for quantum encoding
FEATURE_COLS = [
    ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
    ' Total Length of Fwd Packets', ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
    ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
    ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
    ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd PSH Flags',
    ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
    ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
    ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
    ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
    ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
    ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
]

LABEL_COL = ' Label'


def load_csv(
    csv_path: str = "./output.csv",
    mode: str = "binary",
    max_samples: int = 2000,
    balance: bool = True,
    verbose: bool = True,
):
    """
    Load output.csv and return angle-encoded X, y, label_names.

    Parameters
    ----------
    csv_path : str
        Path to output.csv
    mode : 'binary' or 'multiclass'
    max_samples : int
        Max rows to load before balancing
    balance : bool
        Undersample to equal class sizes
    verbose : bool

    Returns
    -------
    X : np.ndarray, shape (n, 58), values in [0, π]
    y : np.ndarray, shape (n,)
    label_names : dict {int: str}
    """
    if verbose:
        print(f"[Loader] Reading {csv_path}...")

    df = pd.read_csv(csv_path, low_memory=False)

    if verbose:
        print(f"[Loader] Raw shape: {df.shape}")

    # ── Clean label column ─────────────────────────────────────────────────
    df.columns = df.columns.str.strip()
    label_col = 'Label'
    df = df[df[label_col].notna()]
    df = df[df[label_col] != 'Label']  # remove header rows
    df[label_col] = df[label_col].str.strip()

    if verbose:
        print(f"[Loader] Label distribution:\n{df[label_col].value_counts().to_string()}")

    # ── Sample before expensive ops ────────────────────────────────────────
    if len(df) > max_samples * 3:
        df = df.sample(n=min(max_samples * 3, len(df)), random_state=42)

    # ── Binary / multiclass labels ─────────────────────────────────────────
    if mode == "binary":
        df['_y'] = (df[label_col].str.upper() != 'BENIGN').astype(int)
        label_names = {0: 'BENIGN', 1: 'DDoS'}
    else:
        cats = df[label_col].astype('category')
        df['_y'] = cats.cat.codes
        label_names = dict(enumerate(cats.cat.categories))

    # ── Select feature columns ─────────────────────────────────────────────
    # Try exact names first, then strip-matched
    available = list(df.columns)
    feat_cols = []
    for fc in FEATURE_COLS:
        if fc in available:
            feat_cols.append(fc)
        elif fc.strip() in [c.strip() for c in available]:
            matched = next(c for c in available if c.strip() == fc.strip())
            feat_cols.append(matched)

    if len(feat_cols) < 10:
        # Fallback: just grab all numeric columns except metadata
        skip = {'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
                'Destination IP', 'Destination Port', 'Protocol',
                'Timestamp', label_col, '_y', 'SimillarHTTP', 'Inbound'}
        feat_cols = [c for c in df.columns
                     if c not in skip and df[c].dtype != object]

    if verbose:
        print(f"[Loader] Using {len(feat_cols)} feature columns")

    # ── Extract features ───────────────────────────────────────────────────
    X_raw = df[feat_cols].copy()
    y_raw = df['_y'].values

    # Replace inf/nan
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.fillna(0)
    X_raw = X_raw.astype(np.float64)

    # Remove zero-variance columns
    var = X_raw.std(axis=0)
    X_raw = X_raw.loc[:, var > 0]

    # Pad or trim to exactly 58 features
    n_feat = X_raw.shape[1]
    if n_feat < 58:
        pad = np.zeros((len(X_raw), 58 - n_feat))
        X_np = np.hstack([X_raw.values, pad])
    else:
        X_np = X_raw.values[:, :58]

    if verbose:
        print(f"[Loader] Feature matrix: {X_np.shape}")

    # ── Balance classes ────────────────────────────────────────────────────
    if balance:
        classes, counts = np.unique(y_raw, return_counts=True)
        min_count = min(counts.min(), max_samples // len(classes))
        idx_all = []
        for c in classes:
            idx_c = np.where(y_raw == c)[0]
            idx_all.append(np.random.choice(idx_c, size=min_count, replace=False))
        idx_all = np.concatenate(idx_all)
        np.random.shuffle(idx_all)
        X_np = X_np[idx_all]
        y_raw = y_raw[idx_all]
        if verbose:
            print(f"[Loader] Balanced: {min_count} samples/class → {len(y_raw)} total")

    # ── Angle encode to [0, π] ─────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_enc = scaler.fit_transform(X_np)

    if verbose:
        print(f"[Loader] Angle range: [{X_enc.min():.4f}, {X_enc.max():.4f}]")
        print(f"[Loader] ✓ Final: X={X_enc.shape}, classes={label_names}")

    return X_enc, y_raw.astype(int), label_names
