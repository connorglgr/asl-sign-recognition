"""
Build a fixed-size (N, MAX_SEQ_LEN, FEATURE_DIM) training array from the
filtered GISLR parquet subset, plus normalization stats and the label map,
saved into data/processed/ for train.py to consume.
"""

import json
import os

import numpy as np
import pandas as pd

from landmarks import FEATURE_DIM, MAX_SEQ_LEN, extract_sequence_from_parquet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBSET_CSV = os.path.join(DATA_DIR, "train_subset.csv")
OUT_DIR = os.path.join(DATA_DIR, "processed")
VAL_FRACTION = 0.15
SEED = 42


def pad_or_sample(seq, max_len):
    """Uniformly subsample if too long, zero-pad at the end if too short."""
    n = seq.shape[0]
    if n == max_len:
        return seq
    if n > max_len:
        idxs = np.linspace(0, n - 1, max_len).round().astype(int)
        return seq[idxs]
    pad = np.zeros((max_len - n, seq.shape[1]), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sub = pd.read_csv(SUBSET_CSV)

    words = sorted(sub["sign"].unique().tolist())
    word_to_idx = {w: i for i, w in enumerate(words)}
    print(f"Vocabulary ({len(words)}): {words}")

    raw_sequences = []
    labels = []
    skipped = 0
    for i, row in sub.iterrows():
        pq_path = os.path.join(DATA_DIR, row["path"])
        if not os.path.exists(pq_path):
            skipped += 1
            continue
        try:
            seq = extract_sequence_from_parquet(pq_path)
        except Exception:
            skipped += 1
            continue
        raw_sequences.append(seq)
        labels.append(word_to_idx[row["sign"]])
        if (i + 1) % 500 == 0:
            print(f"  extracted {i + 1}/{len(sub)}", flush=True)

    print(f"Extracted {len(raw_sequences)} sequences, skipped {skipped} missing/bad files")

    # Per-feature mean (for filling gaps) and min/max (for normalization),
    # computed once over all valid, non-NaN values across every frame.
    all_frames = np.concatenate(raw_sequences, axis=0)  # (total_frames, FEATURE_DIM)
    feat_mean = np.nanmean(all_frames, axis=0)
    feat_mean = np.nan_to_num(feat_mean, nan=0.0)
    feat_min = np.nanmin(all_frames, axis=0)
    feat_max = np.nanmax(all_frames, axis=0)
    feat_range = np.where(feat_max - feat_min < 1e-6, 1.0, feat_max - feat_min)

    X = np.zeros((len(raw_sequences), MAX_SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    for i, seq in enumerate(raw_sequences):
        seq = np.where(np.isnan(seq), feat_mean, seq)
        seq = (seq - feat_min) / feat_range
        seq = np.clip(seq, 0.0, 1.0)
        X[i] = pad_or_sample(seq, MAX_SEQ_LEN)

    y = np.array(labels, dtype=np.int64)

    # Stratified train/val split (per-sign, so every class is represented in both).
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for cls in range(len(words)):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * VAL_FRACTION)))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)

    stats = {
        "words": words,
        "feature_dim": FEATURE_DIM,
        "max_seq_len": MAX_SEQ_LEN,
        "feat_mean": feat_mean.tolist(),
        "feat_min": feat_min.tolist(),
        "feat_max": feat_max.tolist(),
    }
    with open(os.path.join(OUT_DIR, "preprocess_stats.json"), "w") as f:
        json.dump(stats, f)

    print(f"Saved processed data + stats to {OUT_DIR}")


if __name__ == "__main__":
    main()
