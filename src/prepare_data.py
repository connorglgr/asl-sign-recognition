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
RECORDED_DIR = os.path.join(DATA_DIR, "recorded")
OUT_DIR = os.path.join(DATA_DIR, "processed")
VAL_FRACTION = 0.15
SEED = 42
AUGMENTED_COPIES_PER_RECORDED_CLIP = 2  # extra jittered copies per self-recorded clip


def augment_sequence(seq, rng):
    """
    Apply one random small affine jitter (rotate/scale/shift) to a raw
    (n_frames, FEATURE_DIM) landmark sequence, same transform across every
    frame so the motion itself stays coherent. Self-recorded clips all come
    from one person/camera setup, so without this the model can overfit to
    that exact body size/position and generalize poorly to a different
    signer (e.g. struggled when the recording user's mom - the actual
    intended end user - tried it).
    """
    pts = seq.reshape(seq.shape[0], -1, 2).copy()  # (frames, landmarks, 2)

    scale = rng.uniform(0.9, 1.1)
    angle = np.radians(rng.uniform(-8, 8))
    shift = rng.uniform(-0.03, 0.03, size=2).astype(np.float32)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    center = np.nanmean(pts.reshape(-1, 2), axis=0)
    pts = (pts - center) @ rot.T * scale + center
    pts += shift

    return pts.reshape(seq.shape[0], -1).astype(np.float32)


def load_recorded():
    """
    Load self-recorded clips from data/recorded/<word>/*.npy (raw (n_frames,
    FEATURE_DIM) arrays, NaN for missing landmarks, saved by
    record_samples.py). Words with recorded clips fully replace the
    corresponding Kaggle rows, since the point is usually to fix a
    mislabeled/outdated Kaggle sign rather than dilute it. Each recorded
    clip is expanded with a few randomly-jittered copies (see
    augment_sequence) to reduce overfitting to one signer's exact body
    size/position.
    """
    sequences, labels = [], []
    if not os.path.isdir(RECORDED_DIR):
        return sequences, labels, set()
    rng = np.random.default_rng(SEED)
    words_with_recordings = set()
    for word in sorted(os.listdir(RECORDED_DIR)):
        word_dir = os.path.join(RECORDED_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        clips = sorted(f for f in os.listdir(word_dir) if f.endswith(".npy"))
        if not clips:
            continue
        for clip in clips:
            seq = np.load(os.path.join(word_dir, clip)).astype(np.float32)
            sequences.append(seq)
            labels.append(word)
            for _ in range(AUGMENTED_COPIES_PER_RECORDED_CLIP):
                sequences.append(augment_sequence(seq, rng))
                labels.append(word)
        words_with_recordings.add(word)
        total = len(clips) * (1 + AUGMENTED_COPIES_PER_RECORDED_CLIP)
        print(f"  loaded {len(clips)} recorded clips for '{word}' (+{total - len(clips)} augmented = {total})")
    return sequences, labels, words_with_recordings


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

    print("Checking for self-recorded replacement clips...")
    recorded_sequences, recorded_labels, replaced_words = load_recorded()
    if replaced_words:
        print(f"Replacing Kaggle clips with recordings for: {sorted(replaced_words)}")

    words = sorted(set(sub["sign"].unique().tolist()) | replaced_words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    print(f"Vocabulary ({len(words)}): {words}")

    raw_sequences = []
    labels = []
    skipped = 0
    for i, row in sub.iterrows():
        if row["sign"] in replaced_words:
            continue
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

    print(f"Extracted {len(raw_sequences)} Kaggle sequences, skipped {skipped} missing/bad files")

    for seq, word in zip(recorded_sequences, recorded_labels):
        raw_sequences.append(seq)
        labels.append(word_to_idx[word])
    if recorded_sequences:
        print(f"Added {len(recorded_sequences)} self-recorded sequences")

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
