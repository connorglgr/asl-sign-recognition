"""
Shared landmark selection/ordering used by both training data prep and
live webcam inference, so the two stay in sync.

MediaPipe Holistic emits 543 landmarks per frame in the GISLR parquet
layout: 468 face + 33 pose + 21 left_hand + 21 right_hand.
We reduce that down to a compact, demo-friendly feature set:
  - both hands in full (21 pts each) - the main signal for handshape
  - 6 pose points (shoulders, elbows, wrists) - arm position context
  - 13 face points (nose/mouth/cheek area) - light expression cue
x, y only (z dropped) to keep the feature vector small for a tiny dataset.
"""

import numpy as np
import pandas as pd

ROWS_PER_FRAME = 543

HAND_INDICES = list(range(21))
POSE_INDICES = [11, 12, 13, 14, 15, 16]  # left/right shoulder, elbow, wrist
FACE_INDICES = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348]

# Fixed order the feature vector is assembled in. Every stage of the
# pipeline (data prep, training, live inference) must follow this order.
PARTS = [
    ("left_hand", HAND_INDICES),
    ("right_hand", HAND_INDICES),
    ("pose", POSE_INDICES),
    ("face", FACE_INDICES),
]

NUM_LANDMARKS = sum(len(idxs) for _, idxs in PARTS)  # 21+21+6+13 = 61
NUM_COORDS = 2  # x, y only
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 122

MAX_SEQ_LEN = 48  # covers ~p90 of sequence lengths in our subset


def extract_sequence_from_parquet(pq_path):
    """
    Read a GISLR landmark parquet file and return a (n_frames, FEATURE_DIM)
    float32 array following the fixed PARTS order. Missing landmarks
    (e.g. hand not visible) are NaN.
    """
    df = pd.read_parquet(pq_path, columns=["frame", "type", "landmark_index", "x", "y"])
    frames = np.sort(df["frame"].unique())
    n_frames = len(frames)
    frame_pos = pd.Series(np.arange(n_frames), index=frames)

    out = np.full((n_frames, FEATURE_DIM), np.nan, dtype=np.float32)
    col = 0
    for part_name, idxs in PARTS:
        part_df = df[df["type"] == part_name]
        x_wide = part_df.pivot_table(index="frame", columns="landmark_index", values="x")
        y_wide = part_df.pivot_table(index="frame", columns="landmark_index", values="y")
        x_wide = x_wide.reindex(index=frames, columns=idxs)
        y_wide = y_wide.reindex(index=frames, columns=idxs)

        for i in range(len(idxs)):
            out[:, col] = x_wide.iloc[:, i].to_numpy()
            out[:, col + 1] = y_wide.iloc[:, i].to_numpy()
            col += 2

    return out


def extract_frame_from_mediapipe(results):
    """
    Given a MediaPipe Holistic `results` object for a single live frame,
    return a (FEATURE_DIM,) float32 array in the same PARTS order used
    for training data. Missing landmarks (hand out of frame, etc.) are NaN.
    """
    out = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    col = 0

    def fill(landmark_list, idxs):
        nonlocal col
        if landmark_list is not None:
            pts = landmark_list.landmark
            for idx in idxs:
                out[col] = pts[idx].x
                out[col + 1] = pts[idx].y
                col += 2
        else:
            col += len(idxs) * 2

    fill(results.left_hand_landmarks, HAND_INDICES)
    fill(results.right_hand_landmarks, HAND_INDICES)
    fill(results.pose_landmarks, POSE_INDICES)
    fill(results.face_landmarks, FACE_INDICES)
    return out
