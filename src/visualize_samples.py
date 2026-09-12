"""
Standalone sanity-check tool: renders a few training landmark sequences as
animated GIFs so you can visually confirm the recorded motion actually
matches the labeled sign (compare against a real ASL dictionary video).

Usage:
    python src/visualize_samples.py [word ...] [-n NUM_SAMPLES]

Defaults to the words that perform poorly in live_inference.py.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import extract_sequence_from_parquet

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output" / "viz"

# Column offsets into the FEATURE_DIM=122 vector, matching the PARTS order
# in landmarks.py: left_hand(21) -> right_hand(21) -> pose(6) -> face(13).
LEFT_HAND_OFF, RIGHT_HAND_OFF, POSE_OFF, FACE_OFF = 0, 42, 84, 96

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
# pose order here is [L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist]
POSE_CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]


def part_xy(row, offset, n_points):
    pts = row[offset: offset + n_points * 2].reshape(n_points, 2)
    return pts[:, 0], pts[:, 1]


def connection_lines(row, offset, n_points, connections):
    x, y = part_xy(row, offset, n_points)
    xs, ys = [], []
    for a, b in connections:
        xs += [x[a], x[b], np.nan]
        ys += [y[a], y[b], np.nan]
    return xs, ys


def animate_sequence(seq, title, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # invert y: landmark coords are image-space (0,0)=top-left
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    left_hand, = ax.plot([], [], "o-", color="tab:blue", markersize=3, linewidth=1, label="left hand")
    right_hand, = ax.plot([], [], "o-", color="tab:red", markersize=3, linewidth=1, label="right hand")
    pose, = ax.plot([], [], "o-", color="gray", markersize=4, linewidth=1)
    face, = ax.plot([], [], ".", color="lightgray", markersize=2)
    frame_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    def update(i):
        row = seq[i]
        left_hand.set_data(*connection_lines(row, LEFT_HAND_OFF, 21, HAND_CONNECTIONS))
        right_hand.set_data(*connection_lines(row, RIGHT_HAND_OFF, 21, HAND_CONNECTIONS))
        pose.set_data(*connection_lines(row, POSE_OFF, 6, POSE_CONNECTIONS))
        face.set_data(*part_xy(row, FACE_OFF, 13))
        frame_text.set_text(f"frame {i + 1}/{len(seq)}")
        return left_hand, right_hand, pose, face, frame_text

    ani = animation.FuncAnimation(fig, update, frames=len(seq), interval=120, blit=False)
    ani.save(out_path, writer="pillow", fps=8)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "words", nargs="*", default=["no", "later", "thankyou", "yes", "sad"],
        help="signs to render (default: the ones that underperform live)",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=2, help="clips per word")
    parser.add_argument(
        "--source", choices=["kaggle", "recorded"], default="kaggle",
        help="kaggle = Kaggle GISLR training clips, recorded = your own clips from record_samples.py",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "recorded":
        for word in args.words:
            word_dir = DATA_DIR / "recorded" / word
            clips = sorted(word_dir.glob("*.npy")) if word_dir.is_dir() else []
            for clip_path in clips[: args.num_samples]:
                seq = np.load(clip_path)
                seq = np.nan_to_num(seq, nan=0.0)
                out_path = OUT_DIR / f"{clip_path.stem}.gif"
                animate_sequence(seq, f"{clip_path.stem}  ({len(seq)} frames)", out_path)
                print(f"saved {out_path}")
            if not clips:
                print(f"no recorded clips found for '{word}' (data/recorded/{word}/)")
        return

    df = pd.read_csv(DATA_DIR / "train_subset.csv")
    for word in args.words:
        candidates = df[df["sign"] == word]
        found = 0
        for _, row in candidates.iterrows():
            pq_path = DATA_DIR / row["path"]
            if not pq_path.exists():
                continue
            seq = extract_sequence_from_parquet(pq_path)
            seq = np.nan_to_num(seq, nan=0.0)
            out_path = OUT_DIR / f"{word}_{row['sequence_id']}.gif"
            animate_sequence(seq, f"{word}  (seq {row['sequence_id']}, {len(seq)} frames)", out_path)
            print(f"saved {out_path}")
            found += 1
            if found >= args.num_samples:
                break
        if found == 0:
            print(f"no local parquet files found for '{word}' (not downloaded)")


if __name__ == "__main__":
    main()
