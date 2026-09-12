"""
Record your own webcam clips for one word, to replace/augment noisy or
outdated Kaggle training clips (use visualize_samples.py first to spot bad
ones). Saved clips live in data/recorded/<word>/ and are picked up
automatically by prepare_data.py.

Usage:
    python src/record_samples.py <word> [-n NUM_REPS] [--seconds SECONDS]

Controls: SPACE starts a rep (after a 3-2-1 countdown), 'q' quits early.
Click the camera window first so it has keyboard focus.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import extract_frame_from_mediapipe

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "data" / "recorded"
COUNTDOWN_SECONDS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("word")
    parser.add_argument("-n", "--num-reps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=1.6)
    args = parser.parse_args()

    word_dir = OUT_DIR / args.word
    word_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(list(word_dir.glob("*.npy")))

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    rep = 0
    state = "ready"  # ready -> countdown -> recording
    countdown_start = 0.0
    record_start = 0.0
    buffer = []

    print(f"\nRecording '{args.word}': {args.num_reps} reps, {args.seconds}s each.")
    print("Click the camera window, then press SPACE to record each rep. 'q' to quit early.\n")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and rep < args.num_reps:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            now = time.time()

            if state == "ready":
                cv2.putText(
                    frame, f"'{args.word}'  rep {rep + 1}/{args.num_reps}  -  SPACE to record, q to quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            elif state == "countdown":
                remaining = COUNTDOWN_SECONDS - int(now - countdown_start)
                if remaining <= 0:
                    state = "recording"
                    record_start = now
                    buffer = []
                else:
                    cv2.putText(
                        frame, str(remaining), (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4,
                    )
            elif state == "recording":
                buffer.append(extract_frame_from_mediapipe(results))
                elapsed = now - record_start
                cv2.putText(
                    frame, f"RECORDING {elapsed:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
                if elapsed >= args.seconds:
                    seq = np.stack(buffer, axis=0)
                    out_path = word_dir / f"{args.word}_{start_index + rep + 1:03d}.npy"
                    np.save(out_path, seq)
                    print(f"saved {out_path} ({len(seq)} frames)")
                    rep += 1
                    state = "ready"

            cv2.imshow("Record ASL sample", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and state == "ready":
                state = "countdown"
                countdown_start = now

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {rep} new clip(s) saved to {word_dir}")


if __name__ == "__main__":
    main()
