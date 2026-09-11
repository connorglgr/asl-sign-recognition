"""
Live webcam ASL recognition demo.

Captures video, runs MediaPipe Holistic per frame, buffers MAX_SEQ_LEN
frames (~1.5-2s at typical webcam fps), feeds the sequence into the
trained model, and prints the predicted sign to the terminal.
"""

import json
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from landmarks import FEATURE_DIM, MAX_SEQ_LEN, extract_frame_from_mediapipe

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_model.keras")
STATS_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocess_stats.json")
CONFIDENCE_THRESHOLD = 0.4


def load_stats():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return (
        stats["words"],
        np.array(stats["feat_mean"], dtype=np.float32),
        np.array(stats["feat_min"], dtype=np.float32),
        np.array(stats["feat_max"], dtype=np.float32),
    )


def normalize_sequence(seq, feat_mean, feat_min, feat_max):
    feat_range = np.where(feat_max - feat_min < 1e-6, 1.0, feat_max - feat_min)
    seq = np.where(np.isnan(seq), feat_mean, seq)
    seq = (seq - feat_min) / feat_range
    return np.clip(seq, 0.0, 1.0).astype(np.float32)


def main():
    words, feat_mean, feat_min, feat_max = load_stats()
    print(f"Loaded {len(words)} classes: {words}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    buffer = deque(maxlen=MAX_SEQ_LEN)
    last_prediction_text = "..."

    print("\nSign one of your chosen words at the camera. Press 'q' to quit.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            feat = extract_frame_from_mediapipe(results)
            buffer.append(feat)

            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            fill = len(buffer)
            cv2.rectangle(frame, (0, 0), (int(frame.shape[1] * fill / MAX_SEQ_LEN), 8), (0, 200, 0), -1)

            if fill == MAX_SEQ_LEN:
                seq = np.stack(buffer, axis=0)
                seq = normalize_sequence(seq, feat_mean, feat_min, feat_max)
                seq = seq[np.newaxis, ...]  # (1, MAX_SEQ_LEN, FEATURE_DIM)

                probs = model.predict(seq, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    word = words[pred_idx]
                    last_prediction_text = f"{word} ({confidence:.0%})"
                    print(f">>> {word}  (confidence {confidence:.0%})")
                else:
                    last_prediction_text = f"? ({confidence:.0%})"
                    print(f">>> not confident enough (best guess {words[pred_idx]} at {confidence:.0%})")

                buffer.clear()

            cv2.putText(
                frame, last_prediction_text, (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
            )
            cv2.imshow("ASL Recognition Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
