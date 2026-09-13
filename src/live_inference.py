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
from prepare_data import pad_or_sample
from translate import gloss_to_english

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_model.keras")
STATS_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocess_stats.json")
CONFIDENCE_THRESHOLD = 0.4
MIN_SEQ_LEN = 24  # start attempting predictions once we have at least this many frames
PREDICT_STRIDE = 4  # after MIN_SEQ_LEN, retry every N new frames until confident or full
CONFIRM_COUNT = 2  # require this many consecutive matching predictions before committing


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

    buffer = []
    last_prediction_text = "..."
    frames_since_predict = 0
    recognized_feed = deque(maxlen=8)
    pending_word = None
    pending_count = 0

    print("\nSign one of your chosen words at the camera. Press 'q' to quit, 'c' to clear the feed.\n")

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

            cv2.putText(
                frame, "Sign-to-Text  |  by Connor G", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            fill = len(buffer)
            cv2.rectangle(frame, (0, 0), (int(frame.shape[1] * min(fill, MAX_SEQ_LEN) / MAX_SEQ_LEN), 8), (0, 200, 0), -1)

            frames_since_predict += 1
            ready_to_try = fill >= MIN_SEQ_LEN and frames_since_predict >= PREDICT_STRIDE
            buffer_full = fill >= MAX_SEQ_LEN

            if ready_to_try or buffer_full:
                frames_since_predict = 0
                seq = np.stack(buffer, axis=0)
                seq = normalize_sequence(seq, feat_mean, feat_min, feat_max)
                seq = pad_or_sample(seq, MAX_SEQ_LEN)
                seq = seq[np.newaxis, ...]  # (1, MAX_SEQ_LEN, FEATURE_DIM)

                probs = model.predict(seq, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    word = words[pred_idx]
                    if word == pending_word:
                        pending_count += 1
                    else:
                        pending_word = word
                        pending_count = 1

                    if pending_count >= CONFIRM_COUNT:
                        last_prediction_text = f"{word} ({confidence:.0%})"
                        print(f">>> {word}  (confidence {confidence:.0%})")
                        if not recognized_feed or recognized_feed[-1] != word:
                            recognized_feed.append(word)
                        pending_word = None
                        pending_count = 0
                    else:
                        last_prediction_text = f"{word}...? ({confidence:.0%})"
                        print(f">>> tentative {word}  (confidence {confidence:.0%}), confirming...")
                    buffer = []
                elif buffer_full:
                    last_prediction_text = f"? ({confidence:.0%})"
                    print(f">>> not confident enough (best guess {words[pred_idx]} at {confidence:.0%})")
                    pending_word = None
                    pending_count = 0
                    buffer = []
                # else: not confident yet and buffer isn't full - keep accumulating and retry

            cv2.putText(
                frame, last_prediction_text, (20, frame.shape[0] - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
            )

            feed_text = "Feed: " + " ".join(recognized_feed) if recognized_feed else "Feed: (empty - 'c' to clear)"
            cv2.putText(
                frame, feed_text, (20, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

            translated_text = "English: " + gloss_to_english(recognized_feed)
            cv2.putText(
                frame, translated_text, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2,
            )

            cv2.imshow("ASL Recognition Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                recognized_feed.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
