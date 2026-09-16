# ASL Sign Recognition

A real-time American Sign Language recognizer: webcam → hand/pose/face
landmarks (MediaPipe) → a small GRU sequence model → predicted word, shown
live on screen with a running "feed" of recognized signs and a
rule-based attempt at English translation.

**Status: work in progress.** Started as a software proof-of-concept for a
future physical ASL interpretation device, built for my deaf mom. Currently
recognizes an 11-word vocabulary and is being expanded toward simple
sentences.

> See [DEVLOG.md](DEVLOG.md) (or [DEVLOG.pdf](DEVLOG.pdf)) for the full build
> story, including the debugging process, dead ends, and what actually fixed
> each problem — not just the final result.

## What it does

- Captures webcam video and extracts hand, arm, and face landmarks with
  [MediaPipe Holistic](https://developers.google.com/mediapipe)
- Feeds a rolling window of landmarks into a GRU-based classifier trained on
  a mix of the [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
  Kaggle dataset and self-recorded clips
- Displays the predicted word live, along with a short history of recently
  recognized words and a plain-English translation attempt

Current vocabulary (11 words): `dad`, `happy`, `hello`, `later`, `minemy`
(mine/my), `mom`, `no`, `please`, `sad`, `thankyou`, `yes`.

## Setup

Requires Python 3.9 and a webcam. Tested on Apple Silicon macOS.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run the live recognizer
python src/live_inference.py
```

Controls: press **'q'** to quit, **'c'** to clear the recognized-words feed.

### Training pipeline (optional — a trained model is already included)

```bash
python src/prepare_data.py   # build the training arrays from Kaggle + recorded data
python src/train.py          # train the GRU model
```

### Recording your own training clips

If a word isn't being recognized well, you can record replacement training
clips with your own webcam instead of relying on the Kaggle data:

```bash
python src/record_samples.py <word> -n 20     # record 20 reps of a word
python src/visualize_samples.py <word>        # sanity-check clips as GIFs
```

Any word with files under `data/recorded/<word>/` will use those clips
instead of the Kaggle-derived ones the next time `prepare_data.py` runs.

## How it works

- `src/landmarks.py` — shared landmark selection (a reduced 122-dim feature
  vector: both hands in full, 6 pose points, 13 face points, x/y only)
- `src/prepare_data.py` — builds the training arrays, with light data
  augmentation and class-weighted training to handle small/imbalanced classes
- `src/train.py` — GRU(64) → GRU(32) → Dense classifier
- `src/live_inference.py` — the real-time webcam demo
- `src/translate.py` — hardcoded ASL-gloss → English rules for the current
  vocabulary (not a general solution — see the dev log for why)

## Known limitations

- Small vocabulary (11 words) and a personal-project-scale dataset
- Idle/resting hands can occasionally get misread as a low-confidence word
  rather than correctly reading as "nothing happening" (open issue)
- English translation only covers a few hardcoded sentence patterns

## Possible next steps

- Expand vocabulary toward full sentences
- A browser-based version (TensorFlow.js) so it's tryable without a local
  Python setup
