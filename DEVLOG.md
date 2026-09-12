# ASL Sign Recognition — Dev Log

A running log of the build, the problems hit along the way, and how they got fixed.
Started as a software proof-of-concept for a physical ASL interpretation device,
built for my deaf mom.

## The idea

Webcam → hand/pose/face landmarks (MediaPipe) → sequence model (GRU) → predicted
word, printed live. Starting vocabulary: 10 words — dad, happy, hello, later, mom,
no, please, sad, thankyou, yes.

## Setting up the environment (more painful than expected)

On an Apple Silicon Mac, `mediapipe`, `tensorflow`, and `opencv` turned out to have a
narrow window of versions that actually work together:

- `mediapipe >= 1.0` dropped the legacy `solutions.holistic` API entirely (Tasks API
  only now), so had to pin `mediapipe==0.10.14`.
- `tensorflow 2.20` wants `protobuf >= 5.28`, which conflicts with mediapipe's
  `protobuf < 5` requirement.
- `opencv-python 5.0` wants `numpy >= 2`, which conflicts with tensorflow 2.15's
  `numpy < 2` pin.
- Even with the right `tensorflow==2.15.0` pin, a plain `pip install` left
  `tensorflow/__init__.py` missing on arm64 — had to force-reinstall
  `tensorflow-macos==2.15.0` alongside it.

Working combination pinned in `requirements.txt`: `mediapipe==0.10.14`,
`tensorflow==2.15.0`, `opencv-python==4.10.0.84`, `protobuf==4.25.9`.

## Getting training data (Kaggle)

Used Google's "Isolated Sign Language Recognition" Kaggle dataset (landmark-only,
no raw video — collected via the PopSign app, from parents of deaf children
learning ASL at home, not necessarily fluent/certified signers — this detail
mattered a lot later).

Kaggle auth had its own trap: the new-style `KGAT_...` API tokens generated from
the main "API Tokens" page aren't supported by the `kaggle` or `kagglehub` pip
packages — both only read the classic `kaggle.json` format, generated from a
separate "Legacy API Key" section. Also needed phone verification and to formally
join the competition before downloads worked (silent 403s otherwise).

Then hit Kaggle's per-account rate limit downloading with 12 concurrent workers —
it was still in effect hours later even at low concurrency. Rather than wait it
out, training proceeded on the 446 files (of 3807 available for these 10 words)
that had already downloaded.

## Feature engineering

MediaPipe Holistic emits 543 landmarks/frame (468 face + 33 pose + 21+21 hands).
For a ~450-sample dataset, that's way more parameters than the data could support,
so the feature set was deliberately reduced:

- Dropped z entirely (x, y only)
- Kept both hands in full (21 pts each) — the main signal for handshape
- Only 6 pose points (shoulders/elbows/wrists, not all 33)
- Only 13 face points (nose/mouth/cheek area, not all 468)
- Final feature vector: 122-dim (61 landmarks × 2 coords)

Sequences padded/subsampled to a fixed 48 frames (~p90 of the observed sequence
lengths; median was 18, p95 was 53).

## First training run

GRU(64) → GRU(32) → Dense classifier. 446 training sequences, 55% validation
accuracy (vs. 10% random chance for 10 classes). Live testing showed it was
strongest on stationary/distinct handshapes (dad, mom, hello, later, please) and
weak on motion-heavy signs (no, yes, sad, thankyou) — the obvious read was "not
enough data," but that turned out to be only half the story.

## The real bug: some Kaggle labels don't match modern ASL

Built `src/visualize_samples.py` — renders a training clip's landmark sequence as
an animated 2D skeleton GIF (hand connections + arm/shoulder points), so a human
who actually knows ASL can sanity-check whether a label matches the sign.

That's how this got caught: the Kaggle "later" clips were using an outdated sign
variant, not the modern one (right hand forms an L held upward, then rotates 90°
down, pointer toward camera). And "no" — despite having a totally normal sample
count (~36-44 clips, same ballpark as working classes) — never got predicted
*at all* across multiple live test sessions. That ruled out "not enough data" as
the explanation for "no" specifically; the label itself was the problem, most
likely because the dataset's non-fluent-signer participants didn't consistently
produce a clean, correct version of that sign.

## The fix: record your own replacement clips

Built `src/record_samples.py` — opens the webcam, same MediaPipe setup as live
inference, lets you record N reps of yourself signing a word correctly (SPACE to
record after a countdown), and saves them to `data/recorded/<word>/`.

Extended `src/prepare_data.py` so that any word with files under
`data/recorded/<word>/` has **all** of its Kaggle rows dropped and replaced
entirely by the self-recorded clips — not merged in, since the goal was fixing a
wrong label, not diluting bad data with a little good data.

### Lessons learned re-doing "later," "no," and "yes"

- **7 recorded clips wasn't enough** — even with the correct sign now, "later"
  still never got predicted live until bumped up to 36 clips (then hit
  46-99% confidence consistently). Rule of thumb: match the sample count of the
  strongest existing classes (~35-45), not just "some."
- **Relative class size matters, not just absolute** — after fixing "yes" with
  28-29 clips, it kept losing to "later" (36 clips) in live testing, even though
  "yes" itself was no longer mislabeled. "Later" being more confidently learned
  let it dominate ambiguous predictions. Bumping "yes" to 41 clips (more than
  "later") fixed it.
- **One contaminated clip matters** — deleting a single "yes" recording that
  accidentally had my other arm in frame gave a measurable accuracy bump on its
  own.
- Final result: validation accuracy went from 55% → ~67%, despite the total
  training set actually *shrinking* (446 Kaggle clips → ~187 Kaggle + ~106
  self-recorded) once three classes were fully swapped over. Fixing bad labels
  beat adding more of the same bad data.

## Live inference UX: the "stickiness" bug

Original design: buffer 48 frames, predict once, clear the buffer, wait for a
completely fresh 48 frames before predicting again. Worked, but felt slow
(~1.5-2s dead air between predictions).

First attempt at speeding it up: never clear the buffer, just slide it forward
and predict every 8 new frames. Cycling felt faster, but introduced a much worse
bug — after finishing a sign, the old frames lingered in the window for several
more prediction cycles and kept "overruling" whatever was signed next (e.g. "no"
would keep getting predicted for a few cycles after switching to "yes" or "dad").
This wasn't just visual sign similarity — it was literally leftover frames from
the completed gesture still dominating the window.

Fix: buffer is a plain list, not a sliding deque. Once at least 24 frames are
collected, it attempts a prediction every 4 new frames (zero-padded up to the
model's expected 48-frame input, same as how short Kaggle clips get padded at
training time). Critically, **the buffer is cleared after every prediction
attempt, confident or not** — so a finished sign's frames never leak into the
next one. This keeps the faster cadence without the cross-contamination.

## Current state

10-word vocabulary, ~67% validation accuracy. "Later," "no," and "yes" fixed via
self-recorded replacement data. "Sad" and "thankyou" were spot-checked visually
and looked correct as-is. "Happy" hasn't been investigated yet.

## Possible next steps

- Check "happy" the same way (visualize → decide if it needs re-recording)
- Backfill more of the original 3807 available Kaggle files (only 446 were
  downloaded before hitting the rate limit) for the classes still using Kaggle
  data
- Explore a browser-based version (TensorFlow.js + MediaPipe Tasks Vision JS)
  so it's accessible as a website instead of a local Python script — the model
  itself is small enough that this would run fully client-side, no server
  needed. Main work would be re-implementing the landmark-extraction feature
  layout against the browser MediaPipe API, which is structured differently
  from the Python `solutions.holistic` API used here.
