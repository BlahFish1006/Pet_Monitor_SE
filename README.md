# Pet Monitor SE — Abnormal Barking Detector (MVP)

Scenario 1 (continuous barking) of the Pet Edge Tracking System.

## What it does

1. Captures audio from your microphone.
2. Computes a **rolling baseline** of background noise (median dBFS over the last 30 s).
3. Flags each 100 ms frame as *loud* if it exceeds `baseline + 15 dB`.
4. Slides a **10 s window** over those flags. When ≥80 % of frames in the window are loud, fires an alert (terminal log + Windows system notification).
5. 30 s cooldown so one barking episode only alerts once.

The 80 % rule handles real bark cadence (汪—汪—汪 with gaps) — a pure "continuous" rule would miss most real barking.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# List your audio input devices (find your mic's index):
python bark_detector.py --list-devices

# Run on default mic with live level meter:
python bark_detector.py -v

# Run on a specific device:
python bark_detector.py --device 3 -v
```

Press `Ctrl+C` to stop.

## Tuning knobs

In [bark_detector.py](bark_detector.py):

| Constant | Default | Meaning |
|---|---|---|
| `LOUD_MARGIN_DB` | 15 dB | How far above baseline counts as "loud" |
| `DETECT_WINDOW_S` | 10 s | Sustained-bark window length |
| `LOUD_RATIO` | 0.80 | Fraction of frames in the window that must be loud |
| `BASELINE_WINDOW_S` | 30 s | Rolling baseline length |
| `COOLDOWN_S` | 30 s | Minimum gap between alerts |

## Known MVP limitations

- **No species classification.** Loud TV, vacuum, or someone shouting will also trigger. v2 plan: add a small audio classifier (e.g. YAMNet) before the threshold step.
- **Single-channel.** Array-mic features (direction, denoising) not used.
- **Standalone.** Not yet wired into the team's dashboard UI.
