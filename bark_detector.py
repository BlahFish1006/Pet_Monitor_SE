"""Abnormal barking detector — Scenario 1 of Pet Edge Tracking System.

Pipeline: mic → frame RMS → dBFS → rolling baseline → threshold → 10s sliding
window → trigger Windows notification when sustained loud audio is detected.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

try:
    from plyer import notification
    _HAS_PLYER = True
except ImportError:
    _HAS_PLYER = False


SAMPLE_RATE = 16000
FRAME_MS = 100
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Rolling baseline: median of the last N seconds of frame dBFS values.
BASELINE_WINDOW_S = 30
BASELINE_FRAMES = BASELINE_WINDOW_S * 1000 // FRAME_MS

# A frame counts as "loud" when it exceeds baseline by this margin.
LOUD_MARGIN_DB = 15.0

# Sustained-barking detection window.
DETECT_WINDOW_S = 10
DETECT_FRAMES = DETECT_WINDOW_S * 1000 // FRAME_MS
LOUD_RATIO = 0.80  # ≥80% of frames in the window must be loud

# Cooldown so a single barking episode only fires once.
COOLDOWN_S = 30

# Floor for absolute silence so log10(0) doesn't blow up and a dead mic doesn't
# produce a baseline of -inf that everything trivially exceeds.
SILENCE_FLOOR_DBFS = -80.0
MIN_BASELINE_DBFS = -60.0


def rms_dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
    if rms <= 1e-10:
        return SILENCE_FLOOR_DBFS
    return 20.0 * np.log10(rms)


@dataclass
class DetectorState:
    baseline_buf: deque
    window_buf: deque
    last_alert_ts: float = 0.0


def notify(title: str, message: str) -> None:
    if _HAS_PLYER:
        try:
            notification.notify(title=title, message=message, timeout=10)
            return
        except Exception as e:
            print(f"[warn] plyer notify failed: {e}", file=sys.stderr)
    # Fallback: terminal bell + bold print
    print(f"\a\n*** {title}: {message} ***\n")


def process_frame(frame_db: float, state: DetectorState) -> tuple[float, float, bool]:
    """Returns (baseline_db, loud_ratio, triggered)."""
    state.baseline_buf.append(frame_db)
    # Use median for robustness; clamp so a very quiet room doesn't make
    # ordinary speech look like a barking episode.
    baseline = max(float(np.median(state.baseline_buf)), MIN_BASELINE_DBFS)
    threshold = baseline + LOUD_MARGIN_DB

    is_loud = frame_db > threshold
    state.window_buf.append(is_loud)
    ratio = sum(state.window_buf) / len(state.window_buf)

    triggered = False
    window_full = len(state.window_buf) == DETECT_FRAMES
    cooled_down = (time.time() - state.last_alert_ts) > COOLDOWN_S
    if window_full and ratio >= LOUD_RATIO and cooled_down:
        triggered = True
        state.last_alert_ts = time.time()

    return baseline, ratio, triggered


def run(device: int | None, verbose: bool) -> None:
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    state = DetectorState(
        baseline_buf=deque(maxlen=BASELINE_FRAMES),
        window_buf=deque(maxlen=DETECT_FRAMES),
    )

    print("Pet Monitor — Abnormal Barking Detector (MVP)")
    print(f"  sample_rate={SAMPLE_RATE} Hz, frame={FRAME_MS} ms")
    print(f"  baseline window={BASELINE_WINDOW_S}s, detect window={DETECT_WINDOW_S}s")
    print(f"  trigger when ≥{int(LOUD_RATIO*100)}% of frames exceed baseline+{LOUD_MARGIN_DB}dB")
    print("Listening... (Ctrl+C to stop)\n")

    last_print = 0.0
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        device=device,
        callback=audio_callback,
    ):
        while True:
            frame = audio_q.get()
            frame_db = rms_dbfs(frame)
            baseline, ratio, triggered = process_frame(frame_db, state)

            now = time.time()
            if verbose and now - last_print > 0.5:
                bar = "#" * int(ratio * 20)
                print(
                    f"  level={frame_db:6.1f} dBFS  baseline={baseline:6.1f}  "
                    f"loud_ratio={ratio*100:5.1f}%  [{bar:<20}]",
                    end="\r",
                )
                last_print = now

            if triggered:
                msg = (
                    f"Sustained loud audio detected for {DETECT_WINDOW_S}s "
                    f"(level≈{frame_db:.0f} dBFS, baseline≈{baseline:.0f} dBFS)"
                )
                print(f"\n[ALERT {time.strftime('%H:%M:%S')}] {msg}")
                notify("Pet Monitor: Abnormal Barking", msg)


def list_devices() -> None:
    print(sd.query_devices())


def main() -> None:
    parser = argparse.ArgumentParser(description="Abnormal barking detector (MVP)")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print live levels")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        run(args.device, args.verbose)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
