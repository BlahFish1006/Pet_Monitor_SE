"""Pet Edge Tracking System — unified local dashboard (PySide6).

Combines the two cores of the system behind one window:
  * Vision core  -> yolo_world_detector.py  (YOLO-World dog/cat detection on video)
  * Audio core   -> bark_detector.py        (abnormal/continuous barking detection)

Everything runs locally (no cloud). Two QThread workers feed the GUI thread via
Qt signals, so there is no shared-state race:

    AudioWorker  --status(level, baseline, ratio)-->  MainWindow
                 --bark_alert(message, timestamp)-->  MainWindow
    VideoWorker  --frame(qimage, fps, counts)----->  MainWindow

Usage:
    python pet_dashboard.py --source sample.mp4 --model dogandcat.pt
    optional: --device <mic-index>  --conf 0.25  --no-loop  --stride N
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import numpy as np

# --- Cores (imported as libraries; their CLIs remain usable standalone) ------
import bark_detector as bd
import yolo_world_detector as yw

try:
    from n8n_client import build_alert_trigger, send_event
    _HAS_N8N = True
except ImportError:
    _HAS_N8N = False

# Classes the vision core detects. Only pets are in the vocabulary, so non-pets
# (people, furniture, ...) are simply never detected — no false-positive alarms.
PET_CLASSES = {"dog", "cat"}
DEFAULT_DETECT_CLASSES = ["dog", "cat"]

# --- Qt -----------------------------------------------------------------------
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# OpenCV for video capture / frame handling (already a dependency of the vision core)
import cv2


# ============================================================================ #
# Audio worker — reuses bark_detector's detection logic
# ============================================================================ #
class AudioWorker(QThread):
    """Runs the bark-detection pipeline on either a live microphone or the
    audio track of the input video. Reuses rms_dbfs / process_frame /
    DetectorState and emits Qt signals instead of printing.

    audio_source="mic"   -> sounddevice InputStream (live)
    audio_source="video" -> PyAV decodes the video's audio track, resampled to
                            16 kHz mono, sliced into 100 ms frames, paced to
                            real time so the detector behaves like live input.
    """

    status = Signal(float, float, float)   # (level_dbfs, baseline_dbfs, loud_ratio)
    bark_alert = Signal(str, str)          # (message, timestamp)

    def __init__(self, audio_source="mic", device=None, video_path=None,
                 loop=True, loud_margin=bd.LOUD_MARGIN_DB,
                 loud_ratio=bd.LOUD_RATIO, n8n_webhook=None, parent=None):
        super().__init__(parent)
        self.audio_source = audio_source
        self.device = device
        self.video_path = video_path
        self.loop = loop
        self.loud_margin = loud_margin
        self.loud_ratio = loud_ratio
        self.n8n_webhook = n8n_webhook
        self._running = True
        self._last_emit = 0.0

    def stop(self):
        self._running = False

    def _new_state(self):
        return bd.DetectorState(
            baseline_buf=deque(maxlen=bd.BASELINE_FRAMES),
            window_buf=deque(maxlen=bd.DETECT_FRAMES),
        )

    def _process_chunk(self, chunk, state):
        """Feed one 100 ms frame through the detector and emit signals."""
        frame_db = bd.rms_dbfs(chunk)
        baseline, ratio, triggered = bd.process_frame(
            frame_db, state, self.loud_margin, self.loud_ratio
        )
        now = time.time()
        if now - self._last_emit > 0.2:
            self.status.emit(float(frame_db), float(baseline), float(ratio))
            self._last_emit = now
        if triggered:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"Sustained loud audio for {bd.DETECT_WINDOW_S}s "
                f"(level≈{frame_db:.0f} dBFS, baseline≈{baseline:.0f} dBFS)"
            )
            self.bark_alert.emit(msg, ts)
            bd.notify("Pet Monitor: Abnormal Barking", msg)  # reuse core OS notification
            if self.n8n_webhook and _HAS_N8N:
                send_event(self.n8n_webhook, build_alert_trigger(
                    event_type="abnormal_barking", confidence_pct=round(ratio * 100),
                    timestamp=ts, scenario=1, message=msg,
                ))

    def run(self):
        if self.audio_source == "video":
            self._run_video()
        else:
            self._run_mic()

    def _run_mic(self):
        import queue

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_q.put(indata[:, 0].copy())

        state = self._new_state()
        try:
            import sounddevice as sd
            stream = sd.InputStream(
                samplerate=bd.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=bd.FRAME_SAMPLES,
                device=self.device,
                callback=audio_callback,
            )
        except Exception as e:  # no mic / no sounddevice -> audio panel just stays idle
            self.bark_alert.emit(f"[audio disabled] {e}", time.strftime("%H:%M:%S"))
            return

        with stream:
            while self._running:
                try:
                    frame = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._process_chunk(frame, state)

    def _run_video(self):
        """Decode the video's audio track and run detection paced to real time."""
        try:
            import av
        except ImportError:
            self.bark_alert.emit("[audio disabled] PyAV (av) not installed",
                                 time.strftime("%H:%M:%S"))
            return

        state = self._new_state()
        fs = bd.FRAME_SAMPLES
        frame_dur = bd.FRAME_MS / 1000.0

        while self._running:
            try:
                container = av.open(self.video_path)
            except Exception as e:
                self.bark_alert.emit(f"[audio disabled] cannot open video audio: {e}",
                                     time.strftime("%H:%M:%S"))
                return
            if not any(s.type == "audio" for s in container.streams):
                self.bark_alert.emit("[audio disabled] video has no audio track",
                                     time.strftime("%H:%M:%S"))
                container.close()
                return

            resampler = av.AudioResampler(
                format="flt", layout="mono", rate=bd.SAMPLE_RATE
            )
            buf = np.empty(0, dtype=np.float32)
            try:
                for aframe in container.decode(audio=0):
                    if not self._running:
                        break
                    for of in (resampler.resample(aframe) or []):
                        samples = of.to_ndarray().astype(np.float32).flatten()
                        buf = np.concatenate([buf, samples])
                        while len(buf) >= fs and self._running:
                            chunk, buf = buf[:fs], buf[fs:]
                            t0 = time.time()
                            self._process_chunk(chunk, state)
                            sleep_left = frame_dur - (time.time() - t0)
                            if sleep_left > 0:
                                self.msleep(int(sleep_left * 1000))
            finally:
                container.close()

            if not self.loop:
                break


# ============================================================================ #
# Video worker — reuses yolo_world_detector's model
# ============================================================================ #
class VideoWorker(QThread):
    """Reads frames from a video file, runs YOLO-World per (strided) frame,
    emits the annotated frame, FPS, and per-class detection counts."""

    frame_ready = Signal(object, float, dict)   # (QImage, fps, {"dog":n,"cat":m})
    pet_event = Signal(str, str, str)           # (kind, label, timestamp)
    finished_video = Signal()

    def __init__(self, model, source, conf=yw.DEFAULT_CONF, loop=True,
                 stride=1, imgsz=480, n8n_webhook=None, danger_zone=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.source = source
        self.conf = conf
        self.loop = loop
        self.stride = max(1, int(stride))
        self.imgsz = imgsz
        self.n8n_webhook = n8n_webhook
        self.danger_zone = danger_zone   # (x1,y1,x2,y2) normalized 0-1, or None
        self._running = True
        # Scenario 0 (pet in/out) + Scenario 2 (danger zone) tracking state
        self._pet_state = None           # None / "present" / "absent"
        self._last_pet_seen = 0.0
        self._zone_occupied = False
        self._last_danger = 0.0

    def stop(self):
        self._running = False

    def _to_qimage(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def _fire_event(self, kind, scenario, label, message, confidence=None, to_n8n=True):
        """Emit a GUI event (RECENT EVENTS + banner + log); optionally POST the
        ICD-COMP-UI-001 ALERT_TRIGGER to n8n.

        pet_in/pet_out are kept local-only (to_n8n=False) so they don't show up
        as "Unmatched" in the n8n routing demo.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.pet_event.emit(kind, label, ts)
        if to_n8n and self.n8n_webhook and _HAS_N8N:
            send_event(self.n8n_webhook, build_alert_trigger(
                event_type=kind, confidence_pct=confidence, timestamp=ts,
                scenario=scenario, message=message,
            ))

    def _zone_pixels(self, w, h):
        x1, y1, x2, y2 = self.danger_zone
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

    def _detect_red_zone(self, bgr):
        """Find a red 'Forbidden Zone' box drawn in the frame -> (x1,y1,x2,y2) px."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) |
                cv2.inRange(hsv, (170, 90, 90), (180, 255, 255)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.008 * bgr.shape[0] * bgr.shape[1]:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, x + w, y + h)

    def _check_intrusion(self, pet_boxes, zone_px):
        """Scenario 2: fire a warning when a pet bbox enters the zone (debounced).
        Confidence_% = the highest YOLO confidence among pets inside the zone."""
        zx1, zy1, zx2, zy2 = zone_px
        best_conf = None
        for bx1, by1, bx2, by2, cf in pet_boxes:
            if not (bx2 < zx1 or bx1 > zx2 or by2 < zy1 or by1 > zy2):
                best_conf = cf if best_conf is None else max(best_conf, cf)
        occupied = best_conf is not None
        if occupied and not self._zone_occupied and (time.time() - self._last_danger) > 10:
            self._fire_event("danger_zone", 2, "禁區警報",
                             "Pet entered forbidden zone", confidence=round(best_conf * 100))
            self._last_danger = time.time()
        self._zone_occupied = occupied

    def _update_inout(self, counts):
        """Scenario 0: fire pet_out when pets leave the frame >2 s, pet_in on return."""
        pets = counts.get("dog", 0) + counts.get("cat", 0)
        now = time.time()
        if pets > 0:
            self._last_pet_seen = now
            if self._pet_state == "absent":
                self._fire_event("pet_in", 0, "寵物返回", "Pet entered frame", to_n8n=False)
            self._pet_state = "present"
        elif self._pet_state == "present" and now - self._last_pet_seen > 2.0:
            self._fire_event("pet_out", 0, "寵物外出", "Pet left frame", to_n8n=False)
            self._pet_state = "absent"
        elif self._pet_state is None:
            self._pet_state = "absent"

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished_video.emit()
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_period = 1.0 / src_fps if src_fps > 0 else 0.04

        idx = 0
        last_annotated = None
        last_counts = {"dog": 0, "cat": 0}
        t_prev = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Run detection only every `stride` frames; reuse last boxes otherwise.
            if idx % self.stride == 0:
                results = self.model.predict(
                    frame, conf=self.conf, imgsz=self.imgsz, verbose=False
                )
                r = results[0]
                last_annotated = r.plot()
                counts = {"dog": 0, "cat": 0}
                pet_boxes = []   # list of (x1, y1, x2, y2, conf)
                if r.boxes is not None:
                    for c, box, cf in zip(r.boxes.cls.tolist(),
                                          r.boxes.xyxy.tolist(),
                                          r.boxes.conf.tolist()):
                        name = self.model.names[int(c)]
                        if name in counts:
                            counts[name] += 1
                        if name in PET_CLASSES:
                            pet_boxes.append((*box, cf))
                last_counts = counts

                # Scenario 2: danger zone. "auto" reads the red box drawn in the
                # video; a tuple is a fixed ROI we draw ourselves.
                if self.danger_zone == "auto":
                    zp = self._detect_red_zone(frame)
                    if zp:
                        self._check_intrusion(pet_boxes, zp)
                elif self.danger_zone:
                    h, w = last_annotated.shape[:2]
                    zx1, zy1, zx2, zy2 = self._zone_pixels(w, h)
                    cv2.rectangle(last_annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                    cv2.putText(last_annotated, "DANGER ZONE",
                                (zx1, max(zy1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                    self._check_intrusion(pet_boxes, (zx1, zy1, zx2, zy2))
                # Scenario 0: pet in/out events.
                self._update_inout(counts)
            display = last_annotated if last_annotated is not None else frame
            idx += 1

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = (1.0 / dt) if dt > 0 else 0.0

            self.frame_ready.emit(self._to_qimage(display), fps, dict(last_counts))

            # Throttle toward the source frame rate (detection latency usually dominates).
            sleep_left = frame_period - (time.time() - now)
            if sleep_left > 0:
                self.msleep(int(sleep_left * 1000))

        cap.release()
        self.finished_video.emit()


# ============================================================================ #
# Helper widgets
# ============================================================================ #
DARK_BG = "#15181c"
PANEL_BG = "#1e2228"
ACCENT = "#3fb950"
ALERT = "#e5484d"
TEXT = "#e6edf3"
MUTED = "#8b949e"

# RECENT EVENTS / banner colour + emoji per event kind.
EVENT_STYLES = {
    "danger_zone": ("#e5484d", "🚨"),
    "pet_out": ("#3b82f6", "🚪"),
    "pet_in": ("#3fb950", "🏠"),
}


class StatusRow(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setStyleSheet(f"color:{TEXT}; font-size:14px;")


def _hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color:{MUTED};")
    return line


# ============================================================================ #
# Main window
# ============================================================================ #
class MainWindow(QMainWindow):
    MAX_EVENTS = 8

    def __init__(self, model, args):
        super().__init__()
        self.setWindowTitle("Pet Edge Tracking System — Dashboard")
        self.resize(1180, 720)
        self.setStyleSheet(f"background:{DARK_BG};")

        self._start_ts = time.time()
        self._last_qimage = None  # keep latest frame for event thumbnails
        self.log_path = getattr(args, "log_file", "events.log") or None

        # ---------------- Header ----------------
        self.status_lbl = QLabel("CURRENT STATUS: MONITORING")
        self.status_lbl.setStyleSheet(f"color:{ACCENT}; font-size:18px; font-weight:bold;")
        self.uptime_lbl = QLabel("UP-TIME 00:00:00")
        self.uptime_lbl.setStyleSheet(f"color:{TEXT}; font-size:14px;")
        header = QHBoxLayout()
        header.addWidget(self.status_lbl)
        header.addStretch()
        header.addWidget(self.uptime_lbl)

        # ---------------- Video panel ----------------
        self.video_lbl = QLabel("Loading video…")
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setMinimumSize(720, 480)
        self.video_lbl.setStyleSheet(
            f"background:#000; color:{MUTED}; border:2px solid {PANEL_BG};"
        )
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Alert banner (hidden until a bark fires)
        self.banner = QLabel("")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet(
            f"background:{ALERT}; color:white; font-size:16px; font-weight:bold; padding:8px;"
        )
        self.banner.hide()

        video_box = QVBoxLayout()
        video_box.addWidget(self.banner)
        video_box.addWidget(self.video_lbl, 1)

        # ---------------- Status panel ----------------
        panel = QVBoxLayout()
        title = QLabel("PET REAL-TIME STATUS")
        title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:bold;")
        panel.addWidget(title)
        panel.addWidget(_hline())

        self.fps_row = StatusRow("VISUAL FPS: —")
        self.dog_row = StatusRow("DOGS: 0")
        self.cat_row = StatusRow("CATS: 0")
        panel.addWidget(self.fps_row)
        panel.addWidget(self.dog_row)
        panel.addWidget(self.cat_row)
        panel.addWidget(_hline())

        self.level_row = StatusRow("AUDIO LEVEL: — dBFS")
        self.baseline_row = StatusRow("BASELINE: — dBFS")
        self.ratio_row = StatusRow("LOUD RATIO: 0.0%  [                    ]")
        panel.addWidget(self.level_row)
        panel.addWidget(self.baseline_row)
        panel.addWidget(self.ratio_row)
        panel.addStretch()

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(320)
        panel_widget.setStyleSheet(f"background:{PANEL_BG}; border-radius:6px; padding:10px;")

        # ---------------- Middle row (video + panel) ----------------
        mid = QHBoxLayout()
        mid.addLayout(video_box, 1)
        mid.addWidget(panel_widget)

        # ---------------- Recent events strip ----------------
        ev_title = QLabel("RECENT EVENTS")
        ev_title.setStyleSheet(f"color:{TEXT}; font-size:14px; font-weight:bold;")
        self.events_row = QHBoxLayout()
        self.events_row.addStretch()
        events_widget = QWidget()
        events_widget.setLayout(self.events_row)
        events_widget.setFixedHeight(130)
        events_widget.setStyleSheet(f"background:{PANEL_BG}; border-radius:6px;")

        # ---------------- Assemble ----------------
        root = QVBoxLayout()
        root.addLayout(header)
        root.addLayout(mid, 1)
        root.addWidget(ev_title)
        root.addWidget(events_widget)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # ---------------- Workers ----------------
        self.video_worker = VideoWorker(
            model, args.source, conf=args.conf, loop=not args.no_loop,
            stride=args.stride, n8n_webhook=args.n8n_webhook,
            danger_zone=args.danger_zone,
        )
        self.video_worker.frame_ready.connect(self.on_frame)
        self.video_worker.pet_event.connect(self.on_pet_event)
        self.video_worker.finished_video.connect(self.on_video_finished)

        self.audio_worker = AudioWorker(
            audio_source=args.audio_source, device=args.device,
            video_path=args.source, loop=not args.no_loop,
            loud_margin=args.loud_margin, loud_ratio=args.loud_ratio,
            n8n_webhook=args.n8n_webhook,
        )
        self.audio_worker.status.connect(self.on_audio_status)
        self.audio_worker.bark_alert.connect(self.on_bark_alert)

        self.video_worker.start()
        self.audio_worker.start()

        # Uptime ticker + banner auto-hide
        self._uptime_timer = QTimer(self)
        self._uptime_timer.timeout.connect(self._tick_uptime)
        self._uptime_timer.start(1000)
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self.banner.hide)

    # ---------------- Slots ----------------
    def on_frame(self, qimage, fps, counts):
        self._last_qimage = qimage
        pix = QPixmap.fromImage(qimage).scaled(
            self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_lbl.setPixmap(pix)
        self.fps_row.setText(f"VISUAL FPS: {fps:4.1f}")
        self.dog_row.setText(f"DOGS: {counts.get('dog', 0)}")
        self.cat_row.setText(f"CATS: {counts.get('cat', 0)}")

    def on_audio_status(self, level, baseline, ratio):
        bar = "#" * int(ratio * 20)
        self.level_row.setText(f"AUDIO LEVEL: {level:6.1f} dBFS")
        self.baseline_row.setText(f"BASELINE: {baseline:6.1f} dBFS")
        self.ratio_row.setText(f"LOUD RATIO: {ratio*100:5.1f}%  [{bar:<20}]")

    def _show_banner(self, text, color, ms=6000):
        self.banner.setStyleSheet(
            f"background:{color}; color:white; font-size:16px; font-weight:bold; padding:8px;"
        )
        self.banner.setText(text)
        self.banner.show()
        self._banner_timer.start(ms)

    def _log_event(self, kind, label, ts, detail=""):
        """Append an event to the on-disk log (紀錄日誌)."""
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{ts} | {kind} | {label} | {detail}\n")
        except Exception as e:
            print(f"[warn] log write failed: {e}", file=sys.stderr)

    def on_bark_alert(self, message, timestamp):
        self._show_banner(f"🔊 異常吠叫通知  {timestamp}\n{message}", ALERT)
        self._add_event("異常吠叫", timestamp.split(" ")[-1], ALERT)
        self._log_event("abnormal_barking", "異常吠叫", timestamp, message)

    def on_pet_event(self, kind, label, timestamp):
        color, emoji = EVENT_STYLES.get(kind, (ACCENT, ""))
        self._show_banner(f"{emoji} {label}  {timestamp}", color, 4000)
        self._add_event(label, timestamp.split(" ")[-1], color)
        self._log_event(kind, label, timestamp)

    def on_video_finished(self):
        self.status_lbl.setText("CURRENT STATUS: VIDEO ENDED")
        self.status_lbl.setStyleSheet(f"color:{MUTED}; font-size:18px; font-weight:bold;")

    def _add_event(self, label, ts, color="#e5484d"):
        cell = QVBoxLayout()
        thumb = QLabel()
        thumb.setFixedSize(150, 84)
        thumb.setStyleSheet(f"border:2px solid {color};")
        if self._last_qimage is not None:
            thumb.setPixmap(QPixmap.fromImage(self._last_qimage).scaled(
                150, 84, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            thumb.setStyleSheet(f"background:#000; border:2px solid {color};")
        cap = QLabel(f"{label}\n{ts}")
        cap.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        cap.setAlignment(Qt.AlignCenter)
        cell.addWidget(thumb)
        cell.addWidget(cap)
        holder = QWidget()
        holder.setLayout(cell)
        # newest first (insert at index 0, before the trailing stretch)
        self.events_row.insertWidget(0, holder)

        # cap the number of thumbnails kept
        if self.events_row.count() - 1 > self.MAX_EVENTS:
            item = self.events_row.takeAt(self.events_row.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()

    def _tick_uptime(self):
        s = int(time.time() - self._start_ts)
        self.uptime_lbl.setText(f"UP-TIME {s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}")

    # ---------------- Lifecycle ----------------
    def closeEvent(self, event):
        self.video_worker.stop()
        self.audio_worker.stop()
        self.video_worker.wait(2000)
        self.audio_worker.wait(2000)
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Pet Edge Tracking System — unified dashboard")
    parser.add_argument("--source", required=True, help="Video file path for the vision core")
    parser.add_argument("--model", default="dogandcat.pt", help="YOLO-World model (.pt)")
    parser.add_argument("--conf", type=float, default=yw.DEFAULT_CONF, help="Detection confidence")
    parser.add_argument("--stride", type=int, default=1,
                        help="Run detection every Nth frame (reuse boxes between) for smoother display")
    parser.add_argument("--no-loop", action="store_true", help="Stop at end of video instead of looping")
    parser.add_argument("--audio-source", choices=["mic", "video"], default="mic",
                        help="Audio for bark detection: live microphone (default) or the --source video's audio track")
    parser.add_argument("--device", type=int, default=None, help="Microphone input device index")
    parser.add_argument("--loud-margin", type=float, default=bd.LOUD_MARGIN_DB,
                        help="dB above baseline for a frame to count as loud")
    parser.add_argument("--loud-ratio", type=float, default=bd.LOUD_RATIO,
                        help="fraction of frames in the window that must be loud")
    parser.add_argument("--n8n-webhook", type=str, default=None,
                        help="n8n Webhook URL to POST events to (Action Output stage)")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_DETECT_CLASSES),
                        help="Comma-separated detection vocabulary (pets=dog,cat trigger events; others shown as ignored)")
    parser.add_argument("--danger-zone", type=str, default=None,
                        help="Scenario 2 forbidden zone: 'auto' to detect a red box drawn in the "
                             "video, or x1,y1,x2,y2 normalized 0-1 (e.g. 0.04,0.45,0.30,0.92)")
    parser.add_argument("--log-file", type=str, default="events.log",
                        help="Append events to this log file (use '' to disable)")
    args = parser.parse_args()

    if args.danger_zone and args.danger_zone != "auto":
        try:
            args.danger_zone = tuple(float(v) for v in args.danger_zone.split(","))
            assert len(args.danger_zone) == 4
        except Exception:
            parser.error("--danger-zone must be 'auto' or 4 comma-separated floats: x1,y1,x2,y2")

    print(f"[*] Loading vision model: {args.model}")
    model = yw.load_model(args.model)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        model.set_classes(classes)
        # set_classes updates the detection embeddings but NOT model.names, so a
        # newly-added class (e.g. person -> id 2) would break r.plot() and lookups.
        # Force the id->name map to match the class order we passed.
        model.model.names = {i: c for i, c in enumerate(classes)}
        print(f"[*] Detection classes: {classes}  (pets={sorted(PET_CLASSES)})")
    except Exception as e:
        print(f"[warn] set_classes failed, using model defaults: {e}")

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow(model, args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
