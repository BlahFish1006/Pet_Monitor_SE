"""Generate a reproducible A/V test clip for the dashboard's --audio-source video mode.

Video: letterboxed slideshow of the downloaded test/Dog + test/Cat images.
Audio: quiet noise (baseline) then a sustained loud burst, so the bark detector
should fire while the clip plays. Output: sample_av.mp4 (H.264/MPEG-4 + AAC).
"""
import glob
from fractions import Fraction

import av
import cv2
import numpy as np

W, H, FPS, SEC_PER_IMG = 960, 540, 25, 2
SR = 16000
OUT = "sample_av.mp4"


def letterbox(path):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(im, (nw, nh))
    top, left = (H - nh) // 2, (W - nw) // 2
    return cv2.copyMakeBorder(r, top, H - nh - top, left, W - nw - left,
                              cv2.BORDER_CONSTANT, value=(20, 20, 24))


imgs = sorted(glob.glob("test/Dog/*")) + sorted(glob.glob("test/Cat/*"))
canvases = [c for c in (letterbox(p) for p in imgs) if c is not None]
total_frames = len(canvases) * FPS * SEC_PER_IMG
duration_s = total_frames / FPS
print(f"[*] {len(canvases)} images -> {total_frames} video frames (~{duration_s:.0f}s)")

# --- Audio: quiet baseline, then a loud sustained burst, then quiet ---
rng = np.random.default_rng(0)
n = int(duration_s * SR)
audio = (rng.standard_normal(n).astype(np.float32)) * 0.01  # quiet floor (~-40 dBFS)
loud_start = int(11 * SR)
loud_end = min(int(19 * SR), n)
audio[loud_start:loud_end] += rng.standard_normal(loud_end - loud_start).astype(np.float32) * 0.45
audio = np.clip(audio, -1.0, 1.0)
print(f"[*] audio: {n} samples @ {SR}Hz, loud burst {11}s..{loud_end/SR:.0f}s")

container = av.open(OUT, mode="w")

vstream = container.add_stream("mpeg4", rate=FPS)
vstream.width, vstream.height, vstream.pix_fmt = W, H, "yuv420p"

astream = container.add_stream("aac", rate=SR)
astream.layout = "mono"

# Encode video
for canvas in canvases:
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    vframe = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    for _ in range(FPS * SEC_PER_IMG):
        for pkt in vstream.encode(vframe):
            container.mux(pkt)
for pkt in vstream.encode():
    container.mux(pkt)

# Encode audio in fltp frames of 1024 samples (AAC frame size)
FRAME = 1024
pts = 0
for i in range(0, len(audio), FRAME):
    block = audio[i:i + FRAME]
    af = av.AudioFrame.from_ndarray(block.reshape(1, -1), format="fltp", layout="mono")
    af.sample_rate = SR
    af.pts = pts
    af.time_base = Fraction(1, SR)
    pts += block.shape[0]
    for pkt in astream.encode(af):
        container.mux(pkt)
for pkt in astream.encode():
    container.mux(pkt)

container.close()
print(f"[*] wrote {OUT}")
