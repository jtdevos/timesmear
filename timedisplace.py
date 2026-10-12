# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["opencv-python", "numpy"]
# ///
"""Time displacement effect.

Each output pixel shows the input pixel from some time in the past:

    offset_seconds = mask[y, x] * window

where mask is a grayscale image (black = 0 = now, white = 1 = `window` seconds ago).
Before enough history exists, pixels clamp to the first frame of the video.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def load_mask(path, width, height):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        sys.exit(f"could not read mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    max_val = 65535.0 if mask.dtype == np.uint16 else 255.0
    mask = mask.astype(np.float32) / max_val
    if mask.shape != (height, width):
        print(f"note: stretching {mask.shape[1]}x{mask.shape[0]} mask to fit {width}x{height} video")
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask


class TimeDisplacer:
    def __init__(self, mask, window, fps, fast=False):
        self.h, self.w = mask.shape
        self.fast = fast
        offsets = mask * window * fps  # in frames, per pixel
        if fast:
            self.k0 = np.rint(offsets).astype(np.int64)
            self.frac = None
        else:
            self.k0 = np.floor(offsets).astype(np.int64)
            # Blend weight as an integer 0..256 so blending can stay in integer math.
            self.frac = np.rint((offsets - self.k0) * 256)[..., None].astype(np.int32)
        # Blending reads one frame further back than k0.
        self.size = int(self.k0.max()) + 2
        self.buf = None
        self.pix = np.arange(self.h * self.w, dtype=np.int64).reshape(self.h, self.w)
        self.t = 0

    def _gather(self, k):
        src = np.maximum(self.t - k, 0) % self.size
        flat = self.buf.reshape(-1, 3)
        return flat[src * (self.h * self.w) + self.pix]

    def push(self, frame):
        if self.buf is None:
            self.buf = np.empty((self.size, self.h, self.w, 3), dtype=np.uint8)
        self.buf[self.t % self.size] = frame
        if self.fast:
            out = self._gather(self.k0)
        else:
            a = self._gather(self.k0).astype(np.int32)
            b = self._gather(self.k0 + 1).astype(np.int32)
            out = (a + (((b - a) * self.frac + 128) >> 8)).astype(np.uint8)
        self.t += 1
        return out


CODECS = {
    ".webm": ["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-row-mt", "1", "-cpu-used", "4",
              "-pix_fmt", "yuv420p", "-c:a", "libopus"],
}
DEFAULT_CODECS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac"]


def mux_audio(video_only, original, dest):
    """Re-encode the processed video for dest's container, copying in the original's audio (if any)."""
    codecs = CODECS.get(dest.suffix.lower(), DEFAULT_CODECS)
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(original),
         "-map", "0:v", "-map", "1:a?", *codecs, "-shortest", str(dest)],
    )
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="input video")
    p.add_argument("mask", type=Path, help="grayscale mask image, same size as the video")
    p.add_argument("window", type=float, help="time window in seconds (offset for a white pixel)")
    p.add_argument("-o", "--output", type=Path, help="output video (default: <input>_displaced.mp4)")
    p.add_argument("--fast", action="store_true", help="use nearest frame instead of blending")
    p.add_argument("--preview", action="store_true", help="show frames in a window while processing (q to quit)")
    p.add_argument("--no-output", action="store_true", help="preview only, don't write a file")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mask = load_mask(args.mask, width, height)
    fx = TimeDisplacer(mask, args.window, fps, fast=args.fast)
    mb = fx.size * width * height * 3 / 1e6
    print(f"{width}x{height} @ {fps:.2f}fps, {total} frames; buffering {fx.size} frames (~{mb:.0f} MB)")

    write = not args.no_output
    if write:
        output = args.output or args.input.with_name(f"{args.input.stem}_displaced.mp4")
        tmpdir = tempfile.mkdtemp()
        raw_path = Path(tmpdir) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    delay = max(1, int(1000 / fps))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = fx.push(frame)
        if write:
            writer.write(out)
        if args.preview:
            cv2.imshow("timedisplace", out)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
        n += 1
        if n % 30 == 0:
            print(f"\r{n}/{total}", end="", flush=True)
    print(f"\r{n}/{total}")

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    if write:
        writer.release()
        if shutil.which("ffmpeg") and mux_audio(raw_path, args.input, output):
            print(f"wrote {output}")
        else:
            # The raw file is always MP4, so don't save it under another container's extension.
            fallback = output.with_suffix(".mp4")
            shutil.move(raw_path, fallback)
            print(f"wrote {fallback} (no audio; ffmpeg unavailable or failed)")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
