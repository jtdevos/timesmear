# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["opencv-python", "numpy", "numba"]
# ///
"""timesmear: time displacement video effect.

Each output pixel shows the same pixel from some time in the past:

    delay_seconds = mask[y, x] * window

where the mask is black = now, white = `window` seconds ago, grays in between.
With a color mask, each channel delays its own color: the red channel of the mask
sets how far back the red channel of the video looks, and so on.
Before enough history exists, pixels clamp to the first frame of the video.
"""

EXAMPLES = """\
examples:
  # center live, edges lag 1 second (radial mask generated on the fly)
  uv run timedisplace.py in.webm 1

  # smear across the frame: left edge is live, right edge lags 2 seconds
  uv run timedisplace.py in.webm 2 -t horizontal

  # red, green and blue drift apart in separate plasma clouds, written as WebM
  uv run timedisplace.py in.webm 1.5 -t color-plasma --seed 7 -o out.webm

  # use a mask image you made or painted
  uv run timedisplace.py in.webm 1.5 -m resources/mask-plasma.png

  # quick look at a setting without writing a file
  uv run timedisplace.py in.webm 1 -t plasma --fast --preview --no-output

masks: generate them with -t, make them with makemask.py, or paint one and save it
as a PNG (grayscale, or color for per-channel delays). A mask image of a different
size is stretched to fit the video.
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange

from makemask import PATTERNS, RANDOM_PATTERNS, make


def load_mask(path, width, height):
    """Load a mask as float32 0..1: (h, w) if grayscale, (h, w, 3) BGR if it has real color."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        sys.exit(f"could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, :3]  # drop alpha
        # A gray image saved as RGB has identical channels; treat it as grayscale (faster).
        if (mask == mask[:, :, :1]).all():
            mask = mask[:, :, 0]
    max_val = 65535.0 if mask.dtype == np.uint16 else 255.0
    mask = mask.astype(np.float32) / max_val
    if mask.shape[:2] != (height, width):
        print(f"note: stretching {mask.shape[1]}x{mask.shape[0]} mask to fit {width}x{height} video")
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask


# The kernels compile on first use and are cached in __pycache__, so only the very first run pays.
# k0 and frac have shape (h, w, 1) for a grayscale mask or (h, w, 3) for a color mask.

@njit(parallel=True, cache=True)
def _blend(buf, k0, frac, t, size, out):
    """Blend the frames k0 and k0 + 1 back, with weight frac/256 on the older one."""
    h, w, nk = k0.shape
    for y in prange(h):
        for x in range(w):
            for c in range(3):
                m = c if nk == 3 else 0
                s = t - k0[y, x, m]
                a = np.int32(buf[max(s, 0) % size, y, x, c])
                b = np.int32(buf[max(s - 1, 0) % size, y, x, c])
                out[y, x, c] = a + (((b - a) * frac[y, x, m] + 128) >> 8)


@njit(parallel=True, cache=True)
def _nearest(buf, k0, t, size, out):
    """Copy from the frame k0 back."""
    h, w, nk = k0.shape
    for y in prange(h):
        for x in range(w):
            for c in range(3):
                s = t - k0[y, x, c if nk == 3 else 0]
                out[y, x, c] = buf[max(s, 0) % size, y, x, c]


class TimeDisplacer:
    """Ring buffer of recent frames, sampled per pixel (grayscale mask) or per channel (color mask)."""

    def __init__(self, mask, window, fps, fast=False):
        self.h, self.w = mask.shape[:2]
        self.color = mask.ndim == 3
        self.fast = fast
        offsets = mask * window * fps  # in frames, per pixel (or per pixel channel)
        if not self.color:
            offsets = offsets[..., None]
        if fast:
            self.k0 = np.rint(offsets).astype(np.int32)
            self.frac = None
        else:
            self.k0 = np.floor(offsets).astype(np.int32)
            # Blend weight as an integer 0..256 so blending can stay in integer math.
            self.frac = np.rint((offsets - self.k0) * 256).astype(np.int32)
        # Blending reads one frame further back than k0.
        self.size = int(self.k0.max()) + 2
        self.buf = np.empty((self.size, self.h, self.w, 3), dtype=np.uint8)
        self.t = 0

    def push(self, frame):
        self.buf[self.t % self.size] = frame
        out = np.empty((self.h, self.w, 3), dtype=np.uint8)
        if self.fast:
            _nearest(self.buf, self.k0, self.t, self.size, out)
        else:
            _blend(self.buf, self.k0, self.frac, self.t, self.size, out)
        self.t += 1
        return out


CODECS = {
    ".webm": ["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-row-mt", "1", "-cpu-used", "4",
              "-pix_fmt", "yuv420p", "-c:a", "libopus"],
}
DEFAULT_CODECS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac"]


def open_encoder(dest, original, width, height, fps):
    """Start ffmpeg reading raw BGR frames on stdin, encoding for dest's container and
    copying in the original's audio (if any). Frames are encoded once, straight from memory."""
    codecs = CODECS.get(dest.suffix.lower(), DEFAULT_CODECS)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
         "-i", str(original), "-map", "0:v", "-map", "1:a?", *codecs, "-shortest", str(dest)],
        stdin=subprocess.PIPE,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="input video (anything ffmpeg can read: mp4, webm, mkv, mov, ...)")
    p.add_argument("window", type=float, help="delay in seconds for a white mask pixel")
    source = p.add_mutually_exclusive_group()
    source.add_argument("-m", "--mask", type=Path,
                        help="mask image: grayscale, or color for per-channel delays (16-bit PNG recommended)")
    source.add_argument("-t", "--mask-type", choices=PATTERNS, metavar="TYPE",
                        help="generate a mask at the video's size instead: %(choices)s (default: radial)")
    p.add_argument("--seed", type=int, help="with -t noise, plasma or a color type: random seed (default: random, printed)")
    p.add_argument("--roughness", type=float, default=0.5,
                   help="with -t plasma/color-plasma: 0-1, higher = more fine detail (default 0.5)")
    p.add_argument("-o", "--output", type=Path,
                   help="output video; .webm gives VP9/Opus, anything else H.264/AAC "
                        "(default: <input>_displaced.mp4)")
    p.add_argument("--fast", action="store_true", help="use the nearest frame instead of blending two (slightly steppier motion)")
    p.add_argument("--preview", action="store_true", help="show frames in a window while rendering (q to quit)")
    p.add_argument("--no-output", action="store_true", help="with --preview: don't write a file")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    args = p.parse_args()
    if args.no_output and not args.preview:
        p.error("--no-output only makes sense with --preview")
    if args.window <= 0:
        p.error("window must be greater than 0")
    if not 0 < args.roughness <= 1:
        p.error("--roughness must be between 0 and 1")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.mask:
        mask = load_mask(args.mask, width, height)
    else:
        mask_type = args.mask_type or "radial"
        desc = f"{mask_type} mask"
        if mask_type in RANDOM_PATTERNS:
            seed = args.seed if args.seed is not None else random.randrange(2**31)
            desc += f" (--seed {seed})"
        else:
            seed = None
        mask = make(mask_type, width, height, args.roughness, seed)
        print(f"generated {desc}")
    fx = TimeDisplacer(mask, args.window, fps, fast=args.fast)
    mb = fx.size * width * height * 3 / 1e6
    kind = "color mask, per-channel delays" if fx.color else "grayscale mask"
    print(f"{width}x{height} @ {fps:.2f}fps, {total} frames, {kind}; buffering {fx.size} frames (~{mb:.0f} MB)")

    encoder = writer = None
    if not args.no_output:
        output = args.output or args.input.with_name(f"{args.input.stem}_displaced.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            encoder = open_encoder(output, args.input, width, height, fps)
        else:
            # Without ffmpeg, OpenCV can still write a silent MP4.
            output = output.with_suffix(".mp4")
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    delay = max(1, int(1000 / fps))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = fx.push(frame)
        if encoder:
            try:
                encoder.stdin.write(out.data)
            except BrokenPipeError:
                sys.exit("ffmpeg stopped unexpectedly (see its error above)")
        elif writer:
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
    if encoder:
        encoder.stdin.close()
        if encoder.wait() != 0:
            sys.exit("ffmpeg failed to encode the output (see its error above)")
        print(f"wrote {output}")
    elif writer:
        writer.release()
        print(f"wrote {output} (no audio: ffmpeg not found)")


if __name__ == "__main__":
    main()
