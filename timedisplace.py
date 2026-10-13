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

  # the same, with orange, spring green and violet ghosts instead of red, green and blue
  uv run timedisplace.py in.webm 1.5 -t color-plasma --seed 7 --hue 30

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


# --hue stores frames as amounts of three hue-rotated primaries. Those can fall outside 0..255
# (|amount| <= 255 * sqrt(3)), so they're kept as uint16: (amount + HUE_OFFSET) * HUE_SCALE.
HUE_OFFSET, HUE_SCALE = 450.0, 64.0


def hue_basis(degrees):
    """3x3 matrix whose columns are blue, green and red rotated `degrees` around the gray axis,
    in BGR coordinates. Gray stays gray, and the primaries (and so the ghost colors) shift hue."""
    k = np.full(3, 1 / np.sqrt(3))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.radians(degrees)
    rgb = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K  # Rodrigues' rotation formula
    flip = np.eye(3)[::-1]
    return flip @ rgb @ flip


@njit(parallel=True, cache=True)
def _to_basis(frame, m, out):
    """BGR uint8 -> amounts of the basis colors (m is the inverse basis), stored as uint16."""
    h, w, _ = frame.shape
    for y in prange(h):
        for x in range(w):
            for i in range(3):
                v = m[i, 0] * frame[y, x, 0] + m[i, 1] * frame[y, x, 1] + m[i, 2] * frame[y, x, 2]
                out[y, x, i] = np.uint16((v + HUE_OFFSET) * HUE_SCALE + 0.5)


@njit(parallel=True, cache=True)
def _from_basis(amounts, m, out):
    """Basis amounts (uint16) -> BGR uint8, clipping colors that mixing moments pushed out of range."""
    h, w, _ = amounts.shape
    for y in prange(h):
        for x in range(w):
            a0 = amounts[y, x, 0] / HUE_SCALE - HUE_OFFSET
            a1 = amounts[y, x, 1] / HUE_SCALE - HUE_OFFSET
            a2 = amounts[y, x, 2] / HUE_SCALE - HUE_OFFSET
            for i in range(3):
                v = m[i, 0] * a0 + m[i, 1] * a1 + m[i, 2] * a2
                out[y, x, i] = min(max(int(v + 0.5), 0), 255)


class TimeDisplacer:
    """Ring buffer of recent frames, sampled per pixel (grayscale mask) or per channel (color mask)."""

    def __init__(self, mask, window, fps, fast=False, dtype=np.uint8):
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
        self.buf = np.empty((self.size, self.h, self.w, 3), dtype=dtype)
        self.t = 0

    def push(self, frame):
        self.buf[self.t % self.size] = frame
        out = np.empty((self.h, self.w, 3), dtype=self.buf.dtype)
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
X264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
# Apple's hardware H.264 encoder: ~2x faster than x264, and -q:v 75 matches x264 CRF 18's
# quality (VMAF) for ~14% larger files. Only available on Macs.
VIDEOTOOLBOX = ["-c:v", "h264_videotoolbox", "-q:v", "75", "-pix_fmt", "yuv420p"]


def default_codecs():
    """H.264/AAC, using the hardware encoder when this ffmpeg has it."""
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return (VIDEOTOOLBOX if "h264_videotoolbox" in encoders else X264) + ["-c:a", "aac"]


def open_encoder(dest, original, width, height, fps, audio_delay=0):
    """Start ffmpeg reading raw BGR frames on stdin, encoding for dest's container and
    copying in the original's audio (if any), audio_delay seconds late. Frames are encoded
    once, straight from memory."""
    codecs = CODECS.get(dest.suffix.lower()) or default_codecs()
    # adelay pads the start with silence; -shortest then trims the tail to the video's length.
    delay = ["-af", f"adelay=delays={round(audio_delay * 1000)}:all=1"] if audio_delay > 0 else []
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
         "-i", str(original), "-map", "0:v", "-map", "1:a?", *codecs, *delay, "-shortest", str(dest)],
        stdin=subprocess.PIPE,
    )


def ycc_mask(mask, only):
    """Rearrange a mask to match frames in OpenCV's YCrCb channel order (Y, Cr, Cb).

    A color mask's red, green and blue channels set the Y, Cb and Cr delays. With only="luma"
    or "chroma", a grayscale mask delays just that part of the picture; the rest stays live."""
    if only:
        zero = np.zeros_like(mask)
        return np.dstack([mask, zero, zero] if only == "luma" else [zero, mask, mask])
    if mask.ndim == 3:
        b, g, r = mask[..., 0], mask[..., 1], mask[..., 2]
        return np.dstack([r, b, g])  # Y <- red, Cr <- blue, Cb <- green
    return mask


def default_output(input_path, mask_label, window, fast, space_tag=None):
    """e.g. dance.mp4, color-plasma-seed7, 1.5, False -> dance_color-plasma-seed7_1.5s.mp4"""
    parts = [input_path.stem, mask_label, f"{window:g}s"] + ([space_tag] if space_tag else [])
    parts += ["fast"] if fast else []
    return input_path.with_name("_".join(parts) + ".mp4")


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
                        "(default: <input>_<mask>_<window>s[_ycc|_luma-only|_chroma-only|_hueDEG][_fast].mp4 "
                        "next to the input, "
                        "e.g. dance_color-plasma-seed7_1.5s.mp4)")
    p.add_argument("--space", choices=["rgb", "ycc"], default="rgb",
                   help="channels a color mask delays: rgb = red/green/blue (default); ycc = brightness "
                        "(Y, from the mask's red) and color (Cb, Cr, from green and blue)")
    p.add_argument("--only", choices=["luma", "chroma"],
                   help="delay only brightness (luma) or only color (chroma) by a grayscale mask; "
                        "the other stays live (implies --space ycc)")
    p.add_argument("--audio-delay", type=float, metavar="SECONDS",
                   help="play the original audio this much later, to match the delayed picture "
                        "(default: half the window, the average delay of a mid-gray mask; 0 keeps the "
                        "audio in sync with the input, e.g. for lip sync with a radial mask's live center)")
    p.add_argument("--hue", type=float, default=0, metavar="DEG",
                   help="with a color mask: rotate the three primaries this many degrees around the color "
                        "wheel, changing the ghost colors (0 = red/green/blue, the default; 30 = orange/spring "
                        "green/violet, the most different; 60 = yellow/cyan/magenta, dimmer)")
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
    if args.audio_delay is None:
        args.audio_delay = args.window / 2
    if args.audio_delay < 0:
        p.error("--audio-delay can't be negative (the picture never shows the future)")
    if args.hue and (args.space == "ycc" or args.only):
        p.error("--hue works on red/green/blue, so it can't be combined with --space ycc or --only")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # mask_label names the mask in the default output filename.
    if args.mask:
        mask = load_mask(args.mask, width, height)
        mask_label = args.mask.stem
    else:
        mask_type = args.mask_type or "radial"
        desc = mask_label = mask_type
        if mask_type in RANDOM_PATTERNS:
            seed = args.seed if args.seed is not None else random.randrange(2**31)
            desc += f" mask (--seed {seed})"
            mask_label += f"-seed{seed}"
        else:
            seed = None
            desc += " mask"
        if mask_type in ("plasma", "color-plasma") and args.roughness != 0.5:
            mask_label += f"-rough{args.roughness:g}"
        mask = make(mask_type, width, height, args.roughness, seed)
        print(f"generated {desc}")
    ycc = args.space == "ycc" or args.only is not None
    if ycc:
        if args.only and mask.ndim == 3:
            sys.exit("--only needs a grayscale mask")
        mask = ycc_mask(mask, args.only)
    if args.hue and mask.ndim == 2:
        sys.exit("--hue needs a color mask (with a grayscale mask every primary has the same delay)")
    if args.hue:
        basis = hue_basis(args.hue)
        unbasis = basis.T  # a rotation's inverse is its transpose
        amounts = np.empty((height, width, 3), np.uint16)
        rgb_out = np.empty((height, width, 3), np.uint8)
    fx = TimeDisplacer(mask, args.window, fps, fast=args.fast, dtype=np.uint16 if args.hue else np.uint8)
    mb = fx.buf.nbytes / 1e6
    kind = "color mask, per-channel delays" if fx.color else "grayscale mask"
    if args.only:
        kind = f"{args.only} only"
    elif ycc:
        kind += " in YCbCr"
    elif args.hue:
        kind += f", primaries rotated {args.hue:g} degrees"
    print(f"{width}x{height} @ {fps:.2f}fps, {total} frames, {kind}; buffering {fx.size} frames (~{mb:.0f} MB)")

    encoder = writer = None
    if not args.no_output:
        space_tag = f"{args.only}-only" if args.only else ("ycc" if ycc else (f"hue{args.hue:g}" if args.hue else None))
        output = args.output or default_output(args.input, mask_label, args.window, args.fast, space_tag)
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            encoder = open_encoder(output, args.input, width, height, fps, args.audio_delay)
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
        if ycc:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        if args.hue:
            _to_basis(frame, unbasis, amounts)
            frame = amounts
        out = fx.push(frame)
        if args.hue:
            _from_basis(out, basis, rgb_out)
            out = rgb_out
        if ycc:
            # Values that mix brightness and color from different moments can fall outside
            # RGB; the conversion clips them.
            out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
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
