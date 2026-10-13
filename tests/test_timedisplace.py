# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["opencv-python", "numpy", "numba"]
# ///
"""Checks that the optimized TimeDisplacer matches a plain NumPy reference bit for bit.

Run with:  uv run tests/test_timedisplace.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from makemask import make  # noqa: E402
from timedisplace import TimeDisplacer, _from_basis, _to_basis, hue_basis, ycc_mask  # noqa: E402


def reference(mask, window, fps, frames, fast):
    """The straightforward version of the effect: gather whole frames' worth of pixels with NumPy."""
    offsets = mask * window * fps
    if mask.ndim == 2:
        offsets = np.repeat(offsets[..., None], 3, axis=2)
    k0 = (np.rint(offsets) if fast else np.floor(offsets)).astype(np.int64)
    frac = np.rint((offsets - k0) * 256).astype(np.int32)
    yy, xx, cc = np.indices(k0.shape)
    outs = []
    for t in range(len(frames)):
        a = np.stack(frames)[np.maximum(t - k0, 0), yy, xx, cc].astype(np.int32)
        if fast:
            outs.append(a.astype(np.uint8))
            continue
        b = np.stack(frames)[np.maximum(t - k0 - 1, 0), yy, xx, cc].astype(np.int32)
        outs.append((a + (((b - a) * frac + 128) >> 8)).astype(np.uint8))
    return outs


def check_matches_reference():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (36, 48, 3), dtype=np.uint8) for _ in range(70)]
    for pattern in ("plasma", "radial", "color-plasma", "color-radial"):
        mask = make(pattern, 48, 36, seed=3)
        for fast in (False, True):
            fx = TimeDisplacer(mask, 1.7, 24, fast)
            expected = reference(mask, 1.7, 24, frames, fast)
            for t, frame in enumerate(frames):
                assert (fx.push(frame) == expected[t]).all(), f"{pattern} fast={fast} differs at frame {t}"
    print("ok: matches reference for gray and color masks, blended and --fast")


def check_per_channel_delays():
    frames = [np.full((4, 6, 3), t, np.uint8) for t in range(60)]
    mask = np.zeros((4, 6, 3), np.float32)
    mask[..., 1], mask[..., 2] = 0.5, 1.0  # B now, G 10 frames back, R 20 frames back
    fx = TimeDisplacer(mask, 2.0, 10)
    outs = [fx.push(f) for f in frames]
    assert tuple(outs[45][0, 0]) == (45, 35, 25)
    assert tuple(outs[5][0, 0]) == (5, 0, 0)  # clamps to the first frame
    print("ok: each channel uses its own delay")


def check_gray_equals_equal_channels():
    rng = np.random.default_rng(1)
    gray = rng.random((20, 30), dtype=np.float32)
    frames = [rng.integers(0, 256, (20, 30, 3), dtype=np.uint8) for _ in range(40)]
    a, b = TimeDisplacer(gray, 1.3, 10), TimeDisplacer(np.dstack([gray] * 3), 1.3, 10)
    assert all((a.push(f) == b.push(f)).all() for f in frames)
    print("ok: a grayscale mask equals the same mask in all three channels")


def check_ycc():
    # Frames in OpenCV's YCrCb order whose value in every part is the frame number, so the
    # output reads off how many frames back each part looked.
    frames = [np.full((4, 6, 3), t, np.uint8) for t in range(60)]

    def delays(mask, only=None):
        fx = TimeDisplacer(ycc_mask(mask, only), 2.0, 10)
        outs = [fx.push(f) for f in frames]
        return tuple(45 - int(v) for v in outs[45][0, 0])  # (Y, Cr, Cb)

    color = np.zeros((4, 6, 3), np.float32)
    color[..., 0], color[..., 1], color[..., 2] = 0.25, 0.5, 1.0  # BGR: blue 5, green 10, red 20 frames
    assert delays(color) == (20, 5, 10), "Y from the mask's red, Cr from blue, Cb from green"
    gray = np.full((4, 6), 0.5, np.float32)
    assert delays(gray) == (10, 10, 10)
    assert delays(gray, "luma") == (10, 0, 0), "--only luma keeps color live"
    assert delays(gray, "chroma") == (0, 10, 10), "--only chroma keeps brightness live"

    # Through the real conversions, a picture whose parts all agree comes back as it went in.
    rng = np.random.default_rng(5)
    still = rng.integers(0, 256, (36, 48, 3), dtype=np.uint8)
    fx = TimeDisplacer(ycc_mask(make("color-plasma", 48, 36, seed=3), None), 1.3, 10)
    for _ in range(20):
        out = cv2.cvtColor(fx.push(cv2.cvtColor(still, cv2.COLOR_BGR2YCrCb)), cv2.COLOR_YCrCb2BGR)
    roundtrip = cv2.cvtColor(cv2.cvtColor(still, cv2.COLOR_BGR2YCrCb), cv2.COLOR_YCrCb2BGR)
    assert (out == roundtrip).all()
    print("ok: YCbCr sends each mask channel to the right part of the picture, and --only keeps the other live")


def check_hue():
    basis = hue_basis(30)
    assert np.allclose(hue_basis(0), np.eye(3))
    assert np.allclose(basis @ basis.T, np.eye(3)), "a rotation, so its transpose undoes it"
    assert np.allclose(basis @ np.ones(3), np.ones(3)), "gray stays gray"

    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 256, (36, 48, 3), dtype=np.uint8) for _ in range(40)]
    amounts = np.empty((36, 48, 3), np.uint16)
    out = np.empty((36, 48, 3), np.uint8)
    _to_basis(frames[0], basis.T, amounts)
    _from_basis(amounts, basis, out)
    assert (out == frames[0]).all(), "converting to the rotated primaries and back is exact"

    # With the same delay for every primary, it's the ordinary effect (to within rounding).
    gray = make("plasma", 48, 36, seed=3)
    plain, rotated = TimeDisplacer(gray, 1.3, 10), TimeDisplacer(np.dstack([gray] * 3), 1.3, 10, dtype=np.uint16)
    for f in frames:
        _to_basis(f, basis.T, amounts)
        _from_basis(rotated.push(amounts), basis, out)
        assert np.abs(out.astype(int) - plain.push(f)).max() <= 1

    # Black turns white at frame 20; the rotated red is 10 frames late, so at frame 25 the
    # picture is white minus rotated red: the complement of the shifted primary, not cyan.
    mask = np.zeros((2, 2, 3), np.float32)
    mask[..., 2] = 0.5  # BGR order: the third primary is (rotated) red
    fx = TimeDisplacer(mask, 2.0, 10, dtype=np.uint16)
    amounts, out = amounts[:2, :2].copy(), out[:2, :2].copy()
    for t in range(26):
        _to_basis(np.full((2, 2, 3), 200 if t >= 20 else 0, np.uint8), basis.T, amounts)
        delayed = fx.push(amounts)
    _from_basis(delayed, basis, out)
    expected = np.clip(np.rint(200 * (basis[:, 0] + basis[:, 1])), 0, 255)
    assert (out[0, 0] == expected).all(), (out[0, 0], expected)
    print("ok: --hue round-trips exactly, matches the plain effect for equal delays, and shifts ghost colors")


if __name__ == "__main__":
    check_matches_reference()
    check_per_channel_delays()
    check_gray_equals_equal_channels()
    check_ycc()
    check_hue()
