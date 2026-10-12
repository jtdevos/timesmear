# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["opencv-python", "numpy"]
# ///
"""Generate grayscale mask images for timedisplace.py (black = now, white = oldest)."""

import argparse

import cv2
import numpy as np


def plasma(w, h, roughness, rng):
    """Diamond-square plasma fractal on a (2^n + 1)-sized grid, cropped to w x h."""
    size = 2 ** int(np.ceil(np.log2(max(w, h, 2) - 1))) + 1
    g = np.zeros((size, size), dtype=np.float64)
    g[:: size - 1, :: size - 1] = rng.uniform(-1, 1, (2, 2))
    step, scale = size - 1, 1.0
    while step > 1:
        half = step // 2
        # Diamond step: the center of each square is the average of its 4 corners.
        corners = g[:-1:step, :-1:step] + g[:-1:step, step::step] + g[step::step, :-1:step] + g[step::step, step::step]
        centers = corners / 4 + rng.uniform(-scale, scale, corners.shape)
        g[half::step, half::step] = centers

        # Square step: each edge midpoint averages its neighbors (3 at the border, 4 inside).
        # Midpoints of horizontal edges: left/right are corners, up/down are centers.
        total = g[::step, :-1:step] + g[::step, step::step]
        count = np.full(total.shape, 2.0)
        total[1:] += centers
        count[1:] += 1
        total[:-1] += centers
        count[:-1] += 1
        g[::step, half::step] = total / count + rng.uniform(-scale, scale, total.shape)

        # Midpoints of vertical edges: up/down are corners, left/right are centers.
        total = g[:-1:step, ::step] + g[step::step, ::step]
        count = np.full(total.shape, 2.0)
        total[:, 1:] += centers
        count[:, 1:] += 1
        total[:, :-1] += centers
        count[:, :-1] += 1
        g[half::step, ::step] = total / count + rng.uniform(-scale, scale, total.shape)

        step, scale = half, scale * roughness
    m = g[:h, :w]
    m = (m - m.min()) / (m.max() - m.min())
    return m.astype(np.float32)


def make(pattern, w, h, roughness=0.5, seed=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    if pattern == "horizontal":
        return x
    if pattern == "vertical":
        return y
    if pattern == "radial":
        d = np.hypot(x - 0.5, (y - 0.5) * h / w)
        return d / d.max()
    if pattern == "noise":
        small = rng.random((max(1, h // 64), max(1, w // 64)), dtype=np.float32)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    if pattern == "plasma":
        return plasma(w, h, roughness, rng)
    raise ValueError(pattern)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pattern", choices=["horizontal", "vertical", "radial", "noise", "plasma"])
    p.add_argument("width", type=int)
    p.add_argument("height", type=int)
    p.add_argument("-o", "--output", default="mask.png")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--seed", type=int, help="random seed for noise/plasma (default: random)")
    p.add_argument("--roughness", type=float, default=0.5,
                   help="plasma only: 0-1, higher = more fine detail (default 0.5)")
    args = p.parse_args()

    m = make(args.pattern, args.width, args.height, args.roughness, args.seed)
    if args.invert:
        m = 1 - m
    # 16-bit PNG keeps smooth gradients from banding into 256 steps.
    cv2.imwrite(args.output, (m * 65535).astype(np.uint16))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
