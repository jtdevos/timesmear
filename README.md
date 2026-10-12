# timesmear

A **time displacement** video effect: every pixel of the output shows that same pixel from some
moment in the past, and a grayscale mask decides how far back.

- black (0) = now
- white (1) = `window` seconds ago
- grays fall in between, blended smoothly between frames

A horizontal gradient gives a classic slit-scan smear. A plasma fractal makes scene cuts
dissolve through in cloudy patches.

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python, OpenCV and NumPy on first run)
- [ffmpeg](https://ffmpeg.org/) (to carry over audio and encode the output)

```bash
brew install uv ffmpeg
```

## Quick start

```bash
./bootstrap.sh     # downloads reference clips + generates masks into resources/
uv run timedisplace.py resources/tears-of-steel.webm resources/mask-plasma.png 1.5
```

The output is written next to the input as `<name>_displaced.mp4`.

## timedisplace.py

```bash
uv run timedisplace.py INPUT MASK WINDOW [-o OUTPUT] [--fast] [--preview] [--no-output]
```

| Option | Meaning |
|---|---|
| `WINDOW` | Seconds of delay for a fully white mask pixel |
| `-o` | Output path. `.webm` gives VP9/Opus; anything else gives H.264/AAC |
| `--fast` | Use the nearest frame instead of blending two frames (about 2x faster) |
| `--preview` | Show the output in a window while it renders (press `q` to quit) |
| `--no-output` | Preview only, without writing a file |

If the mask is a different size from the video, it's stretched to fit. The tool keeps `WINDOW`
seconds of frames in memory: 3 s of 1080p at 24 fps is about 450 MB.

## makemask.py

```bash
uv run makemask.py PATTERN WIDTH HEIGHT [-o mask.png] [--invert] [--seed N] [--roughness R]
```

Available patterns are `horizontal`, `vertical`, `radial`, `noise` and `plasma`. `--roughness`
(0–1, default 0.5) controls how much fine detail plasma has. Masks are saved as 16-bit PNGs so
gradients don't band.

You can also paint a mask in any image editor and save it as a grayscale PNG.

## Reference videos

`bootstrap.sh` downloads the clips listed in `resources/videos.txt` with
[yt-dlp](https://github.com/yt-dlp/yt-dlp). They're one-minute excerpts from
[Blender open movies](https://studio.blender.org/films/), licensed CC-BY:
*Big Buck Bunny*, *Tears of Steel* and *Spring*.

To use your own videos without committing them, add them to `resources/videos.local.txt` in the
same format (`name url [start-end]`). That file is gitignored, along with everything else in
`resources/` and all video files.
