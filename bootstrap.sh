#!/usr/bin/env bash
# Download reference videos and generate default masks into resources/.
# Safe to re-run: anything already present is skipped.
set -euo pipefail
cd "$(dirname "$0")"

for tool in uv ffmpeg; do
    command -v "$tool" >/dev/null || { echo "missing required tool: $tool (try: brew install $tool)" >&2; exit 1; }
done
if command -v yt-dlp >/dev/null; then
    ytdlp=(yt-dlp)
else
    ytdlp=(uvx yt-dlp)
fi

# --- videos ---------------------------------------------------------------
lists=(resources/videos.txt)
[[ -f resources/videos.local.txt ]] && lists+=(resources/videos.local.txt)

failed=0
while read -r name url section _; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    if compgen -G "resources/$name.*" >/dev/null; then
        echo "skip $name (already downloaded)"
        continue
    fi
    echo "download $name${section:+ [$section]}"
    # Prefer VP9 over AV1: OpenCV's bundled decoder handles VP9 reliably.
    args=(--no-warnings --no-playlist -S "res:1080,vcodec:vp9" --merge-output-format "webm/mkv"
          --downloader-args "ffmpeg:-hide_banner -loglevel error" -o "resources/$name.%(ext)s")
    [[ -n "${section:-}" ]] && args+=(--download-sections "*$section")
    if ! "${ytdlp[@]}" "${args[@]}" "$url"; then
        echo "FAILED $name" >&2
        failed=1
    fi
done < <(cat "${lists[@]}")

# --- masks ----------------------------------------------------------------
# timedisplace.py stretches masks to fit, so one 1080p set works for any video.
for pattern in horizontal vertical radial plasma; do
    out="resources/mask-$pattern.png"
    if [[ -e "$out" ]]; then
        echo "skip $out (exists)"
    else
        uv run --quiet makemask.py "$pattern" 1920 1080 --seed 1 -o "$out"
    fi
done

exit "$failed"
