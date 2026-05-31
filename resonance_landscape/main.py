"""
Audio-Reactive Resonance Landscape - main rendering pipeline.

Analyzes a cinematic orchestral track, renders the resonance landscape
frame by frame with ModernGL, and exports an .mp4 with the original audio
muxed back in.

Usage:
    python main.py path/to/music.mp3
    python main.py music.mp3 --out resonance.mp4 --fps 30 --width 1280 --height 720
    python main.py music.mp3 --bands 48 --no-audio

Requires ffmpeg on PATH for muxing audio (pip: imageio-ffmpeg provides one).

Course 6178S - Seminar in Visual Computing, Group 08
Team: Dhanush Boobalan, Kongarasan Sathiya Moorthy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np

from audio_analysis import analyze
from renderer import ResonanceRenderer


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def mux_audio(video_no_audio: str, audio_path: str, out_path: str) -> bool:
    """Combine rendered video with original audio. Returns True on success."""
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        print("[warn] ffmpeg not found - keeping silent video.")
        return False
    cmd = [
        ffmpeg, "-y",
        "-i", video_no_audio,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg mux failed: {e}")
        return False


def render(audio_path: str, out_path: str, fps: int, width: int, height: int,
           n_bands: int, with_audio: bool, max_seconds: float | None) -> str:
    print(f"[1/3] Analyzing audio: {audio_path}")
    feats = analyze(audio_path, fps=fps, n_bands=n_bands)
    n_frames = feats.n_frames
    if max_seconds:
        n_frames = min(n_frames, int(max_seconds * fps))
    print(f"      duration={feats.duration:.1f}s  frames={n_frames}  bands={n_bands}")

    print(f"[2/3] Rendering {n_frames} frames at {width}x{height}...")
    rnd = ResonanceRenderer(width=width, height=height, n_bands=n_bands)

    tmp_video = out_path + ".silent.mp4"
    writer = imageio.get_writer(
        tmp_video, fps=fps, codec="libx264", quality=8,
        macro_block_size=None, ffmpeg_log_level="error",
    )
    try:
        for i in range(n_frames):
            f = feats.at(i)
            frame = rnd.render(
                t=i / fps,
                amp=f["amplitude"], bass=f["bass"], onset=f["onset"],
                spectrum=f["spectrum"], waveform=f["waveform"],
            )
            writer.append_data(frame)
            if i % max(1, n_frames // 20) == 0:
                pct = 100.0 * i / max(1, n_frames)
                print(f"      {pct:5.1f}%  frame {i}/{n_frames}", end="\r")
    finally:
        writer.close()
        rnd.release()
    print(f"\n      rendered -> {tmp_video}")

    print("[3/3] Muxing audio + finalizing...")
    if with_audio and mux_audio(tmp_video, audio_path, out_path):
        os.remove(tmp_video)
    else:
        shutil.move(tmp_video, out_path)
    print(f"      done -> {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Audio-Reactive Resonance Landscape")
    p.add_argument("audio", help="input audio file (mp3/wav/flac/...)")
    p.add_argument("--out", default=None, help="output .mp4 path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--bands", type=int, default=48, help="frequency bands")
    p.add_argument("--no-audio", action="store_true", help="skip audio muxing")
    p.add_argument("--seconds", type=float, default=None,
                   help="limit render to first N seconds (preview)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.audio):
        print(f"error: file not found: {args.audio}")
        sys.exit(1)

    out = args.out or (os.path.splitext(args.audio)[0] + "_resonance.mp4")
    render(
        audio_path=args.audio, out_path=out, fps=args.fps,
        width=args.width, height=args.height, n_bands=args.bands,
        with_audio=not args.no_audio, max_seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
