# Audio-Reactive Resonance Landscape

Real-time-style audio visualization for cinematic orchestral music.
Course 6178S — Seminar in Visual Computing, Group 08
Team: Dhanush Boobalan, Kongarasan Sathiya Moorthy

The track is rendered as a digital **resonance landscape**: a bright central
light point as the music's origin, bass-driven resonance rings, neon waveform
ridges on both sides, a background equalizer, and a reflective floor — all on a
dark, symmetrical, cinematic stage.

## Files

- `audio_analysis.py` — feature extraction (librosa/NumPy/SciPy): amplitude,
  bass energy, FFT frequency bands, onset strength, smoothed waveform.
- `renderer.py` — ModernGL renderer. The whole scene is drawn in one
  full-screen GLSL fragment shader; features arrive as uniforms + a 1D texture.
- `main.py` — pipeline: analyze → render frames → export `.mp4` (audio muxed
  back with FFmpeg).

## Feature → visual mapping

| Audio feature        | Visual element                          |
|----------------------|------------------------------------------|
| Overall amplitude    | Global scene brightness / intensity      |
| Low-frequency energy | Central halo size + resonance ring glow  |
| Frequency spectrum   | Equalizer bar heights                    |
| Onset / beat         | Pulse/flash of the central light         |
| Waveform (bands)     | Side neon landscape ridges               |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py path/to/music.mp3
# options:
python main.py music.mp3 --out resonance.mp4 --fps 30 --width 1920 --height 1080
python main.py music.mp3 --seconds 15        # quick preview of first 15s
python main.py music.mp3 --bands 64          # more equalizer bands
python main.py music.mp3 --no-audio          # skip audio muxing
```

Output defaults to `<input>_resonance.mp4`.

## Notes

- Rendering uses an OpenGL standalone context. On a normal Windows/macOS/Linux
  desktop this works out of the box. On a **headless server** install Mesa/EGL
  (`apt install libegl1 libgl1-mesa-dri`); `renderer.py` automatically tries the
  EGL backend first, then the default display backend.
- FFmpeg is needed only for muxing the original audio onto the video.
  `imageio-ffmpeg` ships a bundled binary, so no separate install is required.
