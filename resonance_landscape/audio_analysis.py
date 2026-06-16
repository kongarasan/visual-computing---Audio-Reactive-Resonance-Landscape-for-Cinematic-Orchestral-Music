"""
Audio analysis for the Audio-Reactive Resonance Landscape.

Extracts, per video frame, the features that drive the visuals:
  - amplitude        : overall RMS loudness        -> scene brightness/intensity
  - bass             : low-frequency energy        -> central halo + resonance rings
  - spectrum         : N frequency bands (FFT)     -> equalizer bar heights
  - onset            : onset/beat strength         -> light pulses / flashes
  - waveform         : smoothed envelope per band  -> side neon landscape

Course 6178S - Seminar in Visual Computing, Group 08
Team: Dhanush Boobalan, Kongarasan Sathiya Moorthy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa


@dataclass
class AudioFeatures:
    """Per-frame, normalized (0..1) audio features sampled at the video fps."""
    fps: int
    duration: float
    n_frames: int
    n_bands: int

    amplitude: np.ndarray      # (n_frames,)
    bass: np.ndarray           # (n_frames,)
    melody: np.ndarray         # (n_frames,)  mid-frequency energy -> color hue shift
    treble: np.ndarray         # (n_frames,)  high-frequency energy -> brightness/sparkle
    onset: np.ndarray          # (n_frames,)
    spectrum: np.ndarray       # (n_frames, n_bands)
    waveform: np.ndarray       # (n_frames, n_bands)  smoothed band envelopes

    def at(self, frame: int) -> dict:
        i = int(np.clip(frame, 0, self.n_frames - 1))
        return {
            "amplitude": float(self.amplitude[i]),
            "bass": float(self.bass[i]),
            "melody": float(self.melody[i]),
            "treble": float(self.treble[i]),
            "onset": float(self.onset[i]),
            "spectrum": self.spectrum[i],
            "waveform": self.waveform[i],
        }


def _compress(x: np.ndarray) -> np.ndarray:
    """Perceptual compression of a power signal -> closer to how we hear loudness.

    The spectrogram is power (magnitude^2); sqrt brings it back to magnitude so a
    quiet passage is not crushed against a few loud transients."""
    return np.sqrt(np.maximum(np.asarray(x, dtype=np.float32), 0.0))


def _normalize(
    x: np.ndarray, lo: float = 5.0, hi: float = 99.0, gamma: float = 1.0
) -> np.ndarray:
    """Robust min-max contrast stretch to 0..1.

    Uses a low percentile as the noise floor and a high percentile as the
    reference peak, so the full dynamic range of the track is used (better
    contrast than a top-only divide). Both ends are robust to outliers.
    `gamma` < 1 lifts quiet detail; > 1 emphasises peaks.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    if x.size == 0:
        return x
    floor = float(np.percentile(x, lo))
    ref = float(np.percentile(x, hi))
    if ref - floor <= 1e-9:
        # near-constant signal: fall back to plain max scaling
        m = float(x.max())
        return np.zeros_like(x) if m <= 1e-9 else np.clip(x / m, 0.0, 1.0)
    y = np.clip((x - floor) / (ref - floor), 0.0, 1.0)
    if gamma != 1.0:
        y = np.power(y, gamma, dtype=np.float32)
    return y


def _resample_to_frames(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a (T,) or (T, B) feature curve onto n_frames samples."""
    x = np.asarray(x, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=True)
    dst = np.linspace(0.0, 1.0, num=n_frames, endpoint=True)
    if x.ndim == 1:
        return np.interp(dst, src, x).astype(np.float32)
    out = np.empty((n_frames, x.shape[1]), dtype=np.float32)
    for b in range(x.shape[1]):
        out[:, b] = np.interp(dst, src, x[:, b])
    return out


def analyze(
    audio_path: str,
    fps: int = 30,
    n_bands: int = 48,
    sr: int = 22050,
    n_fft: int = 2048,
    smoothing: float = 0.35,
) -> AudioFeatures:
    """
    Analyze an audio file and return per-frame visual-control features.

    smoothing : exponential smoothing factor for the waveform landscape
                (0 = no smoothing, ->1 = very smooth).
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y) / sr
    n_frames = max(1, int(round(duration * fps)))

    hop = max(1, int(round(sr / fps)))  # ~one STFT frame per video frame

    # --- Magnitude spectrogram -------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2  # power
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # --- Overall amplitude (RMS) ----------------------------------------------
    rms = librosa.feature.rms(S=np.sqrt(S))[0]
    amplitude = _normalize(rms)

    # --- Bass energy (20-200 Hz) ----------------------------------------------
    # Perceptual compression (sqrt) then a robust stretch; a slight gamma lift
    # keeps the low end visibly responsive without clipping on big hits.
    bass_mask = (freqs >= 20) & (freqs <= 200)
    bass_raw = _compress(S[bass_mask, :].sum(axis=0))
    bass = _normalize(bass_raw, gamma=0.8)

    # --- Melody energy (200-2000 Hz) ------------------------------------------
    melody_mask = (freqs >= 200) & (freqs <= 2000)
    melody_raw = _compress(S[melody_mask, :].sum(axis=0))
    melody = _normalize(melody_raw)

    # --- Treble energy (4000-12000 Hz) ----------------------------------------
    treble_mask = (freqs >= 4000) & (freqs <= 12000)
    treble_raw = _compress(S[treble_mask, :].sum(axis=0))
    treble = _normalize(treble_raw, gamma=0.85)

    # --- Onset strength --------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset = _normalize(onset_env, hi=97.0)

    # --- Frequency bands (log-spaced) for equalizer ----------------------------
    # Map linear FFT bins into n_bands log-spaced bands.
    f_min, f_max = 30.0, min(sr / 2.0, 16000.0)
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)
    bands = np.zeros((S.shape[1], n_bands), dtype=np.float32)
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not np.any(m):
            # nearest bin fallback
            idx = int(np.argmin(np.abs(freqs - 0.5 * (edges[b] + edges[b + 1]))))
            bands[:, b] = S[idx, :]
        else:
            bands[:, b] = S[m, :].mean(axis=0)
    # log compression then per-band normalization
    bands = np.log1p(bands)
    spectrum = np.stack([_normalize(bands[:, b]) for b in range(n_bands)], axis=1)

    # --- Resample everything onto the video frame grid ------------------------
    amplitude = _resample_to_frames(amplitude, n_frames)
    bass = _resample_to_frames(bass, n_frames)
    melody = _resample_to_frames(melody, n_frames)
    treble = _resample_to_frames(treble, n_frames)
    onset = _resample_to_frames(onset, n_frames)
    spectrum = _resample_to_frames(spectrum, n_frames)

    # --- Waveform landscape: temporally smoothed spectrum ----------------------
    waveform = np.empty_like(spectrum)
    acc = spectrum[0].copy()
    a = float(np.clip(smoothing, 0.0, 0.99))
    for i in range(n_frames):
        acc = a * acc + (1.0 - a) * spectrum[i]
        waveform[i] = acc

    return AudioFeatures(
        fps=fps,
        duration=duration,
        n_frames=n_frames,
        n_bands=n_bands,
        amplitude=amplitude,
        bass=bass,
        melody=melody,
        treble=treble,
        onset=onset,
        spectrum=spectrum,
        waveform=waveform,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    feats = analyze(path)
    print(f"duration={feats.duration:.1f}s  frames={feats.n_frames}  bands={feats.n_bands}")
    print("amplitude range:", feats.amplitude.min(), feats.amplitude.max())
    print("bass range:", feats.bass.min(), feats.bass.max())
