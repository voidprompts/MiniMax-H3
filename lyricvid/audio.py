"""Audio decode + musical analysis (tempo, beats, bars, energy, spectrum bands)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_wav(src: str, sr: int = 22050, mono: bool = True, dst: str | None = None) -> str:
    if dst is None:
        dst = os.path.join(tempfile.gettempdir(), "lyricvid_" + os.path.basename(src) + f".{sr}.wav")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vn",
           "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "wav", dst]
    subprocess.run(cmd, check=True)
    return dst


def load_wav(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as w:
        n_ch, sampwidth, rate, n_frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n_frames)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    y /= float(np.iinfo(dtype).max)
    if n_ch > 1:
        y = y.reshape(-1, n_ch).mean(axis=1)
    if rate != sr:
        import librosa

        y = librosa.resample(y, orig_sr=rate, target_sr=sr)
        rate = sr
    return np.ascontiguousarray(y), rate


@dataclass
class Analysis:
    sr: int
    duration: float
    tempo: float
    hop: int = 512
    beat_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bar_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onset_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onset_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vocal_energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bands: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    band_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bands_per_frame: int = 56

    def _interp(self, arr: np.ndarray, times: np.ndarray, grid: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros_like(grid)
        return np.interp(grid, times, arr)

    def sample(self, grid: np.ndarray) -> dict[str, np.ndarray]:
        import librosa

        frame_times = librosa.frames_to_time(np.arange(self.onset_env.size), sr=self.sr, hop_length=self.hop)
        rms_t = self.rms_times if self.rms_times.size else frame_times
        return {
            "onset": self._interp(self.onset_env, frame_times, grid),
            "rms": self._interp(self.rms, rms_t, grid),
            "centroid": self._interp(self.centroid, frame_times, grid),
            "vocal": self._interp(self.vocal_energy, frame_times, grid),
        }


def analyze(path: str, sr: int = 22050, bands_per_frame: int = 56, decode: bool = True) -> Analysis:
    import librosa

    wav = decode_to_wav(path, sr=sr) if decode else path
    y, sr = load_wav(wav, sr=sr)
    hop = 512
    duration = float(len(y) / sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, onset_envelope=onset_env,
                                              backtrack=False, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=2048)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    centroid = (S * freqs[:, None]).sum(0) / np.maximum(S.sum(0), 1e-9)

    vmask = (freqs >= 180) & (freqs <= 3500)
    vocal = np.log1p(S[vmask].mean(0) * 40.0) if vmask.any() else rms

    edges = np.geomspace(55.0, min(9000.0, sr / 2 * 0.92), bands_per_frame + 1)
    idx = []
    for i in range(bands_per_frame):
        b = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if b.size == 0:
            b = np.array([max(1, np.searchsorted(freqs, np.sqrt(edges[i] * edges[i + 1])))])
        idx.append(b)
    bands = np.log1p(np.stack([S[b].mean(0) for b in idx]) * 60.0)
    dec = np.zeros_like(bands)
    last = np.zeros(bands.shape[0])
    for f in range(bands.shape[1]):
        last = np.maximum(bands[:, f], last * 0.86)
        dec[:, f] = last
    bands = dec
    band_times = librosa.frames_to_time(np.arange(bands.shape[1]), sr=sr, hop_length=hop)

    bar_times = beat_times[::4] if beat_times.size >= 4 else beat_times.copy()

    return Analysis(sr=sr, duration=duration, tempo=tempo, hop=hop, beat_times=beat_times,
                    bar_times=bar_times, onset_env=onset_env, onset_times=onset_times, rms=rms,
                    rms_times=rms_times, centroid=centroid, vocal_energy=vocal, bands=bands,
                    band_times=band_times, bands_per_frame=bands_per_frame)


def snap(t: float, grid: np.ndarray, tol: float) -> float:
    if grid.size == 0:
        return t
    c = float(grid[np.argmin(np.abs(grid - t))])
    return c if abs(c - t) <= tol else t
