
"""Audio Validation and Preprocessing Module for Vani Kavach.

Provides audio validation, decoding, channel unification (mono),
sample-rate standardization (16 kHz), amplitude normalization,
and CNN spectrogram feature extraction.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


class AudioValidationError(Exception):
    """Raised when an uploaded audio file fails validation."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Audio validation standards
TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 120.0
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

MIN_RMS_THRESHOLD = 0.001
MIN_RMS_ENERGY = MIN_RMS_THRESHOLD


@dataclass
class AudioMetadata:
    filename: str
    original_sample_rate: int
    target_sample_rate: int
    original_channels: int
    duration_seconds: float
    num_samples: int
    peak_amplitude: float
    rms_energy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedAudio:
    signal: np.ndarray
    sample_rate: int
    metadata: AudioMetadata


def validate_file_envelope(content: bytes, filename: str) -> None:
    """Validate file size and extension before decoding."""

    if not content:
        raise AudioValidationError(
            "Audio file is empty (0 bytes).",
            status_code=400,
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise AudioValidationError(
            f"File size exceeds maximum allowed limit of {max_mb} MB.",
            status_code=400,
        )

    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {supported}.",
            status_code=400,
        )


def resample_signal(
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample a 1D waveform using polyphase filtering."""

    if orig_sr == target_sr:
        return signal.astype(np.float32)

    common_gcd = gcd(orig_sr, target_sr)
    up = target_sr // common_gcd
    down = orig_sr // common_gcd

    resampled = scipy.signal.resample_poly(signal, up, down)
    return resampled.astype(np.float32)


def preprocess_audio(
    content: bytes,
    filename: str,
) -> PreprocessedAudio:
    """Validate, decode, and standardize uploaded audio."""

    validate_file_envelope(content, filename)

    try:
        data, orig_sr = sf.read(
            io.BytesIO(content),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise AudioValidationError(
            "Failed to decode audio file. "
            f"The file may be corrupt or malformed: {exc}",
            status_code=400,
        ) from exc

    if data.size == 0:
        raise AudioValidationError(
            "Decoded audio stream contains zero samples.",
            status_code=400,
        )

    if not np.all(np.isfinite(data)):
        raise AudioValidationError(
            "Audio contains non-finite or invalid numeric values (NaN or Inf).",
            status_code=400,
        )

    # Convert stereo or multichannel audio to mono.
    if data.ndim == 1:
        orig_channels = 1
        mono_signal = data
    elif data.ndim == 2:
        orig_channels = data.shape[1]
        mono_signal = np.mean(data, axis=1)
    else:
        raise AudioValidationError(
            f"Unsupported audio dimensionality: {data.ndim}D.",
            status_code=400,
        )

    if mono_signal.size == 0:
        raise AudioValidationError(
            "Decoded audio stream contains zero samples.",
            status_code=400,
        )

    duration = float(len(mono_signal) / orig_sr)

    if duration < MIN_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) is shorter than "
            f"minimum required ({MIN_DURATION_SECONDS}s).",
            status_code=400,
        )

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) exceeds "
            f"maximum allowed limit ({MAX_DURATION_SECONDS}s).",
            status_code=400,
        )

    # Standardize to 16 kHz.
    standardized_signal = resample_signal(
        mono_signal,
        orig_sr,
        TARGET_SAMPLE_RATE,
    )

    if standardized_signal.size == 0:
        raise AudioValidationError(
            "Standardized audio signal contains zero samples.",
            status_code=400,
        )

    if not np.all(np.isfinite(standardized_signal)):
        raise AudioValidationError(
            "Standardized audio contains non-finite numeric values.",
            status_code=400,
        )

    # Reject silence before amplitude normalization.
    raw_peak = float(np.max(np.abs(standardized_signal)))
    raw_rms = float(np.sqrt(np.mean(standardized_signal**2)))

    if raw_peak == 0.0 or raw_rms == 0.0:
        raise AudioValidationError(
            "Audio is completely silent (contains no audible signal).",
            status_code=400,
        )

    if raw_rms < MIN_RMS_THRESHOLD:
        raise AudioValidationError(
            f"Audio is effectively silent (RMS energy {raw_rms:.6f} "
            f"is below minimum threshold of {MIN_RMS_THRESHOLD}).",
            status_code=400,
        )

    # Peak normalization.
    if raw_peak > 0.0:
        standardized_signal = standardized_signal / raw_peak
        peak_normalized = 1.0
    else:
        peak_normalized = 0.0

    rms = float(np.sqrt(np.mean(standardized_signal**2)))

    metadata = AudioMetadata(
        filename=filename,
        original_sample_rate=int(orig_sr),
        target_sample_rate=TARGET_SAMPLE_RATE,
        original_channels=orig_channels,
        duration_seconds=round(duration, 3),
        num_samples=int(len(standardized_signal)),
        peak_amplitude=round(peak_normalized, 3),
        rms_energy=round(rms, 4),
    )

    return PreprocessedAudio(
        signal=standardized_signal,
        sample_rate=TARGET_SAMPLE_RATE,
        metadata=metadata,
    )


# Spectrogram feature extraction settings used by the CNN.
N_FFT = 512
HOP_LENGTH = 160
SPECTROGRAM_FRAMES = 128
SPECTROGRAM_BINS = 128
SPECTROGRAM_TOP_DB = 80.0


def extract_spectrogram(
    signal: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_frames: int = SPECTROGRAM_FRAMES,
    target_bins: int = SPECTROGRAM_BINS,
    top_db: float = SPECTROGRAM_TOP_DB,
) -> np.ndarray:
    """Create a normalized spectrogram matching Colab training.

    Input:
        A mono waveform sampled at 16 kHz.

    Output:
        Float32 array with shape (128, 128, 1).
    """

    if signal.ndim != 1:
        raise ValueError(
            "Signal must be a 1D mono waveform, "
            f"got {signal.ndim}D array with shape {signal.shape}."
        )

    signal_float32 = signal.astype(np.float32)

    # STFT and magnitude.
    stft = librosa.stft(
        signal_float32,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    magnitude = np.abs(stft)

    # Convert magnitude to decibels relative to the maximum.
    spec_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )

    # Retain the first 128 frequency bins.
    spec_db = spec_db[:target_bins, :]

    # Match Colab's librosa.util.fix_length behavior.
    # Default padding is 0 dB, which becomes 1.0 after scaling.
    spec_resized = librosa.util.fix_length(
        spec_db,
        size=target_frames,
        axis=1,
    )

    # Normalize values to [0, 1], matching the training function.
    scaled = np.clip(
        (spec_resized + top_db) / top_db,
        0.0,
        1.0,
    )

    # Add the CNN channel dimension.
    return np.expand_dims(scaled, axis=-1).astype(np.float32)
