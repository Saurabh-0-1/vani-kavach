"""Audio Validation and Preprocessing Module for Vani Kavach.

Provides strict audio validation, decoding, channel unification (mono),
sample-rate standardization (16 kHz), and amplitude normalization.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.signal
import soundfile as sf


class AudioValidationError(Exception):
    """Raised when an uploaded file fails validation or cannot be decoded."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Audio engineering standards
TARGET_SAMPLE_RATE: int = 16000  # 16 kHz: Standard for ASVspoof benchmarks and speech models
MIN_DURATION_SECONDS: float = 0.5  # Need at least 0.5s for meaningful acoustic framing
MAX_DURATION_SECONDS: float = 120.0  # Cap at 2 minutes for baseline processing
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB max file upload limit
SUPPORTED_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MIN_RMS_THRESHOLD: float = 0.001  # -60 dBFS: Minimum acoustic energy to distinguish speech/audio from silence
MIN_RMS_ENERGY: float = MIN_RMS_THRESHOLD


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedAudio:
    signal: np.ndarray
    sample_rate: int
    metadata: AudioMetadata


def validate_file_envelope(content: bytes, filename: str) -> None:
    """Validate file size and extension before attempting decoding."""
    if not content or len(content) == 0:
        raise AudioValidationError("Audio file is empty (0 bytes).", status_code=400)

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
            f"Unsupported audio format '{ext}'. Supported formats: {supported}.",
            status_code=400,
        )


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D signal to target sample rate using polyphase filtering."""
    if orig_sr == target_sr:
        return signal.astype(np.float32)

    common_gcd = gcd(orig_sr, target_sr)
    up = target_sr // common_gcd
    down = orig_sr // common_gcd
    resampled = scipy.signal.resample_poly(signal, up, down)
    return resampled.astype(np.float32)


def preprocess_audio(content: bytes, filename: str) -> PreprocessedAudio:
    """Validate, decode, and preprocess raw audio bytes.

    Steps:
    1. Size and extension checks.
    2. Soundfile decoding (handles WAV, FLAC, OGG, and supported MP3 containers).
    3. Structural & numerical integrity checks (zero samples, non-finite values).
    4. Duration boundary checks [0.5s - 120s].
    5. Stereo/multichannel to mono mixdown.
    6. Resampling to 16 kHz.
    7. Silent-audio and energy floor validation (before peak normalization).
    8. Peak amplitude normalization to [-1.0, 1.0].
    9. Acoustic metadata extraction (duration, peak, RMS energy).
    """
    validate_file_envelope(content, filename)

    try:
        data, orig_sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioValidationError(
            f"Failed to decode audio file. The file may be corrupt or malformed: {exc}",
            status_code=400,
        ) from exc

    if data.size == 0:
        raise AudioValidationError("Decoded audio stream contains zero samples.", status_code=400)

    if not np.all(np.isfinite(data)):
        raise AudioValidationError(
            "Audio contains non-finite or invalid numeric values (NaN or Inf).",
            status_code=400,
        )

    # Determine original channels and mix down to mono
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
        raise AudioValidationError("Decoded audio stream contains zero samples.", status_code=400)

    duration = float(len(mono_signal) / orig_sr)
    if duration < MIN_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) is shorter than minimum required ({MIN_DURATION_SECONDS}s).",
            status_code=400,
        )

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) exceeds maximum allowed limit ({MAX_DURATION_SECONDS}s).",
            status_code=400,
        )

    # Resample to target sample rate (16 kHz)
    standardized_signal = resample_signal(mono_signal, orig_sr, TARGET_SAMPLE_RATE)

    if standardized_signal.size == 0:
        raise AudioValidationError("Standardized audio signal contains zero samples.", status_code=400)

    if not np.all(np.isfinite(standardized_signal)):
        raise AudioValidationError(
            "Standardized audio contains non-finite numeric values.",
            status_code=400,
        )

    # Silent-audio validation (MUST be performed before peak normalization)
    raw_peak = float(np.max(np.abs(standardized_signal)))
    raw_rms = float(np.sqrt(np.mean(standardized_signal**2)))

    if raw_peak == 0.0 or raw_rms == 0.0:
        raise AudioValidationError(
            "Audio is completely silent (contains no audible signal).",
            status_code=400,
        )

    if raw_rms < MIN_RMS_THRESHOLD:
        raise AudioValidationError(
            f"Audio is effectively silent (RMS energy {raw_rms:.6f} is below minimum threshold of {MIN_RMS_THRESHOLD}).",
            status_code=400,
        )

    # Peak normalization
    if raw_peak > 0.0:
        standardized_signal = standardized_signal / raw_peak
        peak_normalized = 1.0
    else:
        peak_normalized = 0.0

    # Calculate RMS energy of normalized signal
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
