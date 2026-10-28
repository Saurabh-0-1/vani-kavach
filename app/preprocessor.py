"""Audio Validation and Preprocessing Module for Vani Kavach.

Provides audio validation, decoding, channel unification (mono),
sample-rate standardization (16 kHz), amplitude normalization,
voice activity detection (VAD), speech window selection,
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

# VAD and speech parameters
MIN_SPEECH_DURATION_SECONDS = 0.3
VAD_FRAME_MS = 25.0
VAD_HOP_MS = 10.0


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
    speech_duration_seconds: float = 0.0
    speech_ratio: float = 0.0
    is_speech_detected: bool = True
    clipping_ratio: float = 0.0
    snr_db: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedAudio:
    signal: np.ndarray
    sample_rate: int
    metadata: AudioMetadata
    speech_window: np.ndarray | None = None


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


def detect_voice_activity(
    signal: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    frame_ms: float = VAD_FRAME_MS,
    hop_ms: float = VAD_HOP_MS,
    min_speech_duration_sec: float = MIN_SPEECH_DURATION_SECONDS,
) -> dict[str, Any]:
    """Detect voice activity across a 1D standardized audio waveform.

    Computes short-time frame energy, dynamic noise-floor estimation,
    energy thresholding with hangover smoothing, and speech metrics.
    """
    frame_len = int(sample_rate * (frame_ms / 1000.0))
    hop_len = int(sample_rate * (hop_ms / 1000.0))

    if len(signal) < frame_len:
        rms_val = float(np.sqrt(np.mean(signal**2))) if len(signal) > 0 else 0.0
        return {
            "is_speech_detected": rms_val >= MIN_RMS_THRESHOLD,
            "speech_duration_seconds": round(len(signal) / sample_rate, 3),
            "speech_ratio": 1.0 if rms_val >= MIN_RMS_THRESHOLD else 0.0,
            "speech_mask": np.array([True]),
            "noise_floor_rms": round(rms_val, 4),
            "snr_db": 0.0,
        }

    num_frames = (len(signal) - frame_len) // hop_len + 1
    # Frame-level RMS energy
    frame_energies = np.empty(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start : start + frame_len]
        frame_energies[i] = np.sqrt(np.mean(frame**2))

    # Adaptive noise floor estimated from bottom 20th percentile
    noise_floor = float(np.percentile(frame_energies, 20))
    min_energy = float(np.min(frame_energies))
    max_energy = float(np.max(frame_energies))

    # If the dynamic range is narrow and overall energy is solid,
    # the entire signal is audible active sound (e.g. continuous speech or tone):
    if noise_floor > 0.01 and (noise_floor / (max_energy + 1e-6)) > 0.3:
        threshold = min_energy * 0.5
    else:
        threshold = max(noise_floor * 2.0, MIN_RMS_THRESHOLD * 2.5)

    raw_speech = frame_energies >= threshold

    # Hangover smoothing: 150ms (15 frames) hangover to preserve pauses and soft syllables
    hangover_frames = int(150 / hop_ms)
    smooth_speech = raw_speech.copy()
    counter = 0
    for i in range(num_frames):
        if raw_speech[i]:
            counter = hangover_frames
        elif counter > 0:
            smooth_speech[i] = True
            counter -= 1

    speech_frames = int(np.sum(smooth_speech))
    speech_duration = round(float(speech_frames * hop_len / sample_rate), 3)
    total_duration = round(float(len(signal) / sample_rate), 3)
    speech_ratio = round(float(speech_duration / total_duration), 3) if total_duration > 0 else 0.0

    # Estimate SNR
    noise_energies = frame_energies[~smooth_speech] if np.any(~smooth_speech) else np.array([noise_floor])
    speech_energies = frame_energies[smooth_speech] if np.any(smooth_speech) else np.array([threshold])

    mean_noise = float(np.mean(noise_energies)) if len(noise_energies) > 0 else 1e-5
    mean_speech = float(np.mean(speech_energies)) if len(speech_energies) > 0 else 1e-4
    snr = round(float(20.0 * np.log10(max(mean_speech, 1e-6) / max(mean_noise, 1e-6))), 1)

    is_speech_detected = bool(speech_duration >= min_speech_duration_sec)

    return {
        "is_speech_detected": is_speech_detected,
        "speech_duration_seconds": speech_duration,
        "speech_ratio": min(1.0, speech_ratio),
        "speech_mask": smooth_speech,
        "noise_floor_rms": round(noise_floor, 4),
        "snr_db": snr,
    }


def select_best_speech_window(
    signal: np.ndarray,
    target_samples: int = 20480,  # 128 frames * 160 hop = 1.28 seconds
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Select the contiguous window of length target_samples with the most speech energy.

    Avoids blindly truncating to leading silence/ambient room noise in longer recordings.
    """
    if len(signal) <= target_samples:
        return signal

    hop_len = int(sample_rate * (VAD_HOP_MS / 1000.0))  # 160 samples (10ms)
    frame_len = int(sample_rate * (VAD_FRAME_MS / 1000.0))  # 400 samples

    num_frames = (len(signal) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return signal[:target_samples]

    frame_energies = np.empty(num_frames, dtype=np.float32)
    for i in range(num_frames):
        st = i * hop_len
        frame = signal[st : st + frame_len]
        frame_energies[i] = np.sqrt(np.mean(frame**2))

    # Number of hops in a target window (e.g. 128 frames)
    win_hops = target_samples // hop_len
    if len(frame_energies) < win_hops:
        return signal[:target_samples]

    # Find the window that maximizes total frame energy
    win_energy = np.convolve(frame_energies, np.ones(win_hops, dtype=np.float32), mode="valid")
    best_hop = int(np.argmax(win_energy))
    start_sample = best_hop * hop_len
    end_sample = start_sample + target_samples

    # Safeguard within array bounds
    if end_sample > len(signal):
        start_sample = max(0, len(signal) - target_samples)
        end_sample = start_sample + target_samples

    return signal[start_sample:end_sample]


def preprocess_audio(
    content: bytes,
    filename: str,
) -> PreprocessedAudio:
    """Validate, decode, standardize, and analyze voice activity for uploaded audio."""
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

    # Clipping detection before normalization
    clipping_ratio = float(np.mean(np.abs(standardized_signal) >= 0.999))

    # Peak normalization.
    if raw_peak > 0.0:
        standardized_signal = standardized_signal / raw_peak
        peak_normalized = 1.0
    else:
        peak_normalized = 0.0

    rms = float(np.sqrt(np.mean(standardized_signal**2)))

    # Voice Activity Detection
    vad_results = detect_voice_activity(standardized_signal, sample_rate=TARGET_SAMPLE_RATE)

    # Extract best speech window for downstream model consumption
    speech_window = select_best_speech_window(standardized_signal, target_samples=SPECTROGRAM_FRAMES * HOP_LENGTH)

    metadata = AudioMetadata(
        filename=filename,
        original_sample_rate=int(orig_sr),
        target_sample_rate=TARGET_SAMPLE_RATE,
        original_channels=orig_channels,
        duration_seconds=round(duration, 3),
        num_samples=int(len(standardized_signal)),
        peak_amplitude=round(peak_normalized, 3),
        rms_energy=round(rms, 4),
        speech_duration_seconds=vad_results["speech_duration_seconds"],
        speech_ratio=vad_results["speech_ratio"],
        is_speech_detected=vad_results["is_speech_detected"],
        clipping_ratio=round(clipping_ratio, 4),
        snr_db=vad_results["snr_db"],
    )

    return PreprocessedAudio(
        signal=standardized_signal,
        sample_rate=TARGET_SAMPLE_RATE,
        metadata=metadata,
        speech_window=speech_window,
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
    use_speech_window: bool = True,
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

    target_samples = target_frames * hop_length

    # If signal is longer than target_frames and use_speech_window is requested,
    # center on the window with the most active speech energy.
    if use_speech_window and len(signal) > target_samples:
        signal_to_process = select_best_speech_window(signal, target_samples=target_samples)
    else:
        signal_to_process = signal

    signal_float32 = signal_to_process.astype(np.float32)

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
