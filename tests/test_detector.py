"""Unit and Integration Tests for Vani Kavach Audio Pipeline.

Validates audio validation, preprocessing, error handling, untrained model contract,
and FastAPI endpoints.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.detector import AudioAuthenticityDetector
from app.main import app
from app.preprocessor import (
    AudioValidationError,
    TARGET_SAMPLE_RATE,
    preprocess_audio,
)

client = TestClient(app)


def generate_test_wav(
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
) -> bytes:
    """Helper to synthesize a clean sine wave WAV file in memory."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)

    if channels == 2:
        # Interleave to create stereo
        signal = np.column_stack((signal, signal))

    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Unit Tests: Audio Preprocessor & Validation
# ---------------------------------------------------------------------------


def test_valid_mono_wav_preprocessing():
    """Test standard 1.0s 16kHz mono audio processing."""
    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1)
    result = preprocess_audio(audio_bytes, "clean_sample.wav")

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.signal.ndim == 1
    assert result.metadata.original_channels == 1
    assert result.metadata.original_sample_rate == 16000
    assert abs(result.metadata.duration_seconds - 1.0) < 0.05
    assert result.metadata.peak_amplitude == 1.0
    assert result.metadata.rms_energy > 0.0


def test_valid_stereo_resampling_to_mono_16k():
    """Test 44.1kHz stereo audio mixdown to mono and resampling to 16kHz."""
    audio_bytes = generate_test_wav(duration=1.5, sample_rate=44100, channels=2)
    result = preprocess_audio(audio_bytes, "stereo_44k.wav")

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.signal.ndim == 1  # Successfully mixed down to mono
    assert result.metadata.original_channels == 2
    assert result.metadata.original_sample_rate == 44100
    assert abs(result.metadata.duration_seconds - 1.5) < 0.05
    expected_samples = int(1.5 * TARGET_SAMPLE_RATE)
    assert abs(len(result.signal) - expected_samples) <= 10


def test_empty_audio_rejection():
    """Test that zero-byte files are rejected."""
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(b"", "empty.wav")
    assert "empty" in exc_info.value.message.lower()


def test_corrupted_audio_bytes_rejection():
    """Test that random non-audio bytes are rejected."""
    corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbage_data_here"
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(corrupt_bytes, "corrupt.wav")
    assert "fail" in exc_info.value.message.lower() or "corrupt" in exc_info.value.message.lower()


def test_unsupported_file_extension():
    """Test that non-audio extensions are rejected."""
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(b"some content", "document.pdf")
    assert "unsupported" in exc_info.value.message.lower()


def test_short_audio_rejection():
    """Test that audio shorter than 0.5s is rejected."""
    short_bytes = generate_test_wav(duration=0.2, sample_rate=16000, channels=1)
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(short_bytes, "too_short.wav")
    assert "shorter than minimum" in exc_info.value.message.lower()


# ---------------------------------------------------------------------------
# Unit Tests: Model Contract & Ethical Invariant
# ---------------------------------------------------------------------------


def test_untrained_detector_contract():
    """Verify that detector strictly adheres to untrained contract without fake scores."""
    detector = AudioAuthenticityDetector()
    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")

    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "untrained"
    assert analysis["message"] == "Model not trained — authenticity analysis unavailable"
    assert analysis["authenticity_score"] is None
    assert analysis["verdict"] is None
    assert analysis["confidence"] is None
    assert "ASVspoof" in analysis["disclaimer"]


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """Test GET /api/v1/health returns healthy status and untrained model indicator."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "untrained"
    assert ".wav" in data["supported_formats"]


def test_api_validate_valid_file():
    """Test POST /api/v1/audio/validate with valid audio upload."""
    wav_bytes = generate_test_wav(duration=1.0, sample_rate=16000)
    files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/api/v1/audio/validate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "validated"
    assert data["preprocessed_specs"]["sample_rate_hz"] == 16000
    assert data["preprocessed_specs"]["channels"] == 1


def test_api_validate_corrupt_file():
    """Test POST /api/v1/audio/validate with corrupted content returns HTTP 400."""
    files = {"file": ("bad.wav", io.BytesIO(b"not a real audio stream"), "audio/wav")}

    response = client.post("/api/v1/audio/validate", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_api_analyze_untrained_status():
    """Test POST /api/v1/analyze emits valid metadata and strict untrained status."""
    wav_bytes = generate_test_wav(duration=1.2, sample_rate=22050)
    files = {"file": ("call_sample.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["audio_validation"]["is_valid"] is True
    assert data["authenticity_analysis"]["model_status"] == "untrained"
    assert (
        data["authenticity_analysis"]["message"]
        == "Model not trained — authenticity analysis unavailable"
    )
    assert data["authenticity_analysis"]["authenticity_score"] is None
    assert data["contextual_risk_assessment"]["risk_tier"] == "NOT_ASSESSED"


def test_web_index_page():
    """Test that the web UI index page renders with disclaimers."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Vani Kavach" in response.text
    assert "UNTRAINED" in response.text
    assert "audioFileInput" in response.text
