"""Unit and Integration Tests for Vani Kavach Audio Pipeline.

Validates audio validation, preprocessing, error handling, untrained model contract,
and FastAPI endpoints.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.detector import AudioAuthenticityDetector
from app.main import app
from app.preprocessor import (
    AudioValidationError,
    MIN_RMS_THRESHOLD,
    TARGET_SAMPLE_RATE,
    extract_spectrogram,
    preprocess_audio,
)

client = TestClient(app)


def generate_test_wav(
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Helper to synthesize a clean sine wave WAV file in memory."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)

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


def test_completely_silent_wav_rejection():
    """Test that zero-amplitude (completely silent) audio is rejected."""
    silent_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.0)
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(silent_bytes, "silent.wav")
    assert exc_info.value.status_code == 400
    assert "completely silent" in exc_info.value.message.lower()


def test_near_silent_wav_rejection():
    """Test that effectively silent audio below RMS threshold is rejected."""
    near_silent_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.0001)
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(near_silent_bytes, "near_silent.wav")
    assert exc_info.value.status_code == 400
    assert "effectively silent" in exc_info.value.message.lower()
    assert "rms energy" in exc_info.value.message.lower()


def test_low_volume_valid_audio_accepted():
    """Test that ordinary low-volume speech-level audio passes without being falsely rejected."""
    # amplitude 0.02 has RMS ~0.014, safely above MIN_RMS_THRESHOLD (0.001)
    low_vol_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.02)
    result = preprocess_audio(low_vol_bytes, "quiet_speech.wav")
    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.metadata.peak_amplitude == 1.0
    assert result.metadata.rms_energy > 0.0
    assert np.all(np.isfinite(result.signal))


def test_non_finite_audio_rejection():
    """Test that audio containing non-finite values (NaN/Inf) is safely rejected."""
    nan_signal = np.array([np.nan] * 16000, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, nan_signal, 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(AudioValidationError) as exc_info:
        preprocess_audio(buffer.getvalue(), "nan_audio.wav")
    assert exc_info.value.status_code == 400
    assert "non-finite" in exc_info.value.message.lower()


# ---------------------------------------------------------------------------
# Unit Tests: Model Contract & Ethical Invariant
# ---------------------------------------------------------------------------


def test_untrained_detector_contract():
    """Verify that detector strictly adheres to untrained contract without fake scores when model_path=None."""
    detector = AudioAuthenticityDetector(model_path=None)
    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")

    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "untrained"
    assert analysis["message"] == "Model not trained — authenticity analysis unavailable"
    assert analysis["authenticity_score"] is None
    assert analysis["verdict"] is None
    assert analysis["confidence"] is None
    assert "ASVspoof" in analysis["disclaimer"]


def test_untrained_fallback_missing_file():
    """Verify that detector falls back to untrained when model file is not found."""
    detector = AudioAuthenticityDetector(model_path="nonexistent_checkpoint.keras")
    assert detector.model_status == "untrained"
    assert detector.is_trained is False

    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "untrained"
    assert analysis["verdict"] is None


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------


def test_health_endpoint_experimental():
    """Test GET /api/v1/health dynamically reports experimental status when CNN is active."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "experimental"
    assert data["is_model_trained"] is True
    assert "development subset" in data["disclaimer"]
    assert "balanced" in data["disclaimer"]


def test_health_endpoint_untrained_fallback(monkeypatch):
    """Test GET /api/v1/health dynamically reports untrained status when model is disabled."""
    monkeypatch.setattr("app.main.detector_instance", AudioAuthenticityDetector(model_path=None))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "untrained"
    assert data["is_model_trained"] is False
    assert "untrained" in data["disclaimer"].lower()


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


def test_api_validate_silent_file():
    """Test POST /api/v1/audio/validate with completely silent audio returns HTTP 400."""
    silent_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.0)
    files = {"file": ("silent.wav", io.BytesIO(silent_bytes), "audio/wav")}

    response = client.post("/api/v1/audio/validate", files=files)
    assert response.status_code == 400
    assert "silent" in response.json()["detail"].lower()


def test_api_validate_near_silent_file():
    """Test POST /api/v1/audio/validate with near-silent audio returns HTTP 400."""
    near_silent_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.0001)
    files = {"file": ("near_silent.wav", io.BytesIO(near_silent_bytes), "audio/wav")}

    response = client.post("/api/v1/audio/validate", files=files)
    assert response.status_code == 400
    assert "effectively silent" in response.json()["detail"].lower()


def test_api_analyze_silent_file():
    """Test POST /api/v1/analyze with silent audio returns HTTP 400."""
    silent_bytes = generate_test_wav(duration=1.0, sample_rate=16000, amplitude=0.0)
    files = {"file": ("silent.wav", io.BytesIO(silent_bytes), "audio/wav")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    assert "silent" in response.json()["detail"].lower()


def test_api_analyze_untrained_status(monkeypatch):
    """Test POST /api/v1/analyze emits valid metadata and strict untrained status when model is disabled."""
    monkeypatch.setattr("app.main.detector_instance", AudioAuthenticityDetector(model_path=None))
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


def test_api_analyze_experimental_status():
    """Test POST /api/v1/analyze emits experimental analysis when default CNN is active."""
    wav_bytes = generate_test_wav(duration=1.2, sample_rate=16000)
    files = {"file": ("call_sample.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["audio_validation"]["is_valid"] is True
    analysis = data["authenticity_analysis"]
    assert analysis["model_status"] == "experimental"
    assert analysis["verdict"] in ["bonafide", "spoof"]
    assert analysis["prediction"] in ["bonafide", "spoof"]
    assert 0.0 <= analysis["uncalibrated_spoof_probability"] <= 1.0
    assert analysis["confidence"] is None
    assert "balanced development subset" in analysis["disclaimer"]


def test_web_index_page_serves_redesigned_frontend():
    """Test that the root page serves the redesigned static homepage."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Vani Kavach | Verify the Signal</title>" in response.text
    assert "Every Voice Has a Signal." in response.text
    assert 'href="premium.css"' in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/style.css",
        "/premium.css",
        "/shared.js",
        "/script.js",
    ],
)
def test_frontend_assets_are_served(path):
    """Test that all assets referenced by the frontend are available."""
    response = client.get(path)
    assert response.status_code == 200
    assert response.content


@pytest.mark.parametrize(
    ("path", "expected_text"),
    [
        ("/about.html", "About Vani Kavach"),
        ("/analyze.html", "Analyze Audio | Vani Kavach"),
        ("/login.html", "Login | Vani Kavach"),
        ("/signup.html", "Create Account | Vani Kavach"),
        ("/forgot-password.html", "Forgot Password | Vani Kavach"),
    ],
)
def test_frontend_pages_are_served(path, expected_text):
    """Test that each redesigned frontend page is available at its linked URL."""
    response = client.get(path)
    assert response.status_code == 200
    assert expected_text in response.text


# ---------------------------------------------------------------------------
# Unit Tests: Spectrogram Feature Extraction (Stage 1 CNN Preprocessing)
# ---------------------------------------------------------------------------


def test_extract_spectrogram_output_shape_and_dtype():
    """Verify that spectrogram extraction yields strictly (128, 128, 1) float32."""
    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    spec = extract_spectrogram(preprocessed.signal)

    assert spec.shape == (128, 128, 1)
    assert spec.dtype == np.float32


def test_extract_spectrogram_finite_values_and_normalized_range():
    """Verify that all spectrogram values are finite and strictly bounded within [0.0, 1.0]."""
    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000, channels=1, amplitude=0.8)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    spec = extract_spectrogram(preprocessed.signal)

    assert np.all(np.isfinite(spec))
    assert float(spec.min()) >= 0.0
    assert float(spec.max()) <= 1.0
    # Peak bin in dB relative to max is 0 dB, which maps to (0 + 80) / 80 = 1.0
    assert np.isclose(float(spec.max()), 1.0, atol=1e-4)


def test_extract_spectrogram_short_audio_padding():
    """Verify that short audio (< 128 frames, e.g. 0.5s) is padded to 128 frames and stays bounded."""
    audio_bytes = generate_test_wav(duration=0.5, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "short_0.5s.wav")
    spec = extract_spectrogram(preprocessed.signal)

    assert spec.shape == (128, 128, 1)
    assert spec.dtype == np.float32
    assert np.all(np.isfinite(spec))
    assert float(spec.min()) >= 0.0
    assert float(spec.max()) <= 1.0
    # Right-padded frames should be 1.0: librosa.util.fix_length pads with
    # 0 dB, which maps to (0 + 80) / 80 = 1.0 after normalization.
    assert np.all(spec[:, -1, 0] == 1.0)


def test_extract_spectrogram_long_audio_truncation():
    """Verify that longer audio (> 128 frames, e.g. 2.5s) is truncated to exactly 128 frames."""
    audio_bytes = generate_test_wav(duration=2.5, sample_rate=16000, channels=1)
    preprocessed = preprocess_audio(audio_bytes, "long_2.5s.wav")
    spec = extract_spectrogram(preprocessed.signal)

    assert spec.shape == (128, 128, 1)
    assert spec.dtype == np.float32
    assert np.all(np.isfinite(spec))
    assert float(spec.min()) >= 0.0
    assert float(spec.max()) <= 1.0


def test_extract_spectrogram_rejects_non_1d_signal():
    """Verify that passing multidimensional array raises ValueError."""
    stereo_signal = np.zeros((16000, 2), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        extract_spectrogram(stereo_signal)
    assert "1D mono" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Unit Tests: Stage 2 Experimental Inference & Mocking
# ---------------------------------------------------------------------------


def test_experimental_inference_mocked_bonafide():
    """Verify experimental inference with mocked prediction below 0.5 returns bonafide."""
    detector = AudioAuthenticityDetector(model_path=None)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.15]], dtype=np.float32)
    detector.model = mock_model
    detector.model_status = "experimental"

    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "experimental"
    assert analysis["verdict"] == "bonafide"
    assert analysis["prediction"] == "bonafide"
    assert analysis["uncalibrated_spoof_probability"] == 0.15
    assert analysis["authenticity_score"] == 0.85
    assert analysis["confidence"] is None
    assert analysis["threshold"] == 0.5
    assert "balanced development subset" in analysis["disclaimer"]
    mock_model.predict.assert_called_once()
    assert mock_model.predict.call_args[0][0].shape == (1, 128, 128, 1)


def test_experimental_inference_mocked_spoof():
    """Verify experimental inference with mocked prediction at/above 0.5 returns spoof."""
    detector = AudioAuthenticityDetector(model_path=None)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.88]], dtype=np.float32)
    detector.model = mock_model
    detector.model_status = "experimental"

    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "experimental"
    assert analysis["verdict"] == "spoof"
    assert analysis["prediction"] == "spoof"
    assert analysis["uncalibrated_spoof_probability"] == 0.88
    assert analysis["authenticity_score"] == 0.12
    assert analysis["confidence"] is None


def test_experimental_inference_mocked_exception_fallback():
    """Verify that runtime inference exception safely falls back to untrained response."""
    detector = AudioAuthenticityDetector(model_path=None)
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Inference tensor error")
    detector.model = mock_model
    detector.model_status = "experimental"

    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000)
    preprocessed = preprocess_audio(audio_bytes, "test.wav")
    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "untrained"
    assert analysis["verdict"] is None
    assert analysis["authenticity_score"] is None


def test_experimental_inference_real_model_smoke():
    """Smoke test running inference through the actual experimental CNN checkpoint."""
    detector = AudioAuthenticityDetector()
    assert detector.is_trained is True
    assert detector.model_status == "experimental"

    audio_bytes = generate_test_wav(duration=1.0, sample_rate=16000)
    preprocessed = preprocess_audio(audio_bytes, "smoke.wav")
    analysis = detector.analyze(preprocessed)

    assert analysis["model_status"] == "experimental"
    assert analysis["verdict"] in ["bonafide", "spoof"]
    assert 0.0 <= analysis["uncalibrated_spoof_probability"] <= 1.0
    assert analysis["confidence"] is None
    assert "balanced development subset" in analysis["disclaimer"]

