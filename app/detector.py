"""Voice Authenticity Detection Pipeline for Vani Kavach.

Provides the detector contract and baseline interface for ASVspoof-style
acoustic authenticity classification.

Supports optional CNN spectrogram model loading for experimental analysis.
When weights are not loaded or unavailable, explicitly maintains an
'untrained' status with no fabricated confidence scores.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Configure Keras to use PyTorch backend before importing Keras
os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import keras
except ImportError:
    keras = None

from app.preprocessor import PreprocessedAudio, extract_spectrogram

logger = logging.getLogger(__name__)

# Default model path relative to project root
DEFAULT_MODEL_PATH = Path("model/cnn_spectrogram_experimental.keras")
CLASSIFICATION_THRESHOLD = 0.5
EXPERIMENTAL_DISCLAIMER = (
    "EXPERIMENTAL PROTOTYPE: This CNN was evaluated only on a selected balanced "
    "development subset (400 recordings, 93.25% development accuracy). It is an "
    "uncalibrated acoustic prototype and is not validated for real-world scam or deepfake detection."
)
UNTRAINED_DISCLAIMER = (
    "Deepfake detection, replay-attack detection, and speaker authenticity "
    "require an evaluated model trained on standard benchmarks (e.g., ASVspoof). "
    "No synthetic probabilities or verdicts are generated without trained weights."
)


class AudioAuthenticityDetector:
    """Acoustic authenticity detector with optional CNN inference and strict untrained fallback."""

    def __init__(self, model_path: Optional[Union[str, Path]] = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = None
        self.model_status = "untrained"
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the CNN model if a path is provided and Keras is available."""
        if self.model_path is None:
            self.model_status = "untrained"
            return

        resolved_path = Path(self.model_path)
        if not resolved_path.is_absolute():
            # Resolve relative to project root (parent of app/)
            project_root = Path(__file__).resolve().parent.parent
            resolved_path = project_root / resolved_path

        if not resolved_path.exists():
            logger.warning("Model file not found at %s. Falling back to untrained mode.", resolved_path)
            self.model_status = "untrained"
            return

        if keras is None:
            logger.warning("Keras package is not installed. Falling back to untrained mode.")
            self.model_status = "untrained"
            return

        try:
            # compile=False avoids torch.compile/Triton dependency on Windows
            self.model = keras.models.load_model(resolved_path, compile=False)
            self.model_status = "experimental"
            logger.info("Loaded experimental CNN model from %s", resolved_path)
        except Exception as exc:
            logger.warning(
                "Failed to load model from %s: %s. Falling back to untrained mode.",
                resolved_path,
                exc,
            )
            self.model = None
            self.model_status = "untrained"

    @property
    def is_trained(self) -> bool:
        """Backward-compatible flag indicating if an evaluated model is active."""
        return self.model is not None

    def analyze(self, preprocessed_audio: PreprocessedAudio) -> Dict[str, Any]:
        """Analyze preprocessed audio signal.

        Returns experimental classification results when a model is loaded,
        or an explicit untrained disclaimer when weights are unavailable.
        """
        if self.model is None or self.model_status != "experimental":
            return self._untrained_response(preprocessed_audio)

        try:
            # 1. Extract standardized (128, 128, 1) spectrogram
            spec = extract_spectrogram(preprocessed_audio.signal)

            # 2. Add batch dimension: shape (1, 128, 128, 1)
            batch = np.expand_dims(spec, axis=0)

            # 3. Model forward pass (verbose=0 suppresses stdout noise)
            raw_output = self.model.predict(batch, verbose=0)
            raw_spoof_prob = float(raw_output[0, 0])

            # 4. Classify using 0.5 threshold (Class 0 = bonafide, Class 1 = spoof)
            verdict = "spoof" if raw_spoof_prob >= CLASSIFICATION_THRESHOLD else "bonafide"

            return {
                "model_status": "experimental",
                "message": "Experimental CNN spectrogram analysis completed",
                "verdict": verdict,
                "prediction": verdict,
                "uncalibrated_spoof_probability": round(raw_spoof_prob, 4),
                "authenticity_score": round(1.0 - raw_spoof_prob, 4),
                "confidence": None,
                "threshold": CLASSIFICATION_THRESHOLD,
                "disclaimer": EXPERIMENTAL_DISCLAIMER,
                "audio_metadata": preprocessed_audio.metadata.to_dict(),
            }
        except Exception as exc:
            logger.error("Inference failed: %s. Falling back to untrained response.", exc)
            return self._untrained_response(preprocessed_audio)

    def _untrained_response(self, preprocessed_audio: PreprocessedAudio) -> Dict[str, Any]:
        """Return standardized untrained response."""
        return {
            "model_status": "untrained",
            "message": "Model not trained — authenticity analysis unavailable",
            "authenticity_score": None,
            "verdict": None,
            "confidence": None,
            "disclaimer": UNTRAINED_DISCLAIMER,
            "audio_metadata": preprocessed_audio.metadata.to_dict(),
        }


# Global detector instance for API endpoints
detector_instance = AudioAuthenticityDetector()

