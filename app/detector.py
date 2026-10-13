"""Voice Authenticity Detection Pipeline for Vani Kavach.

Provides the detector contract and baseline interface for future ASVspoof
trained model integration.

IMPORTANT ARCHITECTURAL & ETHICAL PRINCIPLE:
Until an authentic model is trained, evaluated, and validated against standard
benchmarks (e.g., ASVspoof 2019/2021 LA with EER metrics), this detector
explicitly reports an 'untrained' status. It never generates fabricated
confidence scores, mock probabilities, or deepfake verdicts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.preprocessor import PreprocessedAudio


class AudioAuthenticityDetector:
    """Baseline detector interface for acoustic authenticity classification."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.is_trained = False

    def analyze(self, preprocessed_audio: PreprocessedAudio) -> Dict[str, Any]:
        """Analyze preprocessed audio signal.

        Returns an explicit untrained disclaimer until weights are integrated.
        """
        return {
            "model_status": "untrained",
            "message": "Model not trained — authenticity analysis unavailable",
            "authenticity_score": None,
            "verdict": None,
            "confidence": None,
            "disclaimer": (
                "Deepfake detection, replay-attack detection, and speaker authenticity "
                "require an evaluated model trained on standard benchmarks (e.g., ASVspoof). "
                "No synthetic probabilities or verdicts are generated without trained weights."
            ),
            "audio_metadata": preprocessed_audio.metadata.to_dict(),
        }


# Global detector instance
detector_instance = AudioAuthenticityDetector()
