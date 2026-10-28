"""Repeatable Evaluation Script for Vani Kavach Acoustic Authenticity Model.

Evaluates the experimental CNN model on:
1. The available ASVspoof-derived development subset (vani_kavach_dev_samples.npz).
2. The user's real microphone recording (Recording (2).wav).
3. Active speech windows within the microphone recording to isolate leading silence vs acoustic channel mismatch.

Outputs per-sample tables, metrics, and saves evaluation_results.json.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List

import numpy as np

# Ensure PyTorch backend is set before Keras import
os.environ.setdefault("KERAS_BACKEND", "torch")

from app.detector import CLASSIFICATION_THRESHOLD, detector_instance
from app.preprocessor import extract_spectrogram, preprocess_audio


def load_dev_samples(npz_path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load development spectrogram samples and integer labels.

    Returns:
        X: np.ndarray of shape (N, 128, 128, 1)
        y: np.ndarray of shape (N,) where 0 = bonafide, 1 = spoof
    """
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"]


def evaluate_dev_set(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Evaluate dev set spectrograms through the model forward pass."""
    if detector_instance.model is None:
        return []

    raw_probs = detector_instance.model.predict(X, verbose=0)[:, 0]
    records: List[Dict[str, Any]] = []

    for i, prob in enumerate(raw_probs):
        true_label = "spoof" if int(y[i]) == 1 else "bonafide"
        pred_label = "spoof" if prob >= threshold else "bonafide"
        prob_val = float(prob)
        records.append({
            "sample_id": f"dev_sample_{i:02d}",
            "source": "ASVspoof development subset",
            "true_label": true_label,
            "spoof_probability": round(prob_val, 4),
            "authenticity_score": round(1.0 - prob_val, 4),
            "predicted_label": pred_label,
            "correct": true_label == pred_label,
        })
    return records


def evaluate_audio_file(
    wav_path: str | pathlib.Path,
    true_label: str = "bonafide",
) -> Dict[str, Any]:
    """Run full preprocessing and detector analysis on an audio file."""
    path = pathlib.Path(wav_path)
    with open(path, "rb") as f:
        content = f.read()

    preprocessed = preprocess_audio(content, path.name)
    analysis = detector_instance.analyze(preprocessed)

    spoof_prob = analysis.get("uncalibrated_spoof_probability")
    pred_label = analysis.get("verdict")
    correct = (pred_label == true_label) if pred_label is not None else None

    return {
        "sample_id": path.name,
        "source": "Natural microphone recording",
        "true_label": true_label,
        "spoof_probability": spoof_prob,
        "authenticity_score": analysis.get("authenticity_score"),
        "predicted_label": pred_label,
        "correct": correct,
        "duration_seconds": preprocessed.metadata.duration_seconds,
        "sample_rate": preprocessed.metadata.target_sample_rate,
        "rms_energy": preprocessed.metadata.rms_energy,
    }


def evaluate_speech_windows(
    wav_path: str | pathlib.Path,
    window_frames: int = 128,
    hop_samples: int = 160,
    stride_frames: int = 64,
) -> List[Dict[str, Any]]:
    """Analyze sliding 1.28s windows across the recording to compare silence vs active speech."""
    path = pathlib.Path(wav_path)
    with open(path, "rb") as f:
        content = f.read()

    preprocessed = preprocess_audio(content, path.name)
    sig = preprocessed.signal
    sr = preprocessed.sample_rate

    win_samples = window_frames * hop_samples
    stride_samples = stride_frames * hop_samples
    window_records: List[Dict[str, Any]] = []

    for start in range(0, len(sig) - win_samples + 1, stride_samples):
        chunk = sig[start : start + win_samples]
        spec = extract_spectrogram(chunk)
        batch = np.expand_dims(spec, axis=0)
        prob = float(detector_instance.model.predict(batch, verbose=0)[0, 0])
        rms = float(np.sqrt(np.mean(chunk**2)))
        t_start = start / sr
        t_end = (start + win_samples) / sr
        pred_label = "spoof" if prob >= CLASSIFICATION_THRESHOLD else "bonafide"

        window_records.append({
            "window_range": f"{t_start:.2f}s - {t_end:.2f}s",
            "rms_energy": round(rms, 4),
            "spoof_probability": round(prob, 4),
            "predicted_label": pred_label,
        })
    return window_records


def run_full_evaluation() -> Dict[str, Any]:
    project_root = pathlib.Path(__file__).resolve().parent
    npz_path = project_root / "vani_kavach_dev_samples.npz"
    wav_path = project_root / "Recording (2).wav"

    results: Dict[str, Any] = {
        "model_status": detector_instance.model_status,
        "threshold": CLASSIFICATION_THRESHOLD,
        "dev_samples": [],
        "user_recording": None,
        "user_recording_windows": [],
        "summary": {},
    }

    print("=" * 80)
    print("VANI KAVACH — MODEL EVALUATION & FALSE POSITIVE INVESTIGATION")
    print("=" * 80)

    # 1. Dev set evaluation
    if npz_path.exists():
        X, y = load_dev_samples(npz_path)
        dev_records = evaluate_dev_set(X, y)
        results["dev_samples"] = dev_records

        dev_correct = sum(1 for r in dev_records if r["correct"])
        dev_total = len(dev_records)
        dev_bonafide = [r for r in dev_records if r["true_label"] == "bonafide"]
        dev_spoof = [r for r in dev_records if r["true_label"] == "spoof"]

        # Detailed confusion matrix and classification metrics
        tp = sum(1 for r in dev_records if r["true_label"] == "spoof" and r["predicted_label"] == "spoof")
        tn = sum(1 for r in dev_records if r["true_label"] == "bonafide" and r["predicted_label"] == "bonafide")
        fp = sum(1 for r in dev_records if r["true_label"] == "bonafide" and r["predicted_label"] == "spoof")
        fn = sum(1 for r in dev_records if r["true_label"] == "spoof" and r["predicted_label"] == "bonafide")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"\n[1] ASVspoof Development Subset (N = {dev_total}):")
        print(f"    Threshold:         {CLASSIFICATION_THRESHOLD}")
        print(f"    Overall Accuracy:  {dev_correct}/{dev_total} ({dev_correct/dev_total:.1%})")
        print(f"    Confusion Matrix:  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
        print(f"    Precision (Spoof): {precision:.4f}")
        print(f"    Recall (Spoof):    {recall:.4f}")
        print(f"    F1 Score (Spoof):  {f1:.4f}")
        print(f"    False Pos Rate:    {fpr:.4f}")
        print(f"    False Neg Rate:    {fnr:.4f}")

        print("\n    Per-Sample Breakdown:")
        print(f"    {'Sample ID':<14} | {'True Label':<10} | {'Spoof Prob':<10} | {'Pred Label':<10} | {'Result':<8}")
        print("    " + "-" * 62)
        for r in dev_records:
            status_str = "PASS" if r["correct"] else "FAIL"
            print(f"    {r['sample_id']:<14} | {r['true_label']:<10} | {r['spoof_probability']:<10.4f} | {r['predicted_label']:<10} | {status_str:<8}")
    else:
        print("\n[1] Development subset NPZ not found.")

    # 2. Natural microphone recording evaluation
    if wav_path.exists():
        wav_record = evaluate_audio_file(wav_path, true_label="bonafide")
        results["user_recording"] = wav_record

        print(f"\n[2] Real Microphone Recording ({wav_record['sample_id']}):")
        print(f"    Duration:        {wav_record['duration_seconds']:.2f}s")
        print(f"    True Label:      {wav_record['true_label']}")
        print(f"    Predicted Label: {wav_record['predicted_label']}")
        print(f"    Spoof Prob:      {wav_record['spoof_probability']:.4f}")
        print(f"    Authenticity:    {wav_record['authenticity_score']:.4f}")
        print(f"    Correct:         {wav_record['correct']}")
        if not wav_record["correct"]:
            print("    --> FALSE POSITIVE CONFIRMED: Genuine microphone voice classified as SPOOF.")

        # 3. Windowed analysis of the recording
        windows = evaluate_speech_windows(wav_path)
        results["user_recording_windows"] = windows
        print("\n[3] Sliding 1.28s Window Analysis (Silence vs Active Speech):")
        print(f"    {'Window Range':<16} | {'RMS Energy':<10} | {'Spoof Prob':<10} | {'Prediction':<10}")
        print("    " + "-" * 54)
        for w in windows:
            print(f"    {w['window_range']:<16} | {w['rms_energy']:<10.4f} | {w['spoof_probability']:<10.4f} | {w['predicted_label']:<10}")
    else:
        print("\n[2] Microphone recording not found.")

    # Save summary
    results["summary"] = {
        "dev_samples_count": len(results["dev_samples"]),
        "dev_accuracy": (
            sum(1 for r in results["dev_samples"] if r["correct"]) / len(results["dev_samples"])
            if results["dev_samples"]
            else None
        ),
        "confusion_matrix": {
            "tn": tn if npz_path.exists() else None,
            "fp": fp if npz_path.exists() else None,
            "fn": fn if npz_path.exists() else None,
            "tp": tp if npz_path.exists() else None,
        },
        "metrics": {
            "precision": round(precision, 4) if npz_path.exists() else None,
            "recall": round(recall, 4) if npz_path.exists() else None,
            "f1": round(f1, 4) if npz_path.exists() else None,
            "fpr": round(fpr, 4) if npz_path.exists() else None,
            "fnr": round(fnr, 4) if npz_path.exists() else None,
        },
        "user_recording_evaluated": results["user_recording"] is not None,
        "user_recording_false_positive": (
            results["user_recording"] is not None and not results["user_recording"]["correct"]
        ),
    }

    out_file = project_root / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {out_file.name}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_full_evaluation()
