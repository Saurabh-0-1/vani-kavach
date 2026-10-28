"""Vani Kavach - AI-Powered Voice Authenticity & Scam-Defense API.

FastAPI entrypoint providing audio validation, standardized preprocessing,
health status, and the web frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.detector import detector_instance
from app.preprocessor import (
    AudioValidationError,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLE_RATE,
    preprocess_audio,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Vani Kavach API",
    description="Voice Authenticity and Scam-Defense Baseline Pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(AudioValidationError)
async def audio_validation_exception_handler(_request, exc: AudioValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": "AudioValidationError"},
    )


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """Service health and environment status endpoint."""
    model_status = detector_instance.model_status
    if model_status == "experimental":
        disclaimer = (
            "Authenticity model is an experimental CNN trained on a balanced 400-recording development subset. "
            "It is an uncalibrated prototype and is not validated for real-world scam or deepfake detection."
        )
    else:
        disclaimer = (
            "Authenticity model is currently untrained. "
            "No deepfake detection or synthetic voice scoring is active."
        )

    return {
        "status": "healthy",
        "service": "Vani Kavach - Voice Authenticity & Scam Defense",
        "version": "0.1.0",
        "python_version": sys.version.split()[0],
        "model_status": model_status,
        "is_model_trained": detector_instance.is_trained,
        "features": {
            "vad_speech_activity_detection": True,
            "polyphase_16k_standardization": True,
            "cnn_spectrogram_inference": detector_instance.is_trained,
        },
        "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "disclaimer": disclaimer,
    }


@app.post("/api/v1/audio/validate")
async def validate_audio_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Validate and preprocess an audio file, returning standardized metadata."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    content = await file.read()
    try:
        preprocessed = preprocess_audio(content, file.filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return {
        "status": "validated",
        "filename": file.filename,
        "metadata": preprocessed.metadata.to_dict(),
        "preprocessed_specs": {
            "channels": 1,
            "sample_rate_hz": preprocessed.sample_rate,
            "sample_count": len(preprocessed.signal),
            "normalized": True,
        },
    }


@app.post("/api/v1/analyze")
async def analyze_audio_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Validate audio and run through the detector pipeline.

    Explicitly returns an untrained status with no fabricated probability scores.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    content = await file.read()
    try:
        preprocessed = preprocess_audio(content, file.filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    detection_result = detector_instance.analyze(preprocessed)

    return {
        "status": "success",
        "audio_validation": {
            "is_valid": True,
            "metadata": preprocessed.metadata.to_dict(),
        },
        "authenticity_analysis": detection_result,
        "contextual_risk_assessment": {
            "status": "decoupled",
            "message": "Contextual risk engine is decoupled from audio authenticity. No call metadata supplied.",
            "risk_tier": "NOT_ASSESSED",
        },
    }


@app.get("/")
async def index() -> FileResponse:
    """Serve the redesigned frontend homepage."""
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


def _frontend_page(filename: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / filename, media_type="text/html")


@app.get("/about.html")
async def about_page() -> FileResponse:
    return _frontend_page("about.html")


@app.get("/analyze.html")
async def analyze_page() -> FileResponse:
    return _frontend_page("analyze.html")


@app.get("/login.html")
async def login_page() -> FileResponse:
    return _frontend_page("login.html")


@app.get("/signup.html")
async def signup_page() -> FileResponse:
    return _frontend_page("signup.html")


@app.get("/forgot-password.html")
async def forgot_password_page() -> FileResponse:
    return _frontend_page("forgot-password.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
