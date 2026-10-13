"""Vani Kavach - AI-Powered Voice Authenticity & Scam-Defense API.

FastAPI entrypoint providing audio validation, standardized preprocessing,
health status, and web upload interface.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

from app.detector import detector_instance
from app.preprocessor import (
    AudioValidationError,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLE_RATE,
    preprocess_audio,
)

app = FastAPI(
    title="Vani Kavach API",
    description="Voice Authenticity and Scam-Defense Baseline Pipeline",
    version="0.1.0",
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
    return {
        "status": "healthy",
        "service": "Vani Kavach - Voice Authenticity & Scam Defense",
        "version": "0.1.0",
        "python_version": sys.version.split()[0],
        "model_status": "untrained",
        "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "disclaimer": (
            "Authenticity model is currently untrained. "
            "No deepfake detection or synthetic voice scoring is active."
        ),
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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the interactive web interface for audio upload and validation."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vani Kavach - Voice Authenticity & Scam Defense</title>
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-surface: #111827;
      --bg-card: rgba(17, 24, 39, 0.7);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: #38bdf8;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-blue: #0284c7;
      --accent-cyan: #06b6d4;
      --status-warning: #f59e0b;
      --status-success: #10b981;
      --status-error: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: radial-gradient(circle at 50% -20%, #1e293b, var(--bg-primary) 70%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    .container {
      width: 100%;
      max-width: 900px;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .badge {
      display: inline-block;
      padding: 0.3rem 0.8rem;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 189, 248, 0.25);
      margin-bottom: 0.75rem;
    }
    h1 {
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1rem;
      max-width: 650px;
      margin: 0 auto;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 1rem;
      backdrop-filter: blur(12px);
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .upload-zone {
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 0.75rem;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(255, 255, 255, 0.02);
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--border-focus);
      background: rgba(56, 189, 248, 0.05);
    }
    .upload-icon {
      font-size: 2.5rem;
      margin-bottom: 0.75rem;
    }
    .btn {
      display: inline-block;
      margin-top: 1rem;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      color: #fff;
      font-weight: 600;
      padding: 0.7rem 1.5rem;
      border-radius: 0.5rem;
      border: none;
      cursor: pointer;
      transition: opacity 0.2s ease;
      font-size: 0.95rem;
    }
    .btn:hover { opacity: 0.9; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    input[type="file"] { display: none; }
    .status-banner {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 1rem 1.25rem;
      border-radius: 0.5rem;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
    }
    .status-banner.warning {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fbbf24;
    }
    .results-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 1.25rem;
    }
    @media (max-width: 680px) {
      .results-grid { grid-template-columns: 1fr; }
    }
    .meta-box {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border-subtle);
      border-radius: 0.5rem;
      padding: 1rem;
    }
    .meta-box h3 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }
    .meta-item {
      display: flex;
      justify-content: space-between;
      padding: 0.35rem 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 0.85rem;
    }
    .meta-item:last-child { border-bottom: none; }
    .meta-item span:first-child { color: var(--text-muted); }
    .meta-item span:last-child { font-weight: 600; }
    audio {
      width: 100%;
      margin-top: 1rem;
      border-radius: 0.5rem;
    }
    .tag-untrained {
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 0.3rem;
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      font-weight: 700;
      font-size: 0.75rem;
    }
    .info-footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.8rem;
      margin-top: 2rem;
    }
    .info-footer a { color: var(--accent-cyan); text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">B.Tech Engineering Portfolio • Audio Authenticity Pipeline</div>
      <h1>वाणी कवच (Vani Kavach)</h1>
      <p class="subtitle">AI Voice Authenticity & Scam-Defense Framework. Preprocessing and signal validation stage.</p>
    </header>

    <div class="status-banner warning">
      <span style="font-size: 1.25rem;">⚠️</span>
      <div>
        <strong>Ethical & Scientific Integrity Standard:</strong>
        Authenticity classifier is currently <strong>untrained</strong>. No synthetic probabilities, deepfake verdicts, or replay claims are generated.
      </div>
    </div>

    <div class="card">
      <h2 style="font-size: 1.15rem; margin-bottom: 1rem;">1. Upload Audio For Validation & Preprocessing</h2>
      <div id="dropZone" class="upload-zone">
        <div class="upload-icon">🎙️</div>
        <p style="font-weight: 600; margin-bottom: 0.25rem;">Drop audio file here, or click to browse</p>
        <p style="font-size: 0.85rem; color: var(--text-muted);">Supported: WAV, MP3, FLAC, OGG (0.5s to 120s, max 15MB)</p>
        <input type="file" id="audioFileInput" accept=".wav,.mp3,.flac,.ogg,.m4a">
      </div>
      <div id="fileSelectedInfo" style="display: none; margin-top: 1rem; font-size: 0.9rem;">
        Selected: <strong id="selectedFileName"></strong> (<span id="selectedFileSize"></span>)
        <audio id="audioPreview" controls style="display: none;"></audio>
        <div>
          <button id="uploadBtn" class="btn">Validate & Process Audio</button>
        </div>
      </div>
      <div id="uploadError" style="display: none; margin-top: 1rem; color: var(--status-error); font-size: 0.9rem;"></div>
    </div>

    <div id="resultsCard" class="card" style="display: none;">
      <h2 style="font-size: 1.15rem; margin-bottom: 0.5rem;">2. Processing Pipeline Output</h2>
      <div class="results-grid">
        <div class="meta-box">
          <h3>Signal Standardization</h3>
          <div class="meta-item"><span>Target Sample Rate</span><span id="metaTargetSr">16,000 Hz</span></div>
          <div class="meta-item"><span>Channels (Mixdown)</span><span>Mono (1 channel)</span></div>
          <div class="meta-item"><span>Original Sample Rate</span><span id="metaOrigSr">-</span></div>
          <div class="meta-item"><span>Original Channels</span><span id="metaOrigCh">-</span></div>
          <div class="meta-item"><span>Duration</span><span id="metaDuration">-</span></div>
          <div class="meta-item"><span>Sample Count</span><span id="metaSamples">-</span></div>
          <div class="meta-item"><span>RMS Energy</span><span id="metaRms">-</span></div>
        </div>
        <div class="meta-box">
          <h3>Authenticity Model Status</h3>
          <div style="margin-top: 0.5rem; margin-bottom: 0.75rem;">
            <span class="tag-untrained" id="modelStatusTag">UNTRAINED</span>
          </div>
          <p id="modelMessage" style="font-size: 0.85rem; color: #fbbf24; margin-bottom: 0.5rem;"></p>
          <p style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.4;">
            As per academic benchmarks (ASVspoof), synthetic voice classification requires trained feature extractors (LFCC/CQCC) and evaluated backbones (AASIST/RawNet2). No fabricated scores are emitted.
          </p>
        </div>
      </div>
    </div>

    <footer class="info-footer">
      Vani Kavach Project • Python 3.13.7 • FastAPI • Built with rigorous input sanitization & sound engineering
    </footer>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('audioFileInput');
    const fileInfo = document.getElementById('fileSelectedInfo');
    const fileNameEl = document.getElementById('selectedFileName');
    const fileSizeEl = document.getElementById('selectedFileSize');
    const uploadBtn = document.getElementById('uploadBtn');
    const preview = document.getElementById('audioPreview');
    const resultsCard = document.getElementById('resultsCard');
    const uploadError = document.getElementById('uploadError');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelected();
      }
    });
    fileInput.addEventListener('change', handleFileSelected);

    function handleFileSelected() {
      const file = fileInput.files[0];
      if (!file) return;
      fileNameEl.textContent = file.name;
      fileSizeEl.textContent = (file.size / 1024).toFixed(1) + ' KB';
      fileInfo.style.display = 'block';
      uploadError.style.display = 'none';
      resultsCard.style.display = 'none';

      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    }

    uploadBtn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) return;

      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Processing...';
      uploadError.style.display = 'none';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('/api/v1/analyze', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Validation failed');
        }

        const meta = data.audio_validation.metadata;
        document.getElementById('metaOrigSr').textContent = meta.original_sample_rate + ' Hz';
        document.getElementById('metaOrigCh').textContent = meta.original_channels === 1 ? 'Mono' : 'Stereo (' + meta.original_channels + ')';
        document.getElementById('metaDuration').textContent = meta.duration_seconds + ' s';
        document.getElementById('metaSamples').textContent = meta.num_samples.toLocaleString();
        document.getElementById('metaRms').textContent = meta.rms_energy;

        document.getElementById('modelStatusTag').textContent = data.authenticity_analysis.model_status.toUpperCase();
        document.getElementById('modelMessage').textContent = data.authenticity_analysis.message;

        resultsCard.style.display = 'block';
      } catch (err) {
        uploadError.textContent = 'Error: ' + err.message;
        uploadError.style.display = 'block';
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Validate & Process Audio';
      }
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
