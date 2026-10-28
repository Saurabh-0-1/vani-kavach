# Vani Kavach — AI-Powered Voice Authenticity & Scam Defense

Vani Kavach is an acoustic authenticity and scam-defense prototype designed to inspect audio streams for synthetic speech and deepfake manipulation artifacts.

---

## System Architecture

- **Backend**: FastAPI (`app/main.py`)
  - Endpoints:
    - `GET /api/v1/health`: Environment and service status.
    - `POST /api/v1/audio/validate`: Audio structure, format, clipping, and Voice Activity Detection (VAD) validation.
    - `POST /api/v1/analyze`: Full acoustic authenticity analysis pipeline.
    - `GET /`, `GET /analyze.html`, `GET /about.html`: Static web frontend.
- **Audio Preprocessing**: (`app/preprocessor.py`)
  - Voice Activity Detection (VAD) with adaptive noise-floor estimation and hangover smoothing.
  - Active speech window selection (`select_best_speech_window`) to eliminate leading/trailing silence bias.
  - Standardized polyphase resampling to 16 kHz mono.
  - Silent audio and energy floor checks (rejects pure silence and effectively silent files).
  - Peak amplitude normalization and clipping ratio detection.
- **Detector Pipeline**: (`app/detector.py`)
  - Active Model: Experimental 3-layer CNN spectrogram classifier (`model/cnn_spectrogram_experimental.keras`).
  - Output Categorization: Returns honest verdicts (`likely_genuine`, `potential_spoof`, or `inconclusive`) based on speech presence and acoustic score.
  - Strict fallback to uncalibrated/untrained disclaimer when weights are unverified or disabled.
- **Frontend**: Responsive web workspace (`frontend/`)
  - Drag-and-drop audio file upload.
  - Live microphone recording via browser `MediaRecorder` API.
  - Clear visual badges (`LIKELY GENUINE`, `POTENTIAL SPOOF`, `INCONCLUSIVE`).
  - Signal validation breakdown and honest model limitation disclosures.

---

## Local Startup & Demonstration

### 1. Run with Python Virtual Environment

```bash
# Clone and enter directory
git clone https://github.com/Saurabh-0-1/vani-kavach.git
cd vani-kavach

# Activate virtual environment
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your browser to: **`http://127.0.0.1:8000/analyze.html`**

### 2. Run with Docker

```bash
docker build -t vani-kavach .
docker run -p 8000:8000 vani-kavach
```

---

## Automated Test Suite

Run the full automated test suite (49 tests covering audio validation, silence rejection, VAD, clipping detection, model contracts, API endpoints, label mapping, and evaluation reproducibility):

```bash
.\.venv\Scripts\python.exe -m pytest
```

---

## Reproducible Evaluation

To evaluate the experimental CNN against both the ASVspoof-derived development set and natural microphone recordings:

```bash
.\.venv\Scripts\python.exe evaluate_false_positives.py
```

Detailed evaluation results and confusion matrix are saved to `evaluation_results.json`.
For a technical report on label mapping, OOD sensitivity, and microphone false positive root causes, see [model/README.md](model/README.md).

---

## Current Status & Known Limitations

> **Status: DEMO READY WITH MODEL LIMITATIONS**
>
> The web application, audio preprocessor, VAD engine, FastAPI backend, and test suite are **fully functional and demo-ready**.
>
> However, the acoustic model is an **experimental prototype** trained on a small subset of ASVspoof 2019 studio data. It does not generalize to real-world consumer microphones and can produce false positives on natural human speech. It must not be used as an active security defense in production without retraining on diverse real-world audio.