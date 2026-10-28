// Vani Kavach Frontend Signal Controller
const API_BASE = (
  new URLSearchParams(window.location.search).get("api") ||
  localStorage.getItem("vaniKavachApi") ||
  (window.location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000")
).replace(/\/$/, "");

const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const browseButton = document.querySelector("#browse-button");
const recordButton = document.querySelector("#record-button");
const recordText = document.querySelector("#record-text");
const recIndicator = document.querySelector("#rec-indicator");
const recordingTimer = document.querySelector("#recording-timer");
const fileCard = document.querySelector("#file-card");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const preview = document.querySelector("#audio-preview");
const removeButton = document.querySelector("#remove-file");
const analyzeButton = document.querySelector("#analyze-button");
const statusMessage = document.querySelector("#status-message");
const results = document.querySelector("#results");

let selectedFile = null;
let objectUrl = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingInterval = null;
let recordingStartTime = 0;

const supportedExtensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];
const formatBytes = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
const setStatus = (message, type = "") => {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`;
};

const endpointError = async (response) => {
  let detail = `Request failed (${response.status})`;
  try {
    const body = await response.json();
    detail = body.detail || detail;
  } catch (_) {
    /* use HTTP status */
  }
  return detail;
};

function selectFile(file) {
  if (!file) return;
  const nameParts = file.name.split(".");
  const extension = `.${nameParts[nameParts.length - 1].toLowerCase()}`;

  // Accept custom recorded audio even if named without standard extension
  const isAudioMime = file.type && file.type.startsWith("audio/");
  if (!supportedExtensions.includes(extension) && !isAudioMime) {
    setStatus(`Unsupported format. Use ${supportedExtensions.join(", ")}.`, "error");
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    setStatus("This file is larger than the 15 MB limit.", "error");
    return;
  }

  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);

  fileName.textContent = file.name;
  fileSize.textContent = `${formatBytes(file.size)} · ${extension.slice(1).toUpperCase() || "AUDIO"}`;
  preview.src = objectUrl;

  fileCard.classList.remove("is-hidden");
  dropZone.classList.add("is-hidden");
  analyzeButton.disabled = false;
  results.classList.add("is-hidden");
  setStatus("Signal ready. Click 'Analyze recording' to proceed.");
}

function resetFile() {
  selectedFile = null;
  fileInput.value = "";
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  preview.removeAttribute("src");
  preview.load();
  fileCard.classList.add("is-hidden");
  dropZone.classList.remove("is-hidden");
  analyzeButton.disabled = true;
  results.classList.add("is-hidden");
  setStatus("");
}

// Live Microphone Recording
async function toggleRecording(event) {
  event.stopPropagation();
  if (mediaRecorder && mediaRecorder.state === "recording") {
    // Stop recording
    mediaRecorder.stop();
    clearInterval(recordingInterval);
    recordText.textContent = "Record with Microphone";
    recIndicator.style.animation = "none";
    recordingTimer.classList.add("is-hidden");
    setStatus("Microphone recording captured.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      const mimeType = mediaRecorder.mimeType || "audio/wav";
      const ext = mimeType.includes("ogg") ? ".ogg" : mimeType.includes("webm") ? ".webm" : ".wav";
      const blob = new Blob(audioChunks, { type: mimeType });
      const recordedFile = new File([blob], `live_mic_recording${ext}`, { type: mimeType });
      selectFile(recordedFile);
    };

    mediaRecorder.start(200);
    recordingStartTime = Date.now();
    recordText.textContent = "Stop Recording";
    recIndicator.style.animation = "pulse 1s infinite";
    recordingTimer.classList.remove("is-hidden");
    recordingTimer.textContent = "00:00";

    recordingInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const secs = String(elapsed % 60).padStart(2, "0");
      recordingTimer.textContent = `${mins}:${secs}`;
      if (elapsed >= 120) {
        toggleRecording(event);
      }
    }, 1000);

    setStatus("Listening... Speak clearly into your microphone.", "loading");
  } catch (err) {
    setStatus(`Microphone access failed: ${err.message || "Permission denied."}`, "error");
  }
}

function renderResults(validation, analysis) {
  const metadata = validation.metadata || analysis.audio_validation?.metadata || {};
  const auth = analysis.authenticity_analysis || {};
  const experimental = auth.model_status === "experimental";

  // Category & Badge Styling
  const category = (auth.verdict_category || (auth.verdict === "spoof" ? "potential_spoof" : "likely_genuine")).toUpperCase();
  let badgeColor = "var(--cyan)";
  let badgeText = "INCONCLUSIVE";

  if (category === "LIKELY_GENUINE" || auth.verdict === "bonafide") {
    badgeColor = "#10b981"; // Emerald green
    badgeText = "LIKELY GENUINE";
  } else if (category === "POTENTIAL_SPOOF" || auth.verdict === "spoof") {
    badgeColor = "#ef4444"; // Coral red
    badgeText = "POTENTIAL SPOOF";
  } else {
    badgeColor = "#f59e0b"; // Amber
    badgeText = "INCONCLUSIVE";
  }

  const spoofProb = auth.uncalibrated_spoof_probability !== null && auth.uncalibrated_spoof_probability !== undefined
    ? `${(auth.uncalibrated_spoof_probability * 100).toFixed(1)}%`
    : "—";

  const authScore = auth.authenticity_score !== null && auth.authenticity_score !== undefined
    ? `${(auth.authenticity_score * 100).toFixed(1)}%`
    : "—";

  const speechDuration = metadata.speech_duration_seconds !== undefined
    ? `${metadata.speech_duration_seconds} sec (${Math.round((metadata.speech_ratio || 0) * 100)}% speech)`
    : "Evaluated";

  const explanation = auth.explanation || auth.message || "Acoustic signal analysis completed.";
  const disclaimer = auth.disclaimer || "Authenticity analysis is experimental and based on benchmark datasets.";

  results.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:10px;">
      <h3 style="margin:0;">Analysis Findings</h3>
      <span style="background:${badgeColor}22; border:1px solid ${badgeColor}; color:${badgeColor}; padding:6px 14px; border-radius:20px; font-weight:800; font-size:14px; letter-spacing:0.5px;">
        ${badgeText}
      </span>
    </div>

    <div class="result-grid">
      <div class="result-item">
        <small>Signal Validation</small>
        <strong style="color:#10b981;">PASSED (16 kHz mono)</strong>
      </div>
      <div class="result-item">
        <small>Duration / Speech</small>
        <strong>${metadata.duration_seconds ?? "—"}s total · ${speechDuration}</strong>
      </div>
      <div class="result-item">
        <small>Voice Activity (VAD)</small>
        <strong>${metadata.is_speech_detected ? "Active Speech Present" : "Low / Insufficient Speech"}</strong>
      </div>
      <div class="result-item">
        <small>Acoustic Authenticity</small>
        <strong style="color:${badgeColor}; font-size:1.15rem;">${authScore}</strong>
      </div>
      <div class="result-item">
        <small>Spoof Probability</small>
        <strong>${spoofProb}</strong>
      </div>
      <div class="result-item">
        <small>Model State</small>
        <strong>${auth.model_status === "experimental" ? "Experimental CNN Prototype" : "Untrained Fallback"}</strong>
      </div>
    </div>

    <div style="margin-top:1.25rem; padding:1rem; background:rgba(20,34,56,0.6); border:1px solid var(--line); border-radius:8px;">
      <strong style="color:var(--highlight); display:block; margin-bottom:4px; font-size:14px;">Evaluation Context & Explanation:</strong>
      <p style="margin:0 0 8px 0; font-size:14px; color:var(--text); line-height:1.5;">${explanation}</p>
      <small style="color:var(--muted); display:block; line-height:1.4;">${disclaimer}</small>
    </div>

    <div style="margin-top:1.25rem; display:flex; justify-content:flex-end;">
      <button id="test-another-btn" class="button" type="button" style="background:transparent; border:1px solid var(--line); color:var(--text); padding:8px 16px; border-radius:6px; cursor:pointer;">
        ↺ Analyze Another Audio
      </button>
    </div>
  `;

  document.querySelector("#test-another-btn")?.addEventListener("click", resetFile);
  results.classList.remove("is-hidden");
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  setStatus("Validating audio signal and acoustic energy...", "loading");

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const validationResponse = await fetch(`${API_BASE}/api/v1/audio/validate`, {
      method: "POST",
      body: form,
    });
    if (!validationResponse.ok) throw new Error(await endpointError(validationResponse));
    const validation = await validationResponse.json();

    setStatus("Extracting speech window and analyzing acoustic authenticity...", "loading");

    const analysisForm = new FormData();
    analysisForm.append("file", selectedFile);

    const analysisResponse = await fetch(`${API_BASE}/api/v1/analyze`, {
      method: "POST",
      body: analysisForm,
    });
    if (!analysisResponse.ok) throw new Error(await endpointError(analysisResponse));

    renderResults(validation, await analysisResponse.json());
    setStatus("Analysis complete. Review the acoustic findings below.");
  } catch (error) {
    setStatus(`${error.message || "Unable to reach the Vani Kavach API."}`, "error");
  } finally {
    analyzeButton.disabled = !selectedFile;
  }
}

// Event Listeners
if (browseButton) browseButton.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
if (recordButton) recordButton.addEventListener("click", toggleRecording);
if (dropZone) {
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) =>
    dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add("is-dragging"); })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove("is-dragging"); })
  );
  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  });
}
if (fileInput) fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
if (removeButton) removeButton.addEventListener("click", resetFile);
if (analyzeButton) analyzeButton.addEventListener("click", analyze);
