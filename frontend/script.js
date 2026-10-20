const API_BASE = (new URLSearchParams(window.location.search).get("api") || localStorage.getItem("vaniKavachApi") || "http://127.0.0.1:8000").replace(/\/$/, "");
const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const browseButton = document.querySelector("#browse-button");
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

const supportedExtensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];
const formatBytes = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
const setStatus = (message, type = "") => { statusMessage.textContent = message; statusMessage.className = `status-message ${type}`; };
const endpointError = async (response) => {
  let detail = `Request failed (${response.status})`;
  try { const body = await response.json(); detail = body.detail || detail; } catch (_) { /* use HTTP status */ }
  return detail;
};

function selectFile(file) {
  if (!file) return;
  const extension = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!supportedExtensions.includes(extension)) { setStatus(`Unsupported format. Use ${supportedExtensions.join(", ")}.`, "error"); return; }
  if (file.size > 15 * 1024 * 1024) { setStatus("This file is larger than the 15 MB limit.", "error"); return; }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = `${formatBytes(file.size)} · ${extension.slice(1).toUpperCase()}`;
  preview.src = objectUrl;
  fileCard.classList.remove("is-hidden");
  dropZone.classList.add("is-hidden");
  analyzeButton.disabled = false;
  results.classList.add("is-hidden");
  setStatus("Ready for signal validation.");
}

function resetFile() {
  selectedFile = null;
  fileInput.value = "";
  preview.removeAttribute("src");
  preview.load();
  fileCard.classList.add("is-hidden");
  dropZone.classList.remove("is-hidden");
  analyzeButton.disabled = true;
  results.classList.add("is-hidden");
  setStatus("");
}

function renderResults(validation, analysis) {
  const metadata = validation.metadata || analysis.audio_validation?.metadata || {};
  const auth = analysis.authenticity_analysis || {};
  const experimental = auth.model_status === "experimental";
  const verdict = experimental && auth.verdict ? auth.verdict.toUpperCase() : "UNAVAILABLE";
  const note = auth.disclaimer || "Authenticity analysis requires a trained and evaluated model.";
  results.innerHTML = `<h3>Analysis findings</h3>
    <div class="result-grid">
      <div class="result-item"><small>Validation</small><strong style="color:var(--green)">SIGNAL VALID</strong></div>
      <div class="result-item"><small>Duration</small><strong>${metadata.duration_seconds ?? "—"} sec</strong></div>
      <div class="result-item"><small>Original format</small><strong>${metadata.original_sample_rate ?? "—"} Hz · ${metadata.original_channels ?? "—"} ch</strong></div>
      <div class="result-item"><small>Standardized</small><strong>${metadata.target_sample_rate ?? "—"} Hz · mono</strong></div>
      <div class="result-item"><small>RMS energy</small><strong>${metadata.rms_energy ?? "—"}</strong></div>
      <div class="result-item"><small>Authenticity model</small><strong>${verdict}</strong></div>
    </div>
    <p class="model-note ${experimental ? "experimental" : ""}">${experimental ? `Experimental model output: ${auth.message || "analysis completed"}. ${note}` : `Authenticity analysis is currently unavailable. Audio validation is complete, but a trained authenticity model is required to assess whether this recording is synthetic. ${note}`}</p>`;
  results.classList.remove("is-hidden");
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  setStatus("Validating signal…", "loading");
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const validationResponse = await fetch(`${API_BASE}/api/v1/audio/validate`, { method: "POST", body: form });
    if (!validationResponse.ok) throw new Error(await endpointError(validationResponse));
    const validation = await validationResponse.json();
    setStatus("Running available analysis…", "loading");
    const analysisForm = new FormData();
    analysisForm.append("file", selectedFile);
    const analysisResponse = await fetch(`${API_BASE}/api/v1/analyze`, { method: "POST", body: analysisForm });
    if (!analysisResponse.ok) throw new Error(await endpointError(analysisResponse));
    renderResults(validation, await analysisResponse.json());
    setStatus("Analysis complete. Review the evidence below.");
  } catch (error) {
    setStatus(`${error.message || "Unable to reach the Vani Kavach API."} Check that FastAPI is running at ${API_BASE}.`, "error");
  } finally {
    analyzeButton.disabled = !selectedFile;
  }
}

browseButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
removeButton.addEventListener("click", resetFile);
analyzeButton.addEventListener("click", analyze);
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); }));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); }));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
