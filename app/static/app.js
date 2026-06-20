const state = {
  gestures: [],
  session: null,
  stream: null,
  lastTest: null,
  trained: false,
  knownLabels: [],
  apiBaseUrl: localStorage.getItem("gestureRagApiBaseUrl") || "",
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const base = state.apiBaseUrl.replace(/\/$/, "");
  const url = `${base}${path}`;
  const res = await fetch(url, options);
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    throw new Error(data.detail || res.statusText);
  }
  return data;
}

function log(id, value) {
  $(id).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function option(value) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = value;
  return opt;
}

function setStatus(id, message, tone = "info") {
  const el = $(id);
  el.textContent = message;
  el.dataset.tone = tone;
}

async function withBusy(buttonId, statusId, busyText, fn) {
  const button = $(buttonId);
  const original = button.textContent;
  button.disabled = true;
  button.textContent = busyText;
  setStatus(statusId, busyText);
  try {
    const result = await fn();
    return result;
  } catch (err) {
    console.error(err);
    setStatus(statusId, err.message || String(err), "error");
    alert(err.message || String(err));
    throw err;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function init() {
  bindEvents();
  $("apiBaseUrl").value = state.apiBaseUrl;
  updateApiStatus();
  const data = await api("/api/gestures");
  state.gestures = data.gestures;
  renderGestureSuggestions();
  renderKnownLabels();
  renderTrainingLabels();
  updateApiControls();
}

function renderGestureSuggestions() {
  const list = $("gestureSuggestions");
  list.innerHTML = "";
  state.gestures.forEach((gesture) => list.appendChild(option(gesture)));
}

function currentGestureName() {
  return $("gestureName").value.trim();
}

function rememberGesture(label) {
  if (!label) return;
  if (!state.knownLabels.some((x) => x.toLowerCase() === label.toLowerCase())) {
    state.knownLabels.push(label);
  }
  renderKnownLabels();
  renderTrainingLabels();
}

function renderKnownLabels() {
  const grid = $("gestureGrid");
  if (!state.knownLabels.length) {
    grid.innerHTML = "<p class='muted'>No examples added yet.</p>";
    return;
  }
  grid.innerHTML = state.knownLabels
    .map((label) => `<button class="gesture-chip" type="button" data-label="${escapeHtml(label)}">${escapeHtml(label)}</button>`)
    .join("");
  grid.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      $("gestureName").value = button.dataset.label;
      renderTrainingLabels();
    });
  });
}

function renderTrainingLabels() {
  const select = $("trainingLabel");
  select.innerHTML = "";
  const label = currentGestureName();
  const labels = label ? [label, ...state.knownLabels.filter((x) => x.toLowerCase() !== label.toLowerCase())] : state.knownLabels;
  labels.forEach((item) => select.appendChild(option(item)));
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function resetSession() {
  state.session = null;
  state.lastTest = null;
  state.trained = false;
  $("projectBadge").textContent = "Ready";
  $("modelStatus").textContent = "Model not trained";
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-page").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      $(btn.dataset.tab).classList.add("active");
    });
  });

  $("gestureName").addEventListener("input", renderTrainingLabels);
  $("useTranscription").addEventListener("change", updateApiControls);
  $("useLlm").addEventListener("change", updateApiControls);
  $("saveApiUrlBtn").addEventListener("click", () =>
    withBusy("saveApiUrlBtn", "trainStatus", "Checking backend...", saveApiUrl),
  );
  $("startCameraBtn").addEventListener("click", () =>
    withBusy("startCameraBtn", "trainStatus", "Starting camera...", startCamera),
  );
  $("recordTrainBtn").addEventListener("click", () =>
    withBusy("recordTrainBtn", "trainStatus", "Recording training clip...", async () => {
      await recordToInput("trainingFile", 5000, "trainStatus");
      await uploadTraining();
    }),
  );
  $("recordTestBtn").addEventListener("click", () =>
    withBusy("recordTestBtn", "testStatus", "Recording test clip...", async () => {
      await recordToInput("testFile", 15000, "testStatus");
      await uploadTest();
    }),
  );
  $("uploadTrainBtn").addEventListener("click", () =>
    withBusy("uploadTrainBtn", "trainStatus", "Processing training clip...", uploadTraining),
  );
  $("trainModelBtn").addEventListener("click", () =>
    withBusy("trainModelBtn", "trainStatus", "Training KNN model...", trainModel),
  );
  $("uploadTestBtn").addEventListener("click", () =>
    withBusy("uploadTestBtn", "testStatus", "Running gesture inference...", uploadTest),
  );
  $("runRagBtn").addEventListener("click", () =>
    withBusy("runRagBtn", "ragStatus", "Running RAG analysis...", runRag),
  );
}

async function saveApiUrl() {
  state.apiBaseUrl = $("apiBaseUrl").value.trim().replace(/\/$/, "");
  localStorage.setItem("gestureRagApiBaseUrl", state.apiBaseUrl);
  updateApiStatus();
  state.session = null;
  state.lastTest = null;
  state.trained = false;
  $("projectBadge").textContent = "Ready";
  $("modelStatus").textContent = "Model not trained";
  const data = await api("/api/gestures");
  state.gestures = data.gestures;
  renderGestureSuggestions();
  renderTrainingLabels();
  updateApiControls();
}

function updateApiStatus() {
  $("apiStatus").textContent = state.apiBaseUrl
    ? `Using ${state.apiBaseUrl}`
    : "Using same-origin API.";
}

async function createSession(firstLabel) {
  const session = await api("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_gestures: [firstLabel] }),
  });
  state.session = session;
  $("projectBadge").textContent = "Examples in progress";
  setStatus("trainStatus", "Ready. Add examples, then train.", "success");
  return session;
}

async function ensureSession(label) {
  if (state.session) return state.session;
  return createSession(label);
}

function updateApiControls() {
  const useTx = $("useTranscription").checked;
  const useLlm = $("useLlm").checked;
  $("transcriptionProvider").disabled = !useTx;
  $("transcriptionModel").disabled = !useTx;
  $("transcriptionKey").disabled = !useTx;
  $("llmProvider").disabled = !useLlm;
  $("llmModel").disabled = !useLlm;
  $("llmKey").disabled = !useLlm;
  $("llmBaseUrl").disabled = !useLlm;
}

async function startCamera() {
  if (state.stream) return state.stream;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch (err) {
    throw new Error("Camera access is blocked. Use the video upload field, or reset camera permission in the browser address bar and reload.");
  }
  $("preview").srcObject = state.stream;
  setStatus("trainStatus", "Camera is on. Recording buttons will send clips for processing.", "success");
  return state.stream;
}

async function recordToInput(inputId, ms, statusId) {
  const stream = await startCamera();
  const chunks = [];
  const options = MediaRecorder.isTypeSupported("video/webm") ? { mimeType: "video/webm" } : undefined;
  const recorder = new MediaRecorder(stream, options);
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.start();
  const seconds = Math.ceil(ms / 1000);
  for (let remaining = seconds; remaining > 0; remaining -= 1) {
    setStatus(statusId, `Recording... ${remaining}s`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  recorder.stop();
  await new Promise((resolve) => (recorder.onstop = resolve));
  const type = recorder.mimeType || "video/webm";
  const blob = new Blob(chunks, { type });
  const extension = type.includes("mp4") ? "mp4" : "webm";
  const file = new File([blob], `${inputId}_${Date.now()}.${extension}`, { type });
  const dt = new DataTransfer();
  dt.items.add(file);
  $(inputId).files = dt.files;
  setStatus(statusId, "Recording saved. Sending it to the backend...");
}

async function uploadTraining() {
  const label = $("trainingLabel").value || currentGestureName();
  if (!label) {
    throw new Error("Type a gesture name first.");
  }
  await ensureSession(label);
  const file = $("trainingFile").files[0];
  if (!file) {
    throw new Error("Choose or record a training video.");
  }
  const fd = new FormData();
  fd.append("gesture_label", label);
  fd.append("file", file);
  const result = await api(`/api/projects/${state.session.project_id}/training-video`, {
    method: "POST",
    body: fd,
  });
  rememberGesture(result.gesture_label || label);
  state.session.selected_gestures = result.selected_gestures || state.knownLabels;
  setStatus("trainStatus", `Added ${result.samples_added} samples for ${result.gesture_label || label}.`, "success");
  log("trainingOutput", result);
}

async function trainModel() {
  requireSession();
  const result = await api(`/api/projects/${state.session.project_id}/train`, { method: "POST" });
  state.trained = true;
  $("modelStatus").textContent = `${result.n_samples} samples, k=${result.n_neighbors}`;
  setStatus("trainStatus", "KNN model trained.", "success");
  log("trainingOutput", result);
}

async function uploadTest() {
  requireModel();
  const file = $("testFile").files[0];
  if (!file) {
    throw new Error("Choose or record a test video.");
  }
  const fd = new FormData();
  fd.append("file", file);
  const result = await api(`/api/projects/${state.session.project_id}/test-video`, {
    method: "POST",
    body: fd,
  });
  state.lastTest = result;
  renderSegments(result.segments);
  setStatus("testStatus", `Gesture test complete. ${result.segments.length} segment(s) detected.`, "success");
  log("testOutput", result);
}

function renderSegments(segments) {
  if (!segments.length) {
    $("segmentsTable").innerHTML = "<p class='muted'>No gesture segments detected.</p>";
    return;
  }
  $("segmentsTable").innerHTML = `
    <table>
      <thead><tr><th>Gesture</th><th>Start</th><th>End</th><th>Duration</th><th>Confidence</th></tr></thead>
      <tbody>
        ${segments
          .map(
            (s) => `<tr>
              <td>${s.label}</td>
              <td>${s.start_time_sec}s</td>
              <td>${s.end_time_sec}s</td>
              <td>${s.duration_sec}s</td>
              <td>${s.avg_confidence ?? ""}</td>
            </tr>`,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

async function runRag() {
  requireSession();
  if (!state.lastTest) {
    throw new Error("Run a gesture test first.");
  }
  const fd = new FormData();
  fd.append("query", $("ragQuery").value);
  fd.append("video_file", state.lastTest.video_file);
  fd.append("segments_json", JSON.stringify(state.lastTest.segments));
  fd.append("transcription_provider", $("useTranscription").checked ? $("transcriptionProvider").value : "none");
  fd.append("transcription_model", $("transcriptionModel").value);
  fd.append("transcription_api_key", $("useTranscription").checked ? $("transcriptionKey").value : "");
  fd.append("llm_provider", $("useLlm").checked ? $("llmProvider").value : "none");
  fd.append("llm_model", $("llmModel").value);
  fd.append("llm_api_key", $("useLlm").checked ? $("llmKey").value : "");
  fd.append("llm_base_url", $("useLlm").checked ? $("llmBaseUrl").value : "");

  const result = await api(`/api/projects/${state.session.project_id}/analyze`, {
    method: "POST",
    body: fd,
  });
  setStatus("ragStatus", "RAG analysis complete.", "success");
  log("ragOutput", result);
}

function requireSession() {
  if (!state.session) {
    throw new Error("Add at least one training example first.");
  }
}

function requireModel() {
  requireSession();
  if (!state.trained) {
    throw new Error("Train the model before testing a gesture video.");
  }
}

init().catch((err) => {
  console.error(err);
  $("apiStatus").textContent = `API unavailable: ${err.message}`;
});
