const API_BASE = "http://localhost:8000";

let selectedFile = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const confSlider = document.getElementById("conf-slider");
const confVal = document.getElementById("conf-val");
const btnPredict = document.getElementById("btn-predict");
const btnIcon = document.getElementById("btn-icon");
const btnText = document.getElementById("btn-text");

const kpiTotal = document.getElementById("kpi-total");
const kpiCompliant = document.getElementById("kpi-compliant");
const kpiViolation = document.getElementById("kpi-violation");
const kpiRate = document.getElementById("kpi-rate");

const overallBadge = document.getElementById("overall-badge");
const placeholder = document.getElementById("placeholder");
const resultImg = document.getElementById("result-img");
const tableContainer = document.getElementById("table-container");
const tableBody = document.getElementById("table-body");
const apiStatusText = document.getElementById("api-status-text");

// Initialize & Health Check
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      apiStatusText.textContent = "FastAPI Server Online (YOLO Ready)";
    } else {
      apiStatusText.textContent = "API Server Error";
    }
  } catch (err) {
    apiStatusText.textContent = "API Disconnected (Offline)";
  }
}
checkApiHealth();

// Confidence Slider
confSlider.addEventListener("input", (e) => {
  confVal.textContent = e.target.value;
});

// File Selection Events
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  fileInfo.style.display = "flex";
  btnPredict.disabled = false;
}

// Inference Execution
btnPredict.addEventListener("click", async () => {
  if (!selectedFile) return;

  // UI Loading State
  btnPredict.disabled = true;
  btnIcon.innerHTML = `<div class="spinner"></div>`;
  btnText.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  const conf = confSlider.value;
  const isVideo = selectedFile.type.startsWith("video/") || selectedFile.name.match(/\.(mp4|avi|mov|mkv)$/i);
  const endpoint = isVideo ? `${API_BASE}/predict/video?conf=${conf}` : `${API_BASE}/predict/image?conf=${conf}`;

  try {
    const res = await fetch(endpoint.replace('/predict/image', '/predict'), {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      alert(`API Error: ${errData.detail || "Failed to execute prediction"}`);
      return;
    }

    const data = await res.json();
    renderResults(data, isVideo);
  } catch (err) {
    console.error("Predict error:", err);
    alert(`Server Connection Error: ${err.message}`);
  } finally {
    btnPredict.disabled = false;
    btnIcon.textContent = "⚡";
    btnText.textContent = "Run Safety Check (Inference)";
  }
});

function renderResults(data, isVideo) {
  // Update Image View
  const imageB64 = data.annotated_image || data.preview_image;
  if (imageB64) {
    resultImg.src = imageB64;
    resultImg.style.display = "block";
    placeholder.style.display = "none";
  }

  // Update KPIs
  if (isVideo) {
    kpiTotal.textContent = data.max_people_detected || 0;
    kpiViolation.textContent = data.total_violation_events || 0;
    const compliant = Math.max(0, (data.max_people_detected || 0) - (data.total_violation_events || 0));
    kpiCompliant.textContent = compliant;
    const rate = data.max_people_detected > 0 ? Math.round((compliant / data.max_people_detected) * 100) : 100;
    kpiRate.textContent = `${rate}%`;
  } else {
    kpiTotal.textContent = data.people_count || 0;
    kpiCompliant.textContent = data.compliant_count || 0;
    kpiViolation.textContent = data.violation_count || 0;
    const rate = data.people_count > 0 ? Math.round((data.compliant_count / data.people_count) * 100) : 100;
    kpiRate.textContent = `${rate}%`;
  }

  // Overall Status Badge
  const status = data.compliance_status;
  overallBadge.className = "compliance-badge";

  if (status === "COMPLIANT") {
    overallBadge.classList.add("badge-compliant");
    overallBadge.innerHTML = `🟢 <span>COMPLIANT (LOW RISK)</span>`;
  } else if (status === "VIOLATION") {
    overallBadge.classList.add("badge-violation");
    overallBadge.innerHTML = `🔴 <span>VIOLATION DETECTED (${data.risk_level} RISK)</span>`;
  } else {
    overallBadge.classList.add("badge-none");
    overallBadge.innerHTML = `⚪ <span>NO PERSON DETECTED</span>`;
  }

  // Render Table Breakdown for Images
  if (!isVideo && data.persons && data.persons.length > 0) {
    tableContainer.style.display = "block";
    tableBody.innerHTML = "";

    data.persons.forEach((p) => {
      const tr = document.createElement("tr");

      const helmetPill = p.has_helmet
        ? `<span class="gear-pill gear-ok">✅ Helmet</span>`
        : `<span class="gear-pill gear-miss">❌ Missing Helmet</span>`;

      const vestPill = p.has_vest
        ? `<span class="gear-pill gear-ok">✅ Vest</span>`
        : `<span class="gear-pill gear-miss">❌ Missing Vest</span>`;

      const statusTag = p.is_compliant
        ? `<strong style="color: #34d399;">🟢 Passed</strong>`
        : `<strong style="color: #f87171;">🔴 Violation</strong>`;

      const missingText = p.missing_items.length > 0 ? p.missing_items.join(", ") : "None";

      tr.innerHTML = `
        <td><strong>Person #${p.person_id}</strong></td>
        <td>${helmetPill}</td>
        <td>${vestPill}</td>
        <td>${statusTag}</td>
        <td><span style="color: #cbd5e1;">${missingText}</span></td>
      `;

      tableBody.appendChild(tr);
    });
  } else {
    tableContainer.style.display = "none";
  }
}
