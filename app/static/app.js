// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Video Face Grab — Real-Time SSE Frontend Controller
 */

(() => {
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("video-file-input");
  const dropZone = document.getElementById("drop-zone");
  const selectedFileName = document.getElementById("selected-file-name");
  const selectedFileMeta = document.getElementById("selected-file-meta");
  const startBtn = document.getElementById("start-process-btn");
  const resetBtn = document.getElementById("reset-session-btn");

  const storageBucketLabel = document.getElementById("storage-bucket-label");
  const jobStatusBadge = document.getElementById("job-status-badge");
  const targetGcsFolder = document.getElementById("target-gcs-folder");
  const progressStageText = document.getElementById("progress-stage-text");
  const progressPercentText = document.getElementById("progress-percent-text");
  const progressBarContainer = document.getElementById("progress-bar-container");
  const progressBarFill = document.getElementById("progress-bar-fill");

  const metricFrames = document.getElementById("metric-frames");
  const metricTimestamp = document.getElementById("metric-timestamp");
  const metricPeople = document.getElementById("metric-people");
  const metricSaved = document.getElementById("metric-saved");

  const activityLog = document.getElementById("activity-log");
  const eventCounter = document.getElementById("event-counter");
  const emptyGalleryState = document.getElementById("empty-gallery-state");
  const facesGrid = document.getElementById("faces-grid");
  const galleryCountBadge = document.getElementById("gallery-count-badge");

  const inspectorModal = document.getElementById("frame-inspector-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const modalPersonTitle = document.getElementById("modal-person-title");
  const modalGcsUri = document.getElementById("modal-gcs-uri");
  const modalPreviewImg = document.getElementById("modal-preview-img");
  const modalCaption = document.getElementById("modal-caption");

  let activeEventSource = null;
  let totalEventsCount = 0;
  let savedCount = 0;

  // Fetch initial backend storage configuration
  async function checkBackendHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) return;
      const data = await res.json();
      const modeLabel = data.storage_mode === "gcs" ? "GCS Bucket" : "Local GCS Fallback";
      storageBucketLabel.textContent = `${modeLabel}: ${data.gcs_bucket}`;
    } catch {
      storageBucketLabel.textContent = "Backend connection offline";
    }
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  }

  function appendLog(message, variant = "muted") {
    totalEventsCount += 1;
    eventCounter.textContent = `${totalEventsCount} events`;
    const li = document.createElement("li");
    li.className = `log-entry log-${variant}`;
    const now = new Date().toLocaleTimeString([], { hour12: false });
    li.textContent = `[${now}] ${message}`;
    activityLog.prepend(li);
  }

  function setProgress(percent, stageText) {
    const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
    progressBarFill.style.width = `${clamped}%`;
    progressBarContainer.setAttribute("aria-valuenow", String(Math.round(clamped)));
    progressPercentText.textContent = `${clamped.toFixed(1)}%`;
    if (stageText) {
      progressStageText.textContent = stageText;
    }
  }

  function updateSelectedFile(file) {
    if (!file) {
      selectedFileName.textContent = "Drop a video file here or click to browse";
      selectedFileMeta.textContent =
        "Each unique human face is clustered and saved as a lossless PNG in its own GCS run folder.";
      startBtn.disabled = true;
      return;
    }
    selectedFileName.textContent = file.name;
    selectedFileMeta.textContent = `Size: ${formatBytes(file.size)} · Ready to analyze`;
    startBtn.disabled = false;
  }

  // Drag & Drop handlers
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      fileInput.files = e.dataTransfer.files;
      updateSelectedFile(fileInput.files[0]);
    }
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    updateSelectedFile(fileInput.files?.[0] || null);
  });

  function openInspectorModal({ title, gcsUri, imageUrl, caption }) {
    modalPersonTitle.textContent = title;
    modalGcsUri.textContent = gcsUri;
    modalPreviewImg.src = imageUrl;
    modalCaption.textContent = caption;
    if (typeof inspectorModal.showModal === "function") {
      inspectorModal.showModal();
    }
  }

  closeModalBtn.addEventListener("click", () => {
    inspectorModal.close();
  });

  inspectorModal.addEventListener("click", (e) => {
    if (e.target === inspectorModal) {
      inspectorModal.close();
    }
  });

  function upsertPersonCard(ev, mode) {
    emptyGalleryState.hidden = true;
    const cardId = `person-card-${ev.person_id}`;
    let card = document.getElementById(cardId);

    if (!card) {
      card = document.createElement("article");
      card.id = cardId;
      card.className = "person-card";
      card.innerHTML = `
        <div class="card-media">
          <img class="person-thumb" alt="${ev.label} best face crop" />
          <div class="card-overlay-top">
            <span class="person-pill">${ev.label}</span>
            <span class="clarity-pill">Clarity <span class="score-val"></span></span>
          </div>
        </div>
        <div class="card-body">
          <div class="card-meta-row">
            <span class="sightings-text"></span>
            <span class="frame-time-text"></span>
          </div>
          <div class="quality-bars">
            <div class="q-item">
              <span class="q-label">Sharpness</span>
              <span class="q-val q-sharp"></span>
            </div>
            <div class="q-item">
              <span class="q-label">Frontal</span>
              <span class="q-val q-frontal"></span>
            </div>
            <div class="q-item">
              <span class="q-label">Conf</span>
              <span class="q-val q-conf"></span>
            </div>
          </div>
          <div class="card-gcs-status">
            <span class="gcs-saved-uri">Scanning for clearest frame…</span>
            <div class="card-actions" hidden>
              <button type="button" class="card-btn inspect-full-btn" id="inspect-full-${ev.person_id}">
                View Full Frame PNG
              </button>
              <a class="card-btn download-crop-link" id="download-crop-${ev.person_id}" download>
                Face PNG
              </a>
            </div>
          </div>
        </div>
      `;
      facesGrid.appendChild(card);
    }

    const thumb = card.querySelector(".person-thumb");
    if (ev.preview_data_uri) {
      thumb.src = ev.preview_data_uri;
    } else if (ev.face_crop_url) {
      thumb.src = ev.face_crop_url;
    }

    const q = ev.quality || {};
    card.querySelector(".score-val").textContent = `${Number(q.composite || 0).toFixed(1)}`;
    card.querySelector(".sightings-text").textContent = `${ev.sightings_count || 1} sighting(s)`;
    const frameIdx = ev.frame_index ?? ev.best_frame_index ?? 0;
    const ts = ev.timestamp_sec ?? ev.best_timestamp_sec ?? 0;
    card.querySelector(".frame-time-text").textContent = `Frame #${frameIdx} (${Number(ts).toFixed(2)}s)`;

    card.querySelector(".q-sharp").textContent = `${Number(q.sharpness || 0).toFixed(0)}`;
    card.querySelector(".q-frontal").textContent = `${Number(q.frontal || 0).toFixed(0)}`;
    card.querySelector(".q-conf").textContent = `${Number(q.confidence || 0).toFixed(0)}%`;

    if (mode === "upgraded") {
      card.classList.remove("flash-upgrade");
      void card.offsetWidth;
      card.classList.add("flash-upgrade");
    }

    if (mode === "saved") {
      card.classList.add("flash-saved");
      const uriEl = card.querySelector(".gcs-saved-uri");
      uriEl.textContent = ev.full_frame_uri;
      const actionsRow = card.querySelector(".card-actions");
      actionsRow.hidden = false;

      const inspectBtn = card.querySelector(".inspect-full-btn");
      inspectBtn.onclick = () =>
        openInspectorModal({
          title: `${ev.label} — Best Clearest Full Frame (Frame #${frameIdx} @ ${Number(ts).toFixed(2)}s)`,
          gcsUri: ev.full_frame_uri,
          imageUrl: ev.full_frame_url,
          caption: `Composite Clarity Score: ${q.composite} · Saved to ${ev.full_frame_uri}`,
        });

      const downloadCrop = card.querySelector(".download-crop-link");
      downloadCrop.href = ev.face_crop_url;
    }

    const currentCards = facesGrid.querySelectorAll(".person-card").length;
    metricPeople.textContent = String(currentCards);
    galleryCountBadge.textContent = `${currentCards} ${currentCards === 1 ? "identity" : "identities"}`;
  }

  function connectJobStream(jobInfo) {
    if (activeEventSource) {
      activeEventSource.close();
    }

    const es = new EventSource(jobInfo.stream_url);
    activeEventSource = es;

    es.addEventListener("video_loaded", (msg) => {
      const ev = JSON.parse(msg.data);
      metricFrames.textContent = `0 / ${ev.total_frames}`;
      appendLog(
        `Video opened (${ev.resolution} @ ${ev.native_fps} fps, sampling every ${ev.sample_stride} frames).`,
        "muted"
      );
      setProgress(2, "Scanning frames and clustering human faces…");
    });

    es.addEventListener("frame_progress", (msg) => {
      const ev = JSON.parse(msg.data);
      metricFrames.textContent = `${ev.frame_index} / ${ev.total_frames}`;
      metricTimestamp.textContent = `${Number(ev.timestamp_sec).toFixed(2)}s`;
      setProgress(
        ev.percent,
        `Analyzing frame ${ev.frame_index} of ${ev.total_frames} (${ev.discerned_faces_count} human(s) discerned)…`
      );
    });

    es.addEventListener("face_discovered", (msg) => {
      const ev = JSON.parse(msg.data);
      upsertPersonCard(ev, "discovered");
      appendLog(
        `Discovered ${ev.label} at ${ev.timestamp_sec}s (initial clarity score: ${ev.quality.composite}).`,
        "discovered"
      );
    });

    es.addEventListener("face_upgraded", (msg) => {
      const ev = JSON.parse(msg.data);
      upsertPersonCard(ev, "upgraded");
      appendLog(
        `Upgraded ${ev.label} at frame #${ev.frame_index} (${ev.timestamp_sec}s): clarity ${ev.previous_best_score} → ${ev.quality.composite}.`,
        "upgraded"
      );
    });

    es.addEventListener("uploading_face", (msg) => {
      const ev = JSON.parse(msg.data);
      appendLog(`Uploading ${ev.label} PNGs to ${ev.folder_name}/…`, "muted");
    });

    es.addEventListener("face_saved", (msg) => {
      const ev = JSON.parse(msg.data);
      savedCount += 1;
      metricSaved.textContent = String(savedCount);
      upsertPersonCard(ev, "saved");
      appendLog(`Saved ${ev.label} → ${ev.full_frame_uri}`, "saved");
    });

    es.addEventListener("job_complete", (msg) => {
      const ev = JSON.parse(msg.data);
      const savedIds = new Set((ev.people || []).map((p) => `person-card-${p.person_id}`));
      facesGrid.querySelectorAll(".person-card").forEach((card) => {
        if (!savedIds.has(card.id)) {
          card.remove();
        }
      });
      const currentCards = facesGrid.querySelectorAll(".person-card").length;
      metricPeople.textContent = String(currentCards);
      galleryCountBadge.textContent = `${currentCards} ${currentCards === 1 ? "identity" : "identities"}`;
      if (currentCards === 0) {
        emptyGalleryState.hidden = false;
      }
      setProgress(
        100,
        `Complete! Saved ${ev.total_people_saved} unique human face PNG set(s) to ${ev.folder_name}/.`
      );
      jobStatusBadge.dataset.state = "completed";
      jobStatusBadge.textContent = "Completed";
      resetBtn.hidden = false;
      startBtn.disabled = false;
      appendLog(
        `Finished processing ${ev.total_frames_scanned} frames. ${ev.total_people_saved} human(s) saved to gs://${ev.bucket_name}/${ev.folder_name}/`,
        "saved"
      );
      es.close();
    });

    es.addEventListener("job_error", (msg) => {
      const ev = JSON.parse(msg.data);
      jobStatusBadge.dataset.state = "failed";
      jobStatusBadge.textContent = "Failed";
      progressStageText.textContent = `Error: ${ev.message}`;
      startBtn.disabled = false;
      resetBtn.hidden = false;
      appendLog(`Error: ${ev.message}`, "error");
      es.close();
    });
  }

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) return;

    // Reset UI counters for new run
    facesGrid.innerHTML = "";
    emptyGalleryState.hidden = false;
    savedCount = 0;
    totalEventsCount = 0;
    metricFrames.textContent = "0 / 0";
    metricTimestamp.textContent = "0.00s";
    metricPeople.textContent = "0";
    metricSaved.textContent = "0";
    galleryCountBadge.textContent = "0 identities";
    activityLog.innerHTML = "";

    startBtn.disabled = true;
    jobStatusBadge.dataset.state = "processing";
    jobStatusBadge.textContent = "Processing";
    setProgress(1, "Uploading video file to backend…");
    appendLog(`Uploading '${file.name}' (${formatBytes(file.size)})…`, "muted");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errPayload = await res.json().catch(() => ({}));
        throw new Error(errPayload.detail || `Upload failed with status ${res.status}`);
      }
      const jobInfo = await res.json();
      targetGcsFolder.textContent = `gs://${jobInfo.bucket_name}/${jobInfo.folder_name}/`;
      appendLog(`Created GCS folder prefix: gs://${jobInfo.bucket_name}/${jobInfo.folder_name}/`, "discovered");
      connectJobStream(jobInfo);
    } catch (err) {
      jobStatusBadge.dataset.state = "failed";
      jobStatusBadge.textContent = "Failed";
      progressStageText.textContent = err.message;
      appendLog(`Upload failed: ${err.message}`, "error");
      startBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", () => {
    fileInput.value = "";
    updateSelectedFile(null);
    resetBtn.hidden = true;
    jobStatusBadge.dataset.state = "idle";
    jobStatusBadge.textContent = "Idle";
    setProgress(0, "Waiting for video upload…");
  });

  checkBackendHealth();
})();
