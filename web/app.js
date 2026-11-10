/**
 * DriveGuard - Computer Vision & Improved HOG Experiment
 * Client-Side Application Logic (Vanilla JavaScript)
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    datasetSummary: null,
    currentSample: null,
    currentFile: null,
    currentImageBase64: null,
    lastPipelineResult: null,
    manualRoiEnabled: false,
    manualRoiBox: null,
    isDrawingRoi: false,
    roiStart: { x: 0, y: 0 }
  };

  // DOM Elements
  const elBackendStatus = document.getElementById('backend-status');
  const elBackendStatusText = document.getElementById('backend-status-text');

  // Stats
  const elStatRoot = document.getElementById('stat-root');
  const elStatTotal = document.getElementById('stat-total');
  const elStatDrowsy = document.getElementById('stat-drowsy');
  const elStatNotDrowsy = document.getElementById('stat-notdrowsy');
  const elStatSubjects = document.getElementById('stat-subjects');

  // Controls
  const elSampleSelect = document.getElementById('sample-select');
  const elBtnLoadSample = document.getElementById('btn-load-sample');
  const elFileUpload = document.getElementById('file-upload');
  const elBtnProcess = document.getElementById('btn-process');
  const elBtnReset = document.getElementById('btn-reset');
  const elBtnScanDataset = document.getElementById('btn-scan-dataset');
  const elGtBadge = document.getElementById('gt-badge');
  const elChkManualRoi = document.getElementById('chk-manual-roi');
  const elManualRoiPanel = document.getElementById('manual-roi-panel');
  const elBtnClearRoi = document.getElementById('btn-clear-roi');
  const elRoiX = document.getElementById('roi-x');
  const elRoiY = document.getElementById('roi-y');
  const elRoiW = document.getElementById('roi-w');
  const elRoiH = document.getElementById('roi-h');
  const elBtnExportCsv = document.getElementById('btn-export-csv');
  const elExportStatusBox = document.getElementById('export-status-box');
  const elExportMsg = document.getElementById('export-msg');

  // Stage Images
  const elImgOriginal = document.getElementById('img-original');
  const elCanvasRoi = document.getElementById('canvas-roi');
  const elImgGaussian = document.getElementById('img-gaussian');
  const elImgClahe = document.getElementById('img-clahe');
  const elImgFace = document.getElementById('img-face');
  const elImgEye = document.getElementById('img-eye');
  const elImgRoi = document.getElementById('img-roi');

  // HOG Visuals
  const elImgHogMag = document.getElementById('img-hog-mag');
  const elImgHogOri = document.getElementById('img-hog-ori');
  const elImgHogShifted = document.getElementById('img-hog-shifted');
  const elImgHogBinary = document.getElementById('img-hog-binary');
  const elImgHogBarcode = document.getElementById('img-hog-barcode');

  // Metrics
  const elPipelineTimeBadge = document.getElementById('pipeline-time-badge');
  const elPipelineMs = document.getElementById('pipeline-ms');
  const elMetaOriginal = document.getElementById('meta-original');
  const elMetaFace = document.getElementById('meta-face');
  const elMetaEye = document.getElementById('meta-eye');
  const elMetaRoi = document.getElementById('meta-roi');
  const elPillEyeMode = document.getElementById('pill-eye-mode');
  const elVectorDimBadge = document.getElementById('vector-dim-badge');
  const elValFeatDim = document.getElementById('val-feat-dim');
  const elValNumBlocks = document.getElementById('val-num-blocks');
  const elValVectorSnippet = document.getElementById('val-vector-snippet');

  // =========================================================================
  // INITIALIZATION & HEALTH CHECK
  // =========================================================================
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        elBackendStatus.className = 'status-indicator status-online';
        elBackendStatusText.textContent = `Online (${data.service})`;
        return true;
      }
    } catch (err) {
      elBackendStatus.className = 'status-indicator status-error';
      elBackendStatusText.textContent = 'Backend Offline';
    }
    return false;
  }

  async function loadDatasetSummary() {
    try {
      const res = await fetch('/api/dataset/summary');
      if (!res.ok) return;
      const data = await res.json();
      state.datasetSummary = data;

      elStatRoot.textContent = data.dataset_root.split(/[\\/]/).slice(-2).join('/');
      elStatRoot.title = data.dataset_root;
      elStatTotal.textContent = data.total_samples.toLocaleString();
      elStatDrowsy.textContent = (data.counts.drowsy || 0).toLocaleString();
      elStatNotDrowsy.textContent = (data.counts.notdrowsy || 0).toLocaleString();
      elStatSubjects.textContent = (data.subject_count || 0);

      await loadDatasetSamplesDropdown();
    } catch (err) {
      console.error('Error loading dataset summary:', err);
    }
  }

  async function loadDatasetSamplesDropdown() {
    try {
      const res = await fetch('/api/dataset/samples?limit=150');
      if (!res.ok) return;
      const samples = await res.json();

      elSampleSelect.innerHTML = '<option value="">-- Choose sample from NTHU-DDD --</option>';
      samples.forEach((s, idx) => {
        const opt = document.createElement('option');
        opt.value = s.file_path;
        const subj = s.subject_id ? `Subj ${s.subject_id}` : 'Subj ?';
        const scen = s.scenario ? `[${s.scenario}]` : '';
        opt.textContent = `[${s.label.toUpperCase()}] ${subj} ${scen} #${s.frame_index ?? idx} (${s.source_file})`;
        opt.dataset.metadata = JSON.stringify(s);
        elSampleSelect.appendChild(opt);
      });
    } catch (err) {
      console.error('Error loading samples dropdown:', err);
    }
  }

  // =========================================================================
  // INPUT SELECTION & PREVIEWS
  // =========================================================================
  elBtnLoadSample.addEventListener('click', async () => {
    const selectedPath = elSampleSelect.value;
    if (!selectedPath) return;

    const opt = elSampleSelect.selectedOptions[0];
    const meta = opt ? JSON.parse(opt.dataset.metadata) : null;
    state.currentSample = meta;
    state.currentFile = null;

    updateGroundTruthBadge(meta ? meta.label : null);

    try {
      const res = await fetch('/api/load-sample-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: selectedPath })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setImageSource(elImgOriginal, data.image_base64);
        state.currentImageBase64 = data.image_base64;
        elMetaOriginal.textContent = `Resolution: ${data.width}x${data.height} | File: ${data.filename}`;
        elBtnProcess.disabled = false;
        resetRoiCanvas();
      }
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  });

  elFileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    state.currentFile = file;
    state.currentSample = null;
    updateGroundTruthBadge(null);

    const reader = new FileReader();
    reader.onload = (evt) => {
      state.currentImageBase64 = evt.target.result;
      setImageSource(elImgOriginal, state.currentImageBase64);
      elMetaOriginal.textContent = `Custom Upload: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      elBtnProcess.disabled = false;
      resetRoiCanvas();
    };
    reader.readAsDataURL(file);
  });

  function updateGroundTruthBadge(label) {
    if (label === 'drowsy') {
      elGtBadge.className = 'badge-drowsy';
      elGtBadge.textContent = 'DROWSY (Class 1)';
    } else if (label === 'notdrowsy') {
      elGtBadge.className = 'badge-notdrowsy';
      elGtBadge.textContent = 'NOT DROWSY (Class 0)';
    } else {
      elGtBadge.className = 'badge-neutral';
      elGtBadge.textContent = 'Custom Image (No Ground Truth)';
    }
  }

  function setImageSource(imgEl, base64Url) {
    if (!base64Url) return;
    imgEl.src = base64Url;
    imgEl.style.display = 'block';
    const placeholder = imgEl.parentElement.querySelector('.placeholder-text');
    if (placeholder) placeholder.style.display = 'none';
  }

  // =========================================================================
  // MANUAL CANVAS ROI SELECTION
  // =========================================================================
  elChkManualRoi.addEventListener('change', () => {
    state.manualRoiEnabled = elChkManualRoi.checked;
    elManualRoiPanel.classList.toggle('hidden', !state.manualRoiEnabled);
    elCanvasRoi.classList.toggle('interactive', state.manualRoiEnabled);
    if (state.manualRoiEnabled) {
      elPillEyeMode.textContent = 'Manual Canvas ROI';
      elPillEyeMode.className = 'provenance-pill pill-assump';
      initRoiCanvas();
    } else {
      elPillEyeMode.textContent = 'Haar / Anthro';
      elPillEyeMode.className = 'provenance-pill pill-approx';
      clearRoiCanvas();
    }
  });

  function initRoiCanvas() {
    const rect = elImgOriginal.getBoundingClientRect();
    elCanvasRoi.width = elImgOriginal.naturalWidth || elImgOriginal.width || 640;
    elCanvasRoi.height = elImgOriginal.naturalHeight || elImgOriginal.height || 480;
  }

  function resetRoiCanvas() {
    clearRoiCanvas();
    if (state.manualRoiEnabled) {
      initRoiCanvas();
    }
  }

  function clearRoiCanvas() {
    const ctx = elCanvasRoi.getContext('2d');
    ctx.clearRect(0, 0, elCanvasRoi.width, elCanvasRoi.height);
    state.manualRoiBox = null;
    elRoiX.textContent = '-';
    elRoiY.textContent = '-';
    elRoiW.textContent = '-';
    elRoiH.textContent = '-';
  }

  elBtnClearRoi.addEventListener('click', clearRoiCanvas);

  elCanvasRoi.addEventListener('mousedown', (e) => {
    if (!state.manualRoiEnabled) return;
    const rect = elCanvasRoi.getBoundingClientRect();
    const scaleX = elCanvasRoi.width / rect.width;
    const scaleY = elCanvasRoi.height / rect.height;

    state.isDrawingRoi = true;
    state.roiStart = {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  });

  elCanvasRoi.addEventListener('mousemove', (e) => {
    if (!state.isDrawingRoi || !state.manualRoiEnabled) return;
    const rect = elCanvasRoi.getBoundingClientRect();
    const scaleX = elCanvasRoi.width / rect.width;
    const scaleY = elCanvasRoi.height / rect.height;

    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;

    const x = Math.min(state.roiStart.x, currentX);
    const y = Math.min(state.roiStart.y, currentY);
    const w = Math.abs(currentX - state.roiStart.x);
    const h = Math.abs(currentY - state.roiStart.y);

    const ctx = elCanvasRoi.getContext('2d');
    ctx.clearRect(0, 0, elCanvasRoi.width, elCanvasRoi.height);

    // Draw selection rectangle
    ctx.strokeStyle = '#bc8cff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(188, 140, 255, 0.2)';
    ctx.fillRect(x, y, w, h);
  });

  elCanvasRoi.addEventListener('mouseup', (e) => {
    if (!state.isDrawingRoi || !state.manualRoiEnabled) return;
    state.isDrawingRoi = false;

    const rect = elCanvasRoi.getBoundingClientRect();
    const scaleX = elCanvasRoi.width / rect.width;
    const scaleY = elCanvasRoi.height / rect.height;

    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;

    const x = Math.round(Math.min(state.roiStart.x, currentX));
    const y = Math.round(Math.min(state.roiStart.y, currentY));
    const w = Math.round(Math.abs(currentX - state.roiStart.x));
    const h = Math.round(Math.abs(currentY - state.roiStart.y));

    if (w > 8 && h > 8) {
      state.manualRoiBox = { x, y, width: w, height: h };
      elRoiX.textContent = x;
      elRoiY.textContent = y;
      elRoiW.textContent = w;
      elRoiH.textContent = h;
    }
  });

  // =========================================================================
  // EXECUTE PIPELINE
  // =========================================================================
  elBtnProcess.addEventListener('click', async () => {
    if (!state.currentImageBase64 && !state.currentSample) return;

    elBtnProcess.disabled = true;
    elBtnProcess.textContent = 'Processing...';

    const payload = {
      image_base64: state.currentImageBase64,
      file_path: state.currentSample ? state.currentSample.file_path : null,
      sample_metadata: state.currentSample,
      eye_mode: state.manualRoiEnabled ? 'MANUAL_ROI' : 'APPROXIMATION_HAAR',
      manual_box: state.manualRoiBox
    };

    try {
      const res = await fetch('/api/process-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      state.lastPipelineResult = data;

      if (data.status === 'success') {
        renderPipelineSuccess(data);
      } else {
        alert(`Pipeline execution stopped at stage [${data.stage_failed}]:\n${data.message}`);
      }
    } catch (err) {
      console.error('Pipeline request failed:', err);
      alert('Error communicating with backend pipeline.');
    } finally {
      elBtnProcess.disabled = false;
      elBtnProcess.textContent = 'Execute Pipeline';
    }
  });

  function renderPipelineSuccess(data) {
    const v = data.visualizations || {};

    // 1. Preprocessing images
    setImageSource(elImgGaussian, v.gaussian_blur);
    setImageSource(elImgClahe, v.clahe);

    // 2. Face Detection
    setImageSource(elImgFace, v.face_annotated);
    if (data.face_detection && data.face_detection.box) {
      const b = data.face_detection.box;
      elMetaFace.textContent = `Face Box: [x:${b.x}, y:${b.y}, ${b.width}x${b.height}px] (${data.face_detection.status})`;
    }

    // 3. Eye Localization
    setImageSource(elImgEye, v.eye_annotated);
    if (data.eye_localization && data.eye_localization.box) {
      const b = data.eye_localization.box;
      elMetaEye.textContent = `Eye Box: [x:${b.x}, y:${b.y}, ${b.width}x${b.height}px] (${data.eye_localization.status})`;
    }

    // 4. Eye ROI
    setImageSource(elImgRoi, v.eye_roi_processed);
    if (data.roi && data.roi.dimensions) {
      elMetaRoi.textContent = `Standardized Grid: ${data.roi.dimensions[1]}x${data.roi.dimensions[0]} px (3x8 cells)`;
    }

    // 5. Improved HOG visuals
    setImageSource(elImgHogMag, v.hog_magnitude);
    setImageSource(elImgHogOri, v.hog_orientation);
    setImageSource(elImgHogShifted, v.hog_shifted);
    setImageSource(elImgHogBinary, v.hog_binary_matrix);
    setImageSource(elImgHogBarcode, v.hog_vector_plot);

    // 6. Metrics and details
    elPipelineTimeBadge.classList.remove('hidden');
    elPipelineMs.textContent = data.processing_time_ms;

    if (data.hog_features) {
      const h = data.hog_features;
      elVectorDimBadge.textContent = `Dimension: ${h.feature_dimension} bits`;
      elValFeatDim.textContent = h.feature_dimension;
      elValNumBlocks.textContent = `${h.num_blocks} blocks (${h.pairs_per_block} pairs/block)`;
      if (h.feature_vector_sample) {
        elValVectorSnippet.textContent = h.feature_vector_sample.join('');
      }
    }
  }

  // =========================================================================
  // RESET
  // =========================================================================
  elBtnReset.addEventListener('click', () => {
    state.currentSample = null;
    state.currentFile = null;
    state.currentImageBase64 = null;
    state.lastPipelineResult = null;
    clearRoiCanvas();
    updateGroundTruthBadge(null);

    // Clear previews
    [
      elImgOriginal, elImgGaussian, elImgClahe, elImgFace, elImgEye, elImgRoi,
      elImgHogMag, elImgHogOri, elImgHogShifted, elImgHogBinary, elImgHogBarcode
    ].forEach(img => {
      img.src = '';
      img.style.display = 'none';
      const placeholder = img.parentElement.querySelector('.placeholder-text');
      if (placeholder) placeholder.style.display = 'block';
    });

    elSampleSelect.value = '';
    elFileUpload.value = '';
    elBtnProcess.disabled = true;
    elPipelineTimeBadge.classList.add('hidden');
    elExportStatusBox.classList.add('hidden');
    elMetaOriginal.textContent = 'Resolution: -';
    elMetaFace.textContent = 'AdaBoost cascade substitute';
    elMetaEye.textContent = 'ASM replacement status';
    elMetaRoi.textContent = 'Grid: Divisible by 8x8 cells';
    elVectorDimBadge.textContent = 'Dimension: 0';
    elValFeatDim.textContent = '0';
    elValNumBlocks.textContent = '0';
    elValVectorSnippet.textContent = '-';
  });

  // =========================================================================
  // FEATURE EXPORT
  // =========================================================================
  elBtnExportCsv.addEventListener('click', async () => {
    elBtnExportCsv.disabled = true;
    elBtnExportCsv.textContent = 'Exporting...';

    try {
      const res = await fetch('/api/feature-export', { method: 'POST' });
      const data = await res.json();

      elExportStatusBox.classList.remove('hidden');
      if (data.status === 'success') {
        elExportMsg.innerHTML = `
          <strong>Export Complete!</strong><br>
          Features CSV: <code>${data.features_csv}</code><br>
          Metadata CSV: <code>${data.metadata_csv}</code><br>
          Audit Report: <code>${data.report_json}</code><br>
          Total Exported Samples: <strong>${data.total_success}</strong> (Dimension: ${data.dimension} bits).
        `;
      } else {
        elExportMsg.textContent = `Warning: ${data.message} (Run pipeline on samples first or scan dataset).`;
      }
    } catch (err) {
      console.error('Export failed:', err);
      elExportStatusBox.classList.remove('hidden');
      elExportMsg.textContent = 'Failed to trigger feature export on backend.';
    } finally {
      elBtnExportCsv.disabled = false;
      elBtnExportCsv.textContent = 'Export Current & Batch to CSV';
    }
  });

  elBtnScanDataset.addEventListener('click', async () => {
    elBtnScanDataset.disabled = true;
    elBtnScanDataset.textContent = 'Scanning...';
    await loadDatasetSummary();
    elBtnScanDataset.disabled = false;
    elBtnScanDataset.textContent = 'Rescan Dataset';
  });

  // Start initialization
  checkBackendHealth();
  loadDatasetSummary();
});
