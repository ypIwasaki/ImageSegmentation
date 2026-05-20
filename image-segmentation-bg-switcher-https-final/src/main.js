import './styles.css';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import defaultBackgroundUrl from '../../images/AdobeStock_310895879.optimized.webp';

const video = document.querySelector('#cameraVideo');
const canvas = document.querySelector('#outputCanvas');
const ctx = canvas.getContext('2d', { alpha: true, willReadFrequently: false });

const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const modeSelect = document.querySelector('#modeSelect');
const colorInput = document.querySelector('#colorInput');
const imageInput = document.querySelector('#imageInput');
const mirrorInput = document.querySelector('#mirrorInput');
const modelSelect = document.querySelector('#modelSelect');
const computeDeviceSelect = document.querySelector('#computeDeviceSelect');
const loadDefaultImageButton = document.querySelector('#loadDefaultImageButton');
const resetComparisonButton = document.querySelector('#resetComparisonButton');
const maskStyleSelect = document.querySelector('#maskStyleSelect');
const maskAlphaLowInput = document.querySelector('#maskAlphaLowInput');
const maskAlphaLowValue = document.querySelector('#maskAlphaLowValue');
const maskAlphaHighInput = document.querySelector('#maskAlphaHighInput');
const maskAlphaHighValue = document.querySelector('#maskAlphaHighValue');
const maskMorphInput = document.querySelector('#maskMorphInput');
const maskMorphValue = document.querySelector('#maskMorphValue');
const maskFeatherInput = document.querySelector('#maskFeatherInput');
const maskFeatherValue = document.querySelector('#maskFeatherValue');
const temporalSmoothingInput = document.querySelector('#temporalSmoothingInput');
const temporalSmoothingValue = document.querySelector('#temporalSmoothingValue');
const statusEl = document.querySelector('#status');
const debugEl = document.querySelector('#debugLog');

const colorPickerContainer = document.querySelector('#colorPickerContainer');
const imageInputContainer = document.querySelector('#imageInputContainer');

const SEGMENT_INTERVAL_MS = 33;
const HARD_MASK_THRESHOLD = 0.5;
const ALPHA_LUT_SIZE = 1024;
const DEFAULT_BACKGROUND_NAME = 'AdobeStock_310895879.optimized.webp';

let stream = null;
let segmenter = null;
let running = false;
let rafId = 0;
let backgroundImage = null;
let currentBgUrl = null;
let backgroundCanvas = null;
let backgroundCtx = null;
let backgroundCacheKey = '';
let segmentBusy = false;
let lastSegmentAt = 0;
let frameTimestamp = 0;

let maskImageData = null;
let maskCanvas = null;
let maskCtx = null;
let filteredMaskCanvas = null;
let filteredMaskCtx = null;
let featherWorkCanvas = null;
let featherWorkCtx = null;
let activeMaskCanvas = null;
let temporalAlphaBuffer = null;
let temporalAlphaReady = false;
let alphaMaskBuffer = null;
let alphaMaskScratchBuffer = null;
let lastConfidenceMask = null;
let lastMaskWidth = 0;
let lastMaskHeight = 0;
let maskAlphaLut = null;
let maskAlphaLutDirty = true;

const maskSettings = {
  style: 'hard',
  alphaLow: 0.35,
  alphaHigh: 0.75,
  morphAmount: 0,
  featherPx: 0,
  temporalSmoothing: 0,
};

let personCanvas = null;
let personCtx = null;

function log(message, data) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}${data ? ` ${formatData(data)}` : ''}`;
  console.log(message, data ?? '');
  debugEl.textContent = `${line}\n${debugEl.textContent}`.slice(0, 6000);
}

function formatData(data) {
  if (data instanceof Error) return `${data.name}: ${data.message}`;
  try { return JSON.stringify(data); } catch { return String(data); }
}

function setStatus(message) {
  statusEl.textContent = message;
}

window.addEventListener('error', (event) => {
  const target = event.target;
  log('Global error', {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    target: target?.src || target?.href || target?.tagName,
  });
});

window.addEventListener('unhandledrejection', (event) => {
  log('Unhandled rejection', event.reason);
});

function assertBrowserSupport() {
  if (!window.isSecureContext) {
    throw new Error('Secure Contextではありません。https://localhost:5173 で開いてください。');
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('navigator.mediaDevices.getUserMedia が使えません。ブラウザ/URL/権限を確認してください。');
  }
}

function isGeneralSelected() {
  return Number(modelSelect.value) === 1;
}

function isCpuSelected() {
  return computeDeviceSelect.value === 'cpu';
}

function getSelectedModelLabel() {
  return isGeneralSelected() ? 'General (高精度)' : 'Landscape (軽量 / 高速)';
}

function getSelectedComputeDeviceLabel() {
  return isCpuSelected() ? 'CPU' : 'GPU';
}

async function waitForVideoReady() {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) return;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Video metadata timeout')), 8000);
    const done = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        clearTimeout(timer);
        cleanup();
        resolve();
      }
    };
    const cleanup = () => {
      video.removeEventListener('loadedmetadata', done);
      video.removeEventListener('canplay', done);
      video.removeEventListener('error', fail);
    };
    const fail = () => {
      clearTimeout(timer);
      cleanup();
      reject(new Error('Video element error'));
    };
    video.addEventListener('loadedmetadata', done);
    video.addEventListener('canplay', done);
    video.addEventListener('error', fail);
    done();
  });
}

function resizeCanvasToVideo() {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    backgroundCacheKey = '';
    log('Canvas resized', { w, h });
  }
}

async function startCameraOnly() {
  assertBrowserSupport();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  await waitForVideoReady();
  resizeCanvasToVideo();
  log('Camera started', { width: video.videoWidth, height: video.videoHeight });
}

function closeSegmenter() {
  if (segmenter) {
    try { segmenter.close(); } catch {}
  }
  segmenter = null;
}

async function initSegmenter() {
  if (!ImageSegmenter || !FilesetResolver) {
    throw new Error('MediaPipe Tasks Vision library is not loaded.');
  }

  closeSegmenter();

  log('Initializing modern ImageSegmenter...');
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm'
  );

  const modelAssetPath = isGeneralSelected()
    ? 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'
    : 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite';

  log(`Loading model: ${getSelectedModelLabel()}`, { delegate: getSelectedComputeDeviceLabel() });

  segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath,
      delegate: isCpuSelected() ? 'CPU' : 'GPU',
    },
    runningMode: 'VIDEO',
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });

  frameTimestamp = 0;
  segmenter.segmentForVideo(video, frameTimestamp, (result) => {
    if (result?.confidenceMasks?.length) {
      const mask = result.confidenceMasks[0];
      updateMaskCanvasFromArray(mask.getAsFloat32Array(), mask.width, mask.height, true);
    }
  });
  frameTimestamp += 33;

  log('ImageSegmenter initialized successfully.', {
    model: getSelectedModelLabel(),
    delegate: getSelectedComputeDeviceLabel(),
  });
}

function drawMirroredOnCtx(drawFn, targetCtx, w) {
  targetCtx.save();
  if (mirrorInput.checked) {
    targetCtx.translate(w, 0);
    targetCtx.scale(-1, 1);
  }
  drawFn();
  targetCtx.restore();
}

function drawCoverImageOnCtx(source, targetCtx, cw, ch) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  if (!sw || !sh) return;
  const scale = Math.max(cw / sw, ch / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;
  targetCtx.drawImage(source, dx, dy, dw, dh);
}

function drawRaw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawMirroredOnCtx(() => drawCoverImageOnCtx(video, ctx, canvas.width, canvas.height), ctx, canvas.width);
}

function getBackgroundCacheCanvas() {
  if (!backgroundCanvas) {
    backgroundCanvas = document.createElement('canvas');
    backgroundCtx = backgroundCanvas.getContext('2d');
  }
  return backgroundCanvas;
}

function drawCachedBackground() {
  if (!backgroundImage || !canvas.width || !canvas.height) return;

  const cacheCanvas = getBackgroundCacheCanvas();
  const cacheKey = `${canvas.width}x${canvas.height}:${backgroundImage.naturalWidth}x${backgroundImage.naturalHeight}`;
  if (backgroundCacheKey !== cacheKey) {
    cacheCanvas.width = canvas.width;
    cacheCanvas.height = canvas.height;
    backgroundCtx.clearRect(0, 0, cacheCanvas.width, cacheCanvas.height);
    drawCoverImageOnCtx(backgroundImage, backgroundCtx, cacheCanvas.width, cacheCanvas.height);
    backgroundCacheKey = cacheKey;
  }

  ctx.drawImage(cacheCanvas, 0, 0);
}

function drawBackground(mode) {
  if (mode === 'color') {
    ctx.fillStyle = colorInput.value;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (mode === 'image' && backgroundImage) {
    drawCachedBackground();
    return;
  }
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getMaskAlphaLow() {
  return maskSettings.alphaLow;
}

function getMaskAlphaHigh() {
  return maskSettings.alphaHigh;
}

function updateMaskAlphaValues() {
  maskAlphaLowValue.textContent = getMaskAlphaLow().toFixed(2);
  if (Number(maskAlphaHighInput.value) < getMaskAlphaHigh()) {
    maskAlphaHighInput.value = getMaskAlphaHigh().toFixed(2);
  }
  maskAlphaHighValue.textContent = getMaskAlphaHigh().toFixed(2);
}

function getMaskMorphAmount() {
  return maskSettings.morphAmount;
}

function updateMaskMorphValue() {
  const amount = getMaskMorphAmount();
  maskMorphValue.textContent = `${amount > 0 ? '+' : ''}${amount}px`;
}

function getMaskFeatherPx() {
  return maskSettings.featherPx;
}

function updateMaskFeatherValue() {
  maskFeatherValue.textContent = `${getMaskFeatherPx().toFixed(1)}px`;
}

function getTemporalSmoothing() {
  return maskSettings.temporalSmoothing;
}

function updateTemporalSmoothingValue() {
  temporalSmoothingValue.textContent = `${Math.round(getTemporalSmoothing() * 100)}%`;
}

function syncMaskSettingsFromUI() {
  const nextLow = Number(maskAlphaLowInput.value);
  const nextHigh = Math.max(Number(maskAlphaHighInput.value), nextLow + 0.01);
  const nextStyle = maskStyleSelect.value;
  const nextMorphAmount = Number(maskMorphInput.value);
  const nextFeatherPx = Number(maskFeatherInput.value);
  const nextTemporalSmoothing = Number(temporalSmoothingInput.value);

  const lutChanged = (
    maskSettings.style !== nextStyle ||
    maskSettings.alphaLow !== nextLow ||
    maskSettings.alphaHigh !== nextHigh
  );

  maskSettings.style = nextStyle;
  maskSettings.alphaLow = nextLow;
  maskSettings.alphaHigh = nextHigh;
  maskSettings.morphAmount = nextMorphAmount;
  maskSettings.featherPx = nextFeatherPx;
  maskSettings.temporalSmoothing = nextTemporalSmoothing;

  if (lutChanged) {
    maskAlphaLutDirty = true;
  }
}

function ensureMaskAlphaLut() {
  if (!maskAlphaLut) {
    maskAlphaLut = new Uint8ClampedArray(ALPHA_LUT_SIZE + 1);
    maskAlphaLutDirty = true;
  }
  if (!maskAlphaLutDirty) return;

  for (let i = 0; i <= ALPHA_LUT_SIZE; i++) {
    const confidence = i / ALPHA_LUT_SIZE;
    let alpha;
    if (maskSettings.style === 'hard') {
      alpha = confidence >= HARD_MASK_THRESHOLD ? 255 : 0;
    } else {
      const t = Math.max(
        0,
        Math.min(1, (confidence - maskSettings.alphaLow) / Math.max(maskSettings.alphaHigh - maskSettings.alphaLow, 0.0001))
      );
      const smooth = t * t * (3 - 2 * t);
      alpha = Math.round(smooth * 255);
    }
    maskAlphaLut[i] = alpha;
  }

  maskAlphaLutDirty = false;
}

function confidenceToAlpha(confidence) {
  const index = Math.max(0, Math.min(ALPHA_LUT_SIZE, (confidence * ALPHA_LUT_SIZE) | 0));
  return maskAlphaLut[index];
}

function resetTemporalSmoothing() {
  temporalAlphaBuffer = null;
  temporalAlphaReady = false;
}

function clearMaskState() {
  maskImageData = null;
  maskCanvas = null;
  maskCtx = null;
  filteredMaskCanvas = null;
  filteredMaskCtx = null;
  featherWorkCanvas = null;
  featherWorkCtx = null;
  activeMaskCanvas = null;
  alphaMaskBuffer = null;
  alphaMaskScratchBuffer = null;
  lastConfidenceMask = null;
  lastMaskWidth = 0;
  lastMaskHeight = 0;
  resetTemporalSmoothing();
}

function revokeCurrentBackgroundUrl() {
  if (currentBgUrl) {
    URL.revokeObjectURL(currentBgUrl);
    currentBgUrl = null;
  }
}

function ensureMaskCanvases(w, h) {
  if (!maskCanvas || maskCanvas.width !== w || maskCanvas.height !== h) {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    maskCtx = maskCanvas.getContext('2d');
    maskImageData = maskCtx.createImageData(w, h);
    filteredMaskCanvas = null;
    filteredMaskCtx = null;
    activeMaskCanvas = maskCanvas;
    resetTemporalSmoothing();
  }
}

function ensureFilteredMaskCanvas(w, h) {
  if (!filteredMaskCanvas || filteredMaskCanvas.width !== w || filteredMaskCanvas.height !== h) {
    filteredMaskCanvas = document.createElement('canvas');
    filteredMaskCanvas.width = w;
    filteredMaskCanvas.height = h;
    filteredMaskCtx = filteredMaskCanvas.getContext('2d');
  }
}

function ensureFeatherWorkCanvas(w, h) {
  if (!featherWorkCanvas || featherWorkCanvas.width !== w || featherWorkCanvas.height !== h) {
    featherWorkCanvas = document.createElement('canvas');
    featherWorkCanvas.width = w;
    featherWorkCanvas.height = h;
    featherWorkCtx = featherWorkCanvas.getContext('2d');
  }
}

function ensureAlphaBuffers(length) {
  if (!alphaMaskBuffer || alphaMaskBuffer.length !== length) {
    alphaMaskBuffer = new Uint8ClampedArray(length);
    alphaMaskScratchBuffer = new Uint8ClampedArray(length);
  }
}

function applyBackgroundImage(img, sourceLabel) {
  backgroundImage = img;
  backgroundCacheKey = '';
  modeSelect.value = 'image';
  updateVisibility();
  log('Background image loaded', {
    source: sourceLabel,
    width: img.naturalWidth,
    height: img.naturalHeight,
  });
}

function loadBackgroundImage(url, sourceLabel) {
  const img = new Image();
  img.onload = () => applyBackgroundImage(img, sourceLabel);
  img.onerror = () => {
    log('Background image load failed', { source: sourceLabel, url });
  };
  img.src = url;
}

function getScaledMaskFeatherPx(w, h) {
  const scaleX = w / Math.max(canvas.width, 1);
  const scaleY = h / Math.max(canvas.height, 1);
  return Math.max(0.01, getMaskFeatherPx() * Math.max(scaleX, scaleY));
}

function applyMaskMorphologyInPlace(buffer, scratch, w, h, amount) {
  const radius = Math.abs(amount);
  if (radius === 0) return;

  const useDilation = amount > 0;

  for (let y = 0; y < h; y++) {
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      const xStart = Math.max(0, x - radius);
      const xEnd = Math.min(w - 1, x + radius);
      let value = useDilation ? 0 : 255;

      for (let xx = xStart; xx <= xEnd; xx++) {
        const sample = buffer[rowOffset + xx];
        if (useDilation) {
          if (sample > value) value = sample;
        } else if (sample < value) {
          value = sample;
        }
      }

      scratch[rowOffset + x] = value;
    }
  }

  for (let y = 0; y < h; y++) {
    const yStart = Math.max(0, y - radius);
    const yEnd = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      let value = useDilation ? 0 : 255;

      for (let yy = yStart; yy <= yEnd; yy++) {
        const sample = scratch[yy * w + x];
        if (useDilation) {
          if (sample > value) value = sample;
        } else if (sample < value) {
          value = sample;
        }
      }

      buffer[y * w + x] = value;
    }
  }
}

function renderMaskCanvasFromAlphaBuffer(w, h) {
  if (!maskImageData || !alphaMaskBuffer) return;

  const data = maskImageData.data;
  for (let i = 0; i < alphaMaskBuffer.length; i++) {
    const idx = i * 4;
    data[idx] = 255;
    data[idx + 1] = 255;
    data[idx + 2] = 255;
    data[idx + 3] = alphaMaskBuffer[i];
  }

  maskCtx.putImageData(maskImageData, 0, 0);

  if (getMaskFeatherPx() <= 0) {
    activeMaskCanvas = maskCanvas;
    return;
  }

  const featherPx = getScaledMaskFeatherPx(w, h);
  const useDownsampledFeather = w * h >= 65536;
  ensureFilteredMaskCanvas(w, h);

  if (useDownsampledFeather) {
    const workScale = 0.5;
    const workWidth = Math.max(1, Math.round(w * workScale));
    const workHeight = Math.max(1, Math.round(h * workScale));
    ensureFeatherWorkCanvas(workWidth, workHeight);

    featherWorkCtx.clearRect(0, 0, workWidth, workHeight);
    featherWorkCtx.save();
    featherWorkCtx.filter = `blur(${Math.max(0.01, featherPx * workScale)}px)`;
    featherWorkCtx.drawImage(maskCanvas, 0, 0, workWidth, workHeight);
    featherWorkCtx.restore();

    filteredMaskCtx.clearRect(0, 0, w, h);
    filteredMaskCtx.imageSmoothingEnabled = true;
    filteredMaskCtx.drawImage(featherWorkCanvas, 0, 0, w, h);
  } else {
    filteredMaskCtx.clearRect(0, 0, w, h);
    filteredMaskCtx.save();
    filteredMaskCtx.filter = `blur(${featherPx}px)`;
    filteredMaskCtx.drawImage(maskCanvas, 0, 0);
    filteredMaskCtx.restore();
  }

  activeMaskCanvas = filteredMaskCanvas;
}

function updateMaskCanvasFromArray(maskData, w, h, storeMaskData = false) {
  ensureMaskCanvases(w, h);
  ensureAlphaBuffers(maskData.length);
  ensureMaskAlphaLut();

  if (storeMaskData) {
    if (!lastConfidenceMask || lastConfidenceMask.length !== maskData.length) {
      lastConfidenceMask = new Float32Array(maskData.length);
    }
    lastConfidenceMask.set(maskData);
    lastMaskWidth = w;
    lastMaskHeight = h;
  }

  const sourceData = storeMaskData ? lastConfidenceMask : maskData;
  const temporalSmoothing = maskSettings.temporalSmoothing;
  if (temporalSmoothing > 0) {
    if (!temporalAlphaBuffer || temporalAlphaBuffer.length !== sourceData.length) {
      temporalAlphaBuffer = new Float32Array(sourceData.length);
      temporalAlphaReady = false;
    }
  } else {
    resetTemporalSmoothing();
  }

  for (let i = 0; i < sourceData.length; i++) {
    let alpha = confidenceToAlpha(sourceData[i]);
    if (temporalSmoothing > 0) {
      if (!temporalAlphaReady) {
        temporalAlphaBuffer[i] = alpha;
      } else {
        temporalAlphaBuffer[i] = temporalAlphaBuffer[i] * temporalSmoothing + alpha * (1 - temporalSmoothing);
      }
      alpha = Math.round(temporalAlphaBuffer[i]);
    }
    alphaMaskBuffer[i] = alpha;
  }

  if (temporalSmoothing > 0) {
    temporalAlphaReady = true;
  }

  if (maskSettings.morphAmount !== 0) {
    applyMaskMorphologyInPlace(alphaMaskBuffer, alphaMaskScratchBuffer, w, h, maskSettings.morphAmount);
  }

  renderMaskCanvasFromAlphaBuffer(w, h);
  return activeMaskCanvas;
}

function refreshMaskCanvas() {
  if (!lastConfidenceMask || !lastMaskWidth || !lastMaskHeight) return;
  updateMaskCanvasFromArray(lastConfidenceMask, lastMaskWidth, lastMaskHeight, false);
}

function rerenderMaskCanvas() {
  if (!alphaMaskBuffer || !lastMaskWidth || !lastMaskHeight) return;
  renderMaskCanvasFromAlphaBuffer(lastMaskWidth, lastMaskHeight);
}

function getPersonCanvas(w, h) {
  if (!personCanvas || personCanvas.width !== w || personCanvas.height !== h) {
    personCanvas = document.createElement('canvas');
    personCanvas.width = w;
    personCanvas.height = h;
    personCtx = personCanvas.getContext('2d');
  }
  return personCanvas;
}

function drawMaskOnCtx(targetCtx, w, h) {
  const sourceMaskCanvas = activeMaskCanvas || maskCanvas;
  if (!sourceMaskCanvas) return;
  drawMirroredOnCtx(() => drawCoverImageOnCtx(sourceMaskCanvas, targetCtx, w, h), targetCtx, w);
}

function drawPersonCutout() {
  const w = canvas.width;
  const h = canvas.height;
  getPersonCanvas(w, h);

  personCtx.clearRect(0, 0, w, h);
  personCtx.save();
  drawMirroredOnCtx(() => drawCoverImageOnCtx(video, personCtx, w, h), personCtx, w);
  personCtx.globalCompositeOperation = 'destination-in';
  personCtx.filter = 'none';
  drawMaskOnCtx(personCtx, w, h);
  personCtx.restore();
}

function drawSegmented() {
  if (modeSelect.value === 'raw' || !activeMaskCanvas) {
    drawRaw();
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground(modeSelect.value);
  drawPersonCutout();
  ctx.drawImage(personCanvas, 0, 0);
}

function requestSegmentationIfNeeded(now) {
  if (!segmenter || segmentBusy) return;
  if (now - lastSegmentAt < SEGMENT_INTERVAL_MS) return;
  lastSegmentAt = now;
  segmentBusy = true;

  try {
    segmenter.segmentForVideo(video, frameTimestamp, (result) => {
      if (result?.confidenceMasks?.length) {
        const mask = result.confidenceMasks[0];
        updateMaskCanvasFromArray(mask.getAsFloat32Array(), mask.width, mask.height, true);
      }
    });
    frameTimestamp += 33;
  } catch (error) {
    log('Segmentation failed. Falling back to raw preview.', error);
    setStatus('背景分離に失敗しました。カメラ映像のみ表示しています。');
    closeSegmenter();
  } finally {
    segmentBusy = false;
  }
}

function renderLoop(now = performance.now()) {
  if (!running) return;
  resizeCanvasToVideo();
  requestSegmentationIfNeeded(now);
  drawSegmented();
  rafId = requestAnimationFrame(renderLoop);
}

async function start() {
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus('カメラ起動中...');

  try {
    await startCameraOnly();
    running = true;
    setStatus('カメラ表示中。背景分離を初期化しています...');
    renderLoop();

    try {
      await initSegmenter();
      setStatus(`背景切り替え有効。推論デバイス: ${getSelectedComputeDeviceLabel()}`);
    } catch (error) {
      log('Segmentation initialization failed', error);
      setStatus(`${getSelectedModelLabel()} の初期化に失敗しました。カメラ映像のみ表示します。\n${error.message}`);
    }
  } catch (error) {
    log('Camera start failed', error);
    setStatus(`カメラ起動失敗: ${error.name || 'Error'}\n${error.message}`);
    startButton.disabled = false;
    stopButton.disabled = true;
  }
}

function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  stream = null;
  video.srcObject = null;
  closeSegmenter();
  frameTimestamp = 0;
  clearMaskState();
  personCanvas = null;
  personCtx = null;
  segmentBusy = false;
  lastSegmentAt = 0;
  revokeCurrentBackgroundUrl();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus('停止しました');
}

function updateVisibility() {
  colorPickerContainer.style.display = modeSelect.value === 'color' ? 'grid' : 'none';
  imageInputContainer.style.display = modeSelect.value === 'image' ? 'flex' : 'none';
}

function resetTemporalAndRefreshMask() {
  resetTemporalSmoothing();
  refreshMaskCanvas();
}

function applyComparisonBaseline() {
  modeSelect.value = 'raw';
  maskStyleSelect.value = 'hard';
  maskAlphaLowInput.value = '0.35';
  maskAlphaHighInput.value = '0.75';
  maskMorphInput.value = '0';
  maskFeatherInput.value = '0';
  temporalSmoothingInput.value = '0';
  syncMaskSettingsFromUI();

  updateVisibility();
  updateMaskAlphaValues();
  updateMaskMorphValue();
  updateMaskFeatherValue();
  updateTemporalSmoothingValue();

  backgroundImage = null;
  backgroundCacheKey = '';
  revokeCurrentBackgroundUrl();

  clearMaskState();
  log('Comparison baseline applied');
  setStatus('比較用設定に戻しました');
}

updateVisibility();
syncMaskSettingsFromUI();
updateMaskAlphaValues();
updateMaskMorphValue();
updateMaskFeatherValue();
updateTemporalSmoothingValue();

modeSelect.addEventListener('change', updateVisibility);

imageInput.addEventListener('change', () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  revokeCurrentBackgroundUrl();
  currentBgUrl = URL.createObjectURL(file);
  loadBackgroundImage(currentBgUrl, file.name);
});

loadDefaultImageButton.addEventListener('click', () => {
  loadBackgroundImage(defaultBackgroundUrl, DEFAULT_BACKGROUND_NAME);
});

resetComparisonButton.addEventListener('click', () => {
  applyComparisonBaseline();
});

modelSelect.addEventListener('change', async () => {
  if (!running) return;
  setStatus('モデル設定を変更しました。再初期化しています...');
  try {
    clearMaskState();
    await initSegmenter();
    setStatus(`背景切り替え有効。推論デバイス: ${getSelectedComputeDeviceLabel()}`);
  } catch (error) {
    log('Model reinitialization failed', error);
    setStatus('モデル再初期化に失敗しました。カメラ映像のみ表示します。');
  }
});

computeDeviceSelect.addEventListener('change', async () => {
  log('Compute device changed', { device: getSelectedComputeDeviceLabel() });
  if (!running) return;
  setStatus('推論デバイスを変更しました。再初期化しています...');
  try {
    clearMaskState();
    await initSegmenter();
    setStatus(`背景切り替え有効。推論デバイス: ${getSelectedComputeDeviceLabel()}`);
  } catch (error) {
    log('Compute device reinitialization failed', error);
    setStatus('推論デバイスの再初期化に失敗しました。カメラ映像のみ表示します。');
  }
});

maskStyleSelect.addEventListener('change', () => {
  syncMaskSettingsFromUI();
  log('Mask style changed', { style: maskStyleSelect.value });
  resetTemporalAndRefreshMask();
});

maskAlphaLowInput.addEventListener('input', () => {
  syncMaskSettingsFromUI();
  updateMaskAlphaValues();
  log('Mask alpha low changed', { value: getMaskAlphaLow() });
  resetTemporalAndRefreshMask();
});

maskAlphaHighInput.addEventListener('input', () => {
  syncMaskSettingsFromUI();
  updateMaskAlphaValues();
  log('Mask alpha high changed', { value: getMaskAlphaHigh() });
  resetTemporalAndRefreshMask();
});

maskMorphInput.addEventListener('input', () => {
  syncMaskSettingsFromUI();
  updateMaskMorphValue();
  log('Mask morphology changed', { amount: getMaskMorphAmount() });
  resetTemporalAndRefreshMask();
});

maskFeatherInput.addEventListener('input', () => {
  syncMaskSettingsFromUI();
  updateMaskFeatherValue();
  log('Mask feather changed', { px: getMaskFeatherPx() });
  rerenderMaskCanvas();
});

temporalSmoothingInput.addEventListener('input', () => {
  syncMaskSettingsFromUI();
  updateTemporalSmoothingValue();
  log('Temporal smoothing changed', { ratio: getTemporalSmoothing() });
  resetTemporalAndRefreshMask();
});

startButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);

setStatus(`待機中\nSecureContext: ${window.isSecureContext}\nURL: ${location.href}`);
