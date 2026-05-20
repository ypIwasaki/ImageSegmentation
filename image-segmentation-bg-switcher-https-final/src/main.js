import './styles.css';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import defaultBackgroundUrl from '../../images/AdobeStock_310895879.jpeg';

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
const maskStyleSelect = document.querySelector('#maskStyleSelect');
const maskFeatherInput = document.querySelector('#maskFeatherInput');
const maskFeatherValue = document.querySelector('#maskFeatherValue');
const temporalSmoothingInput = document.querySelector('#temporalSmoothingInput');
const temporalSmoothingValue = document.querySelector('#temporalSmoothingValue');
const statusEl = document.querySelector('#status');
const debugEl = document.querySelector('#debugLog');

let stream = null;
let selfieSegmentation = null;
let running = false;
let rafId = 0;
let backgroundImage = null;
let currentBgUrl = null;
let segmentBusy = false;
let lastSegmentAt = 0;
let frameTimestamp = 0; // MediaPipe用の厳密な単調増加タイムスタンプカウンター
const SEGMENT_INTERVAL_MS = 33;
const HARD_MASK_THRESHOLD = 0.5;
const MASK_ALPHA_LOW = 0.35;
const MASK_ALPHA_HIGH = 0.75;
const DEFAULT_BACKGROUND_NAME = 'AdobeStock_310895879.jpeg';

// 高速レンダリング用のオフスクリーンキャンバス キャッシュ（マスク用 & 人物切り抜き用）
let maskImageData = null;
let maskCanvas = null;
let maskCtx = null;
let temporalAlphaBuffer = null;
let temporalAlphaReady = false;

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

async function initSelfieSegmentation() {
  if (!ImageSegmenter || !FilesetResolver) {
    throw new Error('MediaPipe Tasks Vision library is not loaded.');
  }

  log('Initializing modern ImageSegmenter...');
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
  );

  const isGeneral = Number(modelSelect.value) === 1;
  const modelAssetPath = isGeneral
    ? "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    : "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite";

  log(`Loading model: ${isGeneral ? 'General (高精度)' : 'Landscape (軽量)'}`);

  const instance = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: modelAssetPath,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    outputCategoryMask: false,
    outputConfidenceMasks: true
  });

  // ウォームアップ推論（callback経由で結果を即時安全キャッシュ）
  frameTimestamp = 0;
  instance.segmentForVideo(video, frameTimestamp, (result) => {
    if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
      getMaskCanvas(result.confidenceMasks[0]);
    }
  });
  frameTimestamp += 33;

  selfieSegmentation = instance;
  log('ImageSegmenter (Tasks Vision) initialized successfully.');
}

function drawMirroredOnCtx(drawFn, targetCtx, w) {
  const mirror = mirrorInput.checked;
  targetCtx.save();
  if (mirror) {
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

function drawBackground(mode) {
  if (mode === 'color') {
    ctx.fillStyle = colorInput.value;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (mode === 'image' && backgroundImage) {
    drawCoverImageOnCtx(backgroundImage, ctx, canvas.width, canvas.height);
    return;
  }
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function confidenceToAlpha(confidence) {
  if (maskStyleSelect.value === 'hard') {
    return confidence >= HARD_MASK_THRESHOLD ? 255 : 0;
  }
  const t = Math.max(0, Math.min(1, (confidence - MASK_ALPHA_LOW) / (MASK_ALPHA_HIGH - MASK_ALPHA_LOW)));
  const smooth = t * t * (3 - 2 * t);
  return Math.round(smooth * 255);
}

function getMaskFeatherPx() {
  return Number(maskFeatherInput.value);
}

function updateMaskFeatherValue() {
  maskFeatherValue.textContent = `${getMaskFeatherPx().toFixed(1)}px`;
}

function getTemporalSmoothing() {
  return Number(temporalSmoothingInput.value);
}

function updateTemporalSmoothingValue() {
  temporalSmoothingValue.textContent = `${Math.round(getTemporalSmoothing() * 100)}%`;
}

function resetTemporalSmoothing() {
  temporalAlphaBuffer = null;
  temporalAlphaReady = false;
}

function applyBackgroundImage(img, sourceLabel) {
  backgroundImage = img;
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

// Float32の信頼度マスクから、連続的なアルファを持つソフトマスクを生成
function getMaskCanvas(mask) {
  const w = mask.width;
  const h = mask.height;
  
  if (!maskCanvas || maskCanvas.width !== w || maskCanvas.height !== h) {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    maskCtx = maskCanvas.getContext('2d');
    maskImageData = maskCtx.createImageData(w, h);
    resetTemporalSmoothing();
  }
  
  const data = maskImageData.data;
  const maskData = mask.getAsFloat32Array();
  const temporalSmoothing = getTemporalSmoothing();
  if (temporalSmoothing > 0 && (!temporalAlphaBuffer || temporalAlphaBuffer.length !== maskData.length)) {
    temporalAlphaBuffer = new Float32Array(maskData.length);
    temporalAlphaReady = false;
  }
  
  for (let i = 0; i < maskData.length; i++) {
    const confidence = maskData[i];
    let alpha = confidenceToAlpha(confidence);
    if (temporalSmoothing > 0) {
      if (!temporalAlphaReady) {
        temporalAlphaBuffer[i] = alpha;
      } else {
        temporalAlphaBuffer[i] = temporalAlphaBuffer[i] * temporalSmoothing + alpha * (1 - temporalSmoothing);
      }
      alpha = Math.round(temporalAlphaBuffer[i]);
    }
    const idx = i * 4;
    data[idx] = 255;     // R
    data[idx + 1] = 255; // G
    data[idx + 2] = 255; // B
    data[idx + 3] = alpha; // A (人物信頼度に応じて連続的に変化)
  }
  if (temporalSmoothing > 0) {
    temporalAlphaReady = true;
  }
  
  maskCtx.putImageData(maskImageData, 0, 0);
  return maskCanvas;
}

// オフスクリーンキャンバスの初期化と取得
function getPersonCanvas(w, h) {
  if (!personCanvas || personCanvas.width !== w || personCanvas.height !== h) {
    personCanvas = document.createElement('canvas');
    personCanvas.width = w;
    personCanvas.height = h;
    personCtx = personCanvas.getContext('2d');
  }
  return personCanvas;
}

// メインキャンバスの背景画を巻き添えにせず、人物だけを独立して切り抜いて生成する
function drawPersonCutout() {
  const w = canvas.width;
  const h = canvas.height;
  const pCanvas = getPersonCanvas(w, h);

  // オフスクリーンを真っさらにクリア
  personCtx.clearRect(0, 0, w, h);

  // 1. 人物単体カメラ映像を描画
  personCtx.save();
  drawMirroredOnCtx(() => drawCoverImageOnCtx(video, personCtx, w, h), personCtx, w);

  // 2. マスクのアルファを用いてくり抜く（destination-in）
  personCtx.globalCompositeOperation = 'destination-in';
  personCtx.filter = getMaskFeatherPx() > 0 ? `blur(${getMaskFeatherPx()}px)` : 'none';
  drawMirroredOnCtx(() => drawCoverImageOnCtx(maskCanvas, personCtx, w, h), personCtx, w);
  personCtx.restore();
}

function drawSegmented() {
  const mode = modeSelect.value;
  if (mode === 'raw' || !maskCanvas) {
    drawRaw();
    return;
  }

  // 1. メインキャンバスをクリア
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 2. 背景（カラー または 画像）を描画
  drawBackground(mode);

  // 3. オフスクリーンキャンバスで人物を切り抜く
  drawPersonCutout();

  // 4. 背景の上に人物画像を合成
  ctx.drawImage(personCanvas, 0, 0);
}

function requestSegmentationIfNeeded(now) {
  if (!selfieSegmentation || segmentBusy) return;
  if (now - lastSegmentAt < SEGMENT_INTERVAL_MS) return;
  lastSegmentAt = now;
  segmentBusy = true;
  try {
    // segmentForVideoは同期的に動作し、コールバック関数は即座に実行完了します
    selfieSegmentation.segmentForVideo(video, frameTimestamp, (result) => {
      if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
        getMaskCanvas(result.confidenceMasks[0]);
      }
    });
    frameTimestamp += 33;
  } catch (error) {
    log('Segmentation failed. Falling back to raw preview.', error);
    setStatus('背景分離に失敗しました。カメラ映像のみ表示しています。');
    selfieSegmentation = null;
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
      await initSelfieSegmentation();
      setStatus('背景切り替え有効。モードを変更してください。');
    } catch (error) {
      log('SelfieSegmentation initialization failed', error);
      setStatus('MediaPipe初期化に失敗しました。カメラ映像のみ表示します。\n' + error.message);
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
  if (selfieSegmentation) {
    try { selfieSegmentation.close(); } catch (e) {}
  }
  selfieSegmentation = null;
  frameTimestamp = 0;
  maskCanvas = null;
  maskCtx = null;
  maskImageData = null;
  resetTemporalSmoothing();
  personCanvas = null;
  personCtx = null;
  if (currentBgUrl) {
    URL.revokeObjectURL(currentBgUrl);
    currentBgUrl = null;
  }
  backgroundImage = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus('停止しました');
}

// UI Elements for dynamic display
const colorPickerContainer = document.querySelector('#colorPickerContainer');
const imageInputContainer = document.querySelector('#imageInputContainer');

function updateVisibility() {
  const mode = modeSelect.value;
  colorPickerContainer.style.display = mode === 'color' ? 'grid' : 'none';
  imageInputContainer.style.display = mode === 'image' ? 'flex' : 'none';
}

modeSelect.addEventListener('change', updateVisibility);

// Run once initially to hide non-active controls
updateVisibility();
updateMaskFeatherValue();
updateTemporalSmoothingValue();
loadBackgroundImage(defaultBackgroundUrl, DEFAULT_BACKGROUND_NAME);

imageInput.addEventListener('change', () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  // 古いURLがあれば解放してメモリリークを防ぐ
  if (currentBgUrl) {
    URL.revokeObjectURL(currentBgUrl);
  }

  // 新しいオブジェクトURLを生成（onload完了まで解放しない）
  currentBgUrl = URL.createObjectURL(file);
  loadBackgroundImage(currentBgUrl, file.name);
});

modelSelect.addEventListener('change', async () => {
  if (!running) return;
  setStatus('モデル設定を変更しました。再初期化しています...');
  resetTemporalSmoothing();
  try {
    await initSelfieSegmentation();
    setStatus('背景切り替え有効。');
  } catch (error) {
    log('Model reinitialization failed', error);
    setStatus('モデル再初期化に失敗しました。カメラ映像のみ表示します。');
  }
});

maskStyleSelect.addEventListener('change', () => {
  resetTemporalSmoothing();
  log('Mask style changed', { style: maskStyleSelect.value });
});

maskFeatherInput.addEventListener('input', () => {
  updateMaskFeatherValue();
  log('Mask feather changed', { px: getMaskFeatherPx() });
});

temporalSmoothingInput.addEventListener('input', () => {
  updateTemporalSmoothingValue();
  resetTemporalSmoothing();
  log('Temporal smoothing changed', { ratio: getTemporalSmoothing() });
});

startButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);

setStatus(`待機中\nSecureContext: ${window.isSecureContext}\nURL: ${location.href}`);
