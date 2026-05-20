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

let stream = null;
let selfieSegmentation = null;
let running = false;
let rafId = 0;
let backgroundImage = null;
let currentBgUrl = null;
let segmentBusy = false;
let lastSegmentAt = 0;
let frameTimestamp = 0; // MediaPipe用の厳密な単調増加タイムスタンプカウンター
const MEDIAPIPE_SEGMENT_INTERVAL_MS = 33;
const MODNET_SEGMENT_INTERVAL_MS = 50;
const HARD_MASK_THRESHOLD = 0.5;
const DEFAULT_BACKGROUND_NAME = 'AdobeStock_310895879.jpeg';
const ORT_CDN_BASE = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.21.1/';
const MODNET_MODEL_URL = 'https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx';
const MODNET_INPUT_SIZE = 512;

// 高速レンダリング用のオフスクリーンキャンバス キャッシュ（マスク用 & 人物切り抜き用）
let maskImageData = null;
let maskCanvas = null;
let maskCtx = null;
let temporalAlphaBuffer = null;
let temporalAlphaReady = false;
let alphaMaskBuffer = null;
let alphaMaskScratchBuffer = null;

let modnetSession = null;
let modnetInputCanvas = null;
let modnetInputCtx = null;

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

function isModnetSelected() {
  return modelSelect.value === 'modnet';
}

function isMediaPipeGeneralSelected() {
  return modelSelect.value === 'mediapipe-general';
}

function getSelectedModelLabel() {
  switch (modelSelect.value) {
    case 'modnet':
      return 'MODNet (matting / 高品質)';
    case 'mediapipe-landscape':
      return 'MediaPipe Landscape (軽量 / 高速)';
    case 'mediapipe-general':
    default:
      return 'MediaPipe General (高精度)';
  }
}

function getSegmentationIntervalMs() {
  return isModnetSelected() ? MODNET_SEGMENT_INTERVAL_MS : MEDIAPIPE_SEGMENT_INTERVAL_MS;
}

function getOrt() {
  if (!window.ort) {
    throw new Error('ONNX Runtime Web のロードに失敗しました。');
  }
  return window.ort;
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

  if (selfieSegmentation) {
    try { selfieSegmentation.close(); } catch {}
    selfieSegmentation = null;
  }

  log('Initializing modern ImageSegmenter...');
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
  );

  const isGeneral = isMediaPipeGeneralSelected();
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

async function initModnet() {
  const ort = getOrt();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = ORT_CDN_BASE;

  if (modnetSession) return;

  log('Initializing MODNet portrait matting...');
  const sessionOptions = {
    executionProviders: ['webgpu', 'wasm'],
  };

  try {
    modnetSession = await ort.InferenceSession.create(MODNET_MODEL_URL, sessionOptions);
    log('MODNet initialized', { providers: sessionOptions.executionProviders });
  } catch (error) {
    log('MODNet WebGPU init failed, retrying with WASM only', error);
    modnetSession = await ort.InferenceSession.create(MODNET_MODEL_URL, {
      executionProviders: ['wasm'],
    });
    log('MODNet initialized', { providers: ['wasm'] });
  }
}

async function initSelectedSegmentationModel() {
  closeSegmentationEngines();
  clearMaskState();
  frameTimestamp = 0;
  segmentBusy = false;
  lastSegmentAt = 0;

  if (isModnetSelected()) {
    setStatus('MODNet を初期化しています... 初回はモデルのダウンロードに時間がかかります。');
    await initModnet();
    return;
  }

  await initSelfieSegmentation();
}

function closeSegmentationEngines() {
  if (selfieSegmentation) {
    try { selfieSegmentation.close(); } catch {}
  }
  selfieSegmentation = null;

  if (modnetSession) {
    try { modnetSession.release?.(); } catch {}
  }
  modnetSession = null;
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
  if (isModnetSelected()) {
    return alphaToCanvasAlpha(confidence);
  }
  if (maskStyleSelect.value === 'hard') {
    return confidence >= HARD_MASK_THRESHOLD ? 255 : 0;
  }
  const low = getMaskAlphaLow();
  const high = getMaskAlphaHigh();
  const t = Math.max(0, Math.min(1, (confidence - low) / Math.max(high - low, 0.0001)));
  const smooth = t * t * (3 - 2 * t);
  return Math.round(smooth * 255);
}

function getMaskAlphaLow() {
  return Number(maskAlphaLowInput.value);
}

function getMaskAlphaHigh() {
  return Math.max(Number(maskAlphaHighInput.value), getMaskAlphaLow() + 0.01);
}

function updateMaskAlphaValues() {
  maskAlphaLowValue.textContent = getMaskAlphaLow().toFixed(2);
  if (Number(maskAlphaHighInput.value) < getMaskAlphaHigh()) {
    maskAlphaHighInput.value = getMaskAlphaHigh().toFixed(2);
  }
  maskAlphaHighValue.textContent = getMaskAlphaHigh().toFixed(2);
}

function getMaskMorphAmount() {
  return Number(maskMorphInput.value);
}

function updateMaskMorphValue() {
  const amount = getMaskMorphAmount();
  maskMorphValue.textContent = `${amount > 0 ? '+' : ''}${amount}px`;
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

function clearMaskState() {
  maskCanvas = null;
  maskCtx = null;
  maskImageData = null;
  alphaMaskBuffer = null;
  alphaMaskScratchBuffer = null;
  resetTemporalSmoothing();
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

function alphaToCanvasAlpha(alpha) {
  return Math.round(Math.max(0, Math.min(1, alpha)) * 255);
}

function applyMaskMorphology(srcBuffer, scratchBuffer, w, h, amount) {
  const radius = Math.abs(amount);
  if (radius === 0) return srcBuffer;

  const useDilation = amount > 0;
  for (let y = 0; y < h; y++) {
    const yStart = Math.max(0, y - radius);
    const yEnd = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      const xStart = Math.max(0, x - radius);
      const xEnd = Math.min(w - 1, x + radius);
      let value = useDilation ? 0 : 255;

      for (let yy = yStart; yy <= yEnd; yy++) {
        const rowOffset = yy * w;
        for (let xx = xStart; xx <= xEnd; xx++) {
          const sample = srcBuffer[rowOffset + xx];
          if (useDilation) {
            if (sample > value) value = sample;
          } else if (sample < value) {
            value = sample;
          }
        }
      }

      scratchBuffer[y * w + x] = value;
    }
  }

  return scratchBuffer;
}

// Float32マスクから、描画用のアルファマスクキャンバスを生成
function updateMaskCanvasFromArray(maskData, w, h, useDirectAlpha = false) {
  
  if (!maskCanvas || maskCanvas.width !== w || maskCanvas.height !== h) {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    maskCtx = maskCanvas.getContext('2d');
    maskImageData = maskCtx.createImageData(w, h);
    resetTemporalSmoothing();
  }
  const data = maskImageData.data;
  if (!alphaMaskBuffer || alphaMaskBuffer.length !== maskData.length) {
    alphaMaskBuffer = new Uint8ClampedArray(maskData.length);
    alphaMaskScratchBuffer = new Uint8ClampedArray(maskData.length);
  }
  const temporalSmoothing = getTemporalSmoothing();
  if (temporalSmoothing > 0 && (!temporalAlphaBuffer || temporalAlphaBuffer.length !== maskData.length)) {
    temporalAlphaBuffer = new Float32Array(maskData.length);
    temporalAlphaReady = false;
  }
  
  for (let i = 0; i < maskData.length; i++) {
    const value = maskData[i];
    let alpha = useDirectAlpha ? alphaToCanvasAlpha(value) : confidenceToAlpha(value);
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

  const maskAlphaOutput = applyMaskMorphology(
    alphaMaskBuffer,
    alphaMaskScratchBuffer,
    w,
    h,
    getMaskMorphAmount(),
  );
  for (let i = 0; i < maskAlphaOutput.length; i++) {
    const idx = i * 4;
    data[idx] = 255;     // R
    data[idx + 1] = 255; // G
    data[idx + 2] = 255; // B
    data[idx + 3] = maskAlphaOutput[i]; // A (人物信頼度に応じて連続的に変化)
  }
  
  maskCtx.putImageData(maskImageData, 0, 0);
  return maskCanvas;
}

function getMaskCanvas(mask) {
  return updateMaskCanvasFromArray(mask.getAsFloat32Array(), mask.width, mask.height);
}

function getModnetInputCanvas() {
  if (!modnetInputCanvas) {
    modnetInputCanvas = document.createElement('canvas');
    modnetInputCanvas.width = MODNET_INPUT_SIZE;
    modnetInputCanvas.height = MODNET_INPUT_SIZE;
    modnetInputCtx = modnetInputCanvas.getContext('2d', { willReadFrequently: true });
  }
  return modnetInputCanvas;
}

function createModnetInputTensor() {
  const ort = getOrt();
  const inputCanvas = getModnetInputCanvas();
  modnetInputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
  modnetInputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);

  const imageData = modnetInputCtx.getImageData(0, 0, inputCanvas.width, inputCanvas.height);
  const hw = inputCanvas.width * inputCanvas.height;
  const tensorData = new Float32Array(3 * hw);

  for (let i = 0; i < hw; i++) {
    const pixelIdx = i * 4;
    tensorData[i] = (imageData.data[pixelIdx] - 127.5) / 127.5;
    tensorData[hw + i] = (imageData.data[pixelIdx + 1] - 127.5) / 127.5;
    tensorData[hw * 2 + i] = (imageData.data[pixelIdx + 2] - 127.5) / 127.5;
  }

  return new ort.Tensor('float32', tensorData, [1, 3, inputCanvas.height, inputCanvas.width]);
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

function drawMaskOnCtx(targetCtx, w, h) {
  if (isModnetSelected()) {
    drawMirroredOnCtx(() => targetCtx.drawImage(maskCanvas, 0, 0, w, h), targetCtx, w);
    return;
  }
  drawMirroredOnCtx(() => drawCoverImageOnCtx(maskCanvas, targetCtx, w, h), targetCtx, w);
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
  drawMaskOnCtx(personCtx, w, h);
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

function requestMediaPipeSegmentationIfNeeded(now) {
  if (!selfieSegmentation || segmentBusy) return;
  if (now - lastSegmentAt < getSegmentationIntervalMs()) return;
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

function requestModnetSegmentationIfNeeded(now) {
  if (!modnetSession || segmentBusy) return;
  if (now - lastSegmentAt < getSegmentationIntervalMs()) return;
  lastSegmentAt = now;
  segmentBusy = true;

  try {
    const inputTensor = createModnetInputTensor();
    const feeds = { [modnetSession.inputNames[0]]: inputTensor };

    modnetSession.run(feeds).then((results) => {
      const outputName = modnetSession.outputNames[0];
      const outputTensor = results[outputName];
      updateMaskCanvasFromArray(outputTensor.data, MODNET_INPUT_SIZE, MODNET_INPUT_SIZE, true);
    }).catch((error) => {
      log('MODNet inference failed. Falling back to raw preview.', error);
      setStatus('MODNet推論に失敗しました。カメラ映像のみ表示しています。');
      modnetSession = null;
    }).finally(() => {
      segmentBusy = false;
    });
  } catch (error) {
    log('MODNet preprocessing failed. Falling back to raw preview.', error);
    setStatus('MODNet前処理に失敗しました。カメラ映像のみ表示しています。');
    modnetSession = null;
    segmentBusy = false;
  }
}

function requestSegmentationIfNeeded(now) {
  if (isModnetSelected()) {
    requestModnetSegmentationIfNeeded(now);
    return;
  }
  requestMediaPipeSegmentationIfNeeded(now);
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
      await initSelectedSegmentationModel();
      setStatus('背景切り替え有効。モードを変更してください。');
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
  closeSegmentationEngines();
  frameTimestamp = 0;
  clearMaskState();
  personCanvas = null;
  personCtx = null;
  modnetInputCanvas = null;
  modnetInputCtx = null;
  segmentBusy = false;
  lastSegmentAt = 0;
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

function updateModelControlState() {
  const usingModnet = isModnetSelected();
  maskStyleSelect.disabled = usingModnet;
  maskAlphaLowInput.disabled = usingModnet;
  maskAlphaHighInput.disabled = usingModnet;
  maskStyleSelect.title = usingModnet ? 'MODNet はアルファマットを直接出力するため、この設定は使いません。' : '';
  maskAlphaLowInput.title = usingModnet ? 'MODNet はアルファマットを直接出力するため、この設定は使いません。' : '';
  maskAlphaHighInput.title = usingModnet ? 'MODNet はアルファマットを直接出力するため、この設定は使いません。' : '';
}

modeSelect.addEventListener('change', updateVisibility);

// Run once initially to hide non-active controls
updateVisibility();
updateModelControlState();
updateMaskAlphaValues();
updateMaskMorphValue();
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
  updateModelControlState();
  if (!running) return;
  setStatus('モデル設定を変更しました。再初期化しています...');
  try {
    await initSelectedSegmentationModel();
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

maskAlphaLowInput.addEventListener('input', () => {
  updateMaskAlphaValues();
  resetTemporalSmoothing();
  log('Mask alpha low changed', { value: getMaskAlphaLow() });
});

maskAlphaHighInput.addEventListener('input', () => {
  updateMaskAlphaValues();
  resetTemporalSmoothing();
  log('Mask alpha high changed', { value: getMaskAlphaHigh() });
});

maskMorphInput.addEventListener('input', () => {
  updateMaskMorphValue();
  resetTemporalSmoothing();
  log('Mask morphology changed', { amount: getMaskMorphAmount() });
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
