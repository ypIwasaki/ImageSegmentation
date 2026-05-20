# Realtime Background Switcher HTTPS Final

Webカメラ映像に対して、人物セグメンテーションを行い、背景を以下に切り替える最小アプリです。

- カメラ映像そのまま
- 背景透明
- 背景色
- 背景画像差し替え
- 背景ぼかし

## 重要な修正点

`@mediapipe/selfie_segmentation` を Vite の `import` でバンドルすると、環境によって以下のエラーが出ます。

```txt
RuntimeError: Aborted(Module.arguments has been replaced with plain arguments_ ...)
```

この修正版では、MediaPipe Selfie Segmentation を **npm importせず**、`index.html` の classic script で読み込んでいます。

```html
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@0.1/selfie_segmentation.js"></script>
```

これにより、Emscripten/WASMまわりの `Module.arguments` エラーを回避します。

## 起動

```bash
npm install
npm run dev:https
```

ブラウザで開くURL:

```txt
https://localhost:5173
```

初回は自己署名証明書の警告が出ます。ローカル検証なら詳細表示から許可してください。

## Docker起動

```bash
docker compose up --build
```

```txt
https://localhost:5173
```

## 背景透明について

Canvasは透明になりますが、ページ背景に市松模様を敷いているため、ブラウザ上では市松模様が見えます。`canvas.captureStream()` で配信用ストリームにすると、背景透明ではなく黒や合成先依存になる場合があります。通常のWebRTCではアルファ付き映像トラックをそのまま扱えないため、配信用途では背景画像/背景色/ぼかしで合成してください。

## トラブルシュート

### カメラは映るが背景切り替えできない

Consoleに以下が出ていないか確認してください。

```txt
SelfieSegmentation initialization failed
```

Networkタブで `selfie_segmentation` / `.wasm` / `.binarypb` の読み込み失敗も確認してください。

### Module.arguments has been replaced...

古いzipや npm import 版を実行しています。この修正版では `src/main.js` に `@mediapipe/selfie_segmentation` の import はありません。

### HTTPSにならない

```bash
npm run dev:https
```

で起動してください。Viteの表示が以下ならOKです。

```txt
https://localhost:5173/
```
