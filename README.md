# Watermark Remover

A private, browser-only tool to remove watermarks from images using the LaMa (Large Mask Inpainting) ONNX model.

## Features

- **100% Private**: All processing happens locally in your browser. Your images are never uploaded to a server.
- **Automated**: Automatically targets common watermark locations (bottom-right corner).
- **Format Support**: Upload PNG, JPG, or WebP and export to your preferred format.
- **Hardware Acceleration**: Supports WASM (CPU) with hooks for WebGL/WebGPU.

## Quick Start

Since this project uses ES Modules and web workers, it must be served via a local HTTP server (you cannot simply open `index.html` in a browser).

### 1. Serve the project
If you have **Node.js** installed, you can use `npx`:
```bash
npx serve .
```

Alternatively, if you have **Python** installed:
```bash
python3 -m http.server 8000
```

### 2. Access the App
Open your browser and navigate to:
`http://localhost:3000` (for `serve`) or `http://localhost:8000` (for Python).

### 3. Usage
1. **Upload**: Drag and drop an image or click to browse.
2. **Process**: Click **"Remove Watermark"**. 
   - *Note: The first run will download the ~100MB AI model and cache it in your browser's IndexedDB for future instant use.*
3. **Download**: Once processing is complete, click **"Download Result"**.

## Development

### Running Tests
This project uses the native Node.js test runner:
```bash
node --test tests/processor.test.js
```

### Project Structure
- `index.html`: Simplified user interface.
- `src/app.js`: Main application logic and UI orchestration.
- `src/local-onnx.js`: ONNX Runtime integration and mask generation.
- `src/model-cache.js`: Handles IndexedDB caching of the model.
- `src/processor.js`: Image conversion and download utilities.

## Technical Details
- **Model**: LaMa (Resolution: 512x512).
- **Mask**: Circular mask (Radius: 70px) offset by 100px from the bottom-right corner.
- **Runtime**: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).
