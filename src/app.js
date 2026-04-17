import {
  createSession,
  runLamaModel,
} from "./local-onnx.js";
import { fetchAndCacheModel } from "./model-cache.js";
import { buildDownloadName, canvasToBlob } from "./processor.js";

const elements = {
  dropzone: document.querySelector("[data-dropzone]"),
  fileInput: document.querySelector("[data-file-input]"),
  processButton: document.querySelector("[data-process]"),
  downloadLink: document.querySelector("[data-download]"),
  status: document.querySelector("[data-status]"),
  modelUrl: document.querySelector("[data-model-url]"),
  provider: document.querySelector("[data-provider]"),
  formatSelect: document.querySelector("[data-format]"),
  qualityRange: document.querySelector("[data-quality]"),
  privacyMode: document.querySelector("[data-privacy]"),
  feather: document.querySelector("[data-feather]"),
  previewImage: document.querySelector("[data-preview-image]"),
  previewMeta: document.querySelector("[data-preview-meta]"),
};

let currentFile = null;
let currentSession = null;
let currentDownloadUrl = null;
let loadedModelUrl = "";
let imageReady = false;

function updateStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.dataset.state = isError ? "error" : "idle";
}

function revokeDownloadUrl() {
  if (currentDownloadUrl) {
    URL.revokeObjectURL(currentDownloadUrl);
    currentDownloadUrl = null;
  }
}

function setPreview(file) {
  revokeDownloadUrl();

  const objectUrl = URL.createObjectURL(file);
  imageReady = false;
  elements.previewImage.src = objectUrl;
  elements.previewImage.onload = () => {
    imageReady = true;
    URL.revokeObjectURL(objectUrl);
  };

  elements.previewMeta.textContent = `${file.name} • ${Math.round(file.size / 1024)} KB`;
  elements.downloadLink.hidden = true;
  updateStatus("Image loaded. Click 'Remove Watermark' to start.");
}

function handleFiles(fileList) {
  const [file] = fileList;

  if (!file) {
    return;
  }

  if (!file.type.startsWith("image/")) {
    updateStatus("Please choose a PNG, JPEG, or WebP image.", true);
    return;
  }

  currentFile = file;
  setPreview(file);
}

async function loadModel() {
  const modelUrl = elements.modelUrl.value.trim();

  if (!modelUrl) {
    throw new Error("Model URL is missing.");
  }

  updateStatus("Loading intelligence model locally...");

  const modelBuffer = await fetchAndCacheModel(modelUrl, (progress) => {
    if (progress.phase === "cache-hit") {
      updateStatus("Model ready from local cache.");
      return;
    }

    if (progress.phase === "download") {
      const total = progress.total ? ` / ${Math.round(progress.total / 1024 / 1024)} MB` : "";
      updateStatus(
        `Downloading model assets: ${Math.round(progress.loaded / 1024 / 1024)} MB${total}`,
      );
    }
  });

  currentSession = await createSession(modelBuffer, elements.provider.value);
  loadedModelUrl = modelUrl;
}

async function processCurrentFile() {
  if (!currentFile) {
    updateStatus("Please upload an image first.", true);
    return;
  }

  if (!imageReady) {
    updateStatus("Please wait for the image to load.", true);
    return;
  }

  elements.processButton.disabled = true;

  try {
    if (!currentSession) {
      await loadModel();
    }

    updateStatus("Removing watermark...");

    const runResult = await runLamaModel({
      file: currentFile,
      session: currentSession,
      executionProvider: elements.provider.value,
      blurRadius: Number(elements.feather.value),
    });

    const result = await canvasToBlob(runResult.canvas, {
      format: elements.formatSelect.value,
      quality: elements.qualityRange.value,
      privacy: elements.privacyMode.value,
    });

    revokeDownloadUrl();
    currentDownloadUrl = URL.createObjectURL(result.blob);
    elements.downloadLink.href = currentDownloadUrl;
    elements.downloadLink.download = buildDownloadName(currentFile.name, elements.formatSelect.value);
    elements.downloadLink.hidden = false;

    // Update the preview image with the processed result
    elements.previewImage.src = currentDownloadUrl;

    updateStatus(
      `Success! Watermark removed. Preview updated.`,
    );
  } catch (error) {
    updateStatus(error.message || "Something went wrong during processing.", true);
  } finally {
    elements.processButton.disabled = false;
  }
}

elements.dropzone.addEventListener("click", () => elements.fileInput.click());
elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.dataset.dragging = "true";
});
elements.dropzone.addEventListener("dragleave", () => {
  elements.dropzone.dataset.dragging = "false";
});
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.dataset.dragging = "false";
  handleFiles(event.dataTransfer.files);
});
elements.fileInput.addEventListener("change", (event) => handleFiles(event.target.files));
elements.processButton.addEventListener("click", () => void processCurrentFile());

updateStatus("Drop an image to start. Everything stays in your browser.");
