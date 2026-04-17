import test from "node:test";
import assert from "node:assert/strict";

import {
  getFormatConfig,
  buildDownloadName,
  clampQuality,
} from "../src/processor.js";
import {
  detectPromptSupport,
  inferImageTensorShape,
  computeContainRect,
  computeWatermarkMask,
  normaliseMaskValue,
  normaliseExecutionProvider,
} from "../src/local-onnx.js";

test("getFormatConfig returns expected MIME and extension for webp", () => {
  assert.deepEqual(getFormatConfig("webp"), {
    format: "webp",
    mimeType: "image/webp",
    extension: "webp",
  });
});

test("getFormatConfig falls back to webp for unknown formats", () => {
  assert.equal(getFormatConfig("tiff").format, "webp");
});

test("buildDownloadName swaps the extension", () => {
  assert.equal(buildDownloadName("holiday.photo.png", "jpeg"), "holiday.photo.jpeg");
});

test("clampQuality keeps quality within browser-safe bounds", () => {
  assert.equal(clampQuality(-3), 0.1);
  assert.equal(clampQuality(2), 1);
  assert.equal(clampQuality(0.82), 0.82);
});

test("detectPromptSupport identifies text-conditioned model inputs", () => {
  assert.equal(detectPromptSupport(["image", "input_ids"]), true);
  assert.equal(detectPromptSupport(["input", "mask"]), false);
});

test("inferImageTensorShape returns square fallback when metadata is dynamic", () => {
  assert.deepEqual(inferImageTensorShape(["batch", 3, "height", "width"]), {
    width: 512,
    height: 512,
    channels: 3,
  });
});

test("normaliseExecutionProvider falls back to wasm for unknown providers", () => {
  assert.equal(normaliseExecutionProvider("made-up"), "wasm");
  assert.equal(normaliseExecutionProvider("webgl"), "webgl");
});

test("computeContainRect fits a landscape image into a square canvas", () => {
  assert.deepEqual(computeContainRect(2048, 1024, 512), {
    drawWidth: 512,
    drawHeight: 256,
    offsetX: 0,
    offsetY: 128,
    scale: 0.25,
  });
});

test("normaliseMaskValue uses binary threshold at 128", () => {
  assert.equal(normaliseMaskValue(0), 0.0);
  assert.equal(normaliseMaskValue(127), 0.0);
  assert.equal(normaliseMaskValue(129), 1.0);
  assert.equal(normaliseMaskValue(255), 1.0);
});

test("computeWatermarkMask returns a bottom-right offset circle descriptor", () => {
  assert.deepEqual(computeWatermarkMask(1000, 800, 70), {
    centerX: 900,
    centerY: 700,
    radius: 70,
  });
});
