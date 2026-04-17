const DEFAULT_SHAPE = {
  width: 512,
  height: 512,
  channels: 3,
};

const SUPPORTED_EXECUTION_PROVIDERS = new Set(["wasm", "webgl", "webgpu"]);

function getOrt() {
  if (!globalThis.ort) {
    throw new Error("ONNX Runtime Web is not loaded.");
  }

  return globalThis.ort;
}

export function normaliseExecutionProvider(provider) {
  return SUPPORTED_EXECUTION_PROVIDERS.has(provider) ? provider : "wasm";
}

export function detectPromptSupport(inputNames) {
  return inputNames.some((name) => /(prompt|text|token|input_ids)/i.test(name));
}

export function inferImageTensorShape(dims) {
  if (!Array.isArray(dims) || dims.length < 4) {
    return DEFAULT_SHAPE;
  }

  const channels = Number.isFinite(dims[1]) ? dims[1] : 3;
  const height = Number.isFinite(dims[2]) ? dims[2] : DEFAULT_SHAPE.height;
  const width = Number.isFinite(dims[3]) ? dims[3] : DEFAULT_SHAPE.width;

  return {
    width,
    height,
    channels,
  };
}

export function computeContainRect(sourceWidth, sourceHeight, targetSize) {
  const scale = Math.min(targetSize / sourceWidth, targetSize / sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);

  return {
    drawWidth,
    drawHeight,
    offsetX: Math.floor((targetSize - drawWidth) / 2),
    offsetY: Math.floor((targetSize - drawHeight) / 2),
    scale,
  };
}

export function computeWatermarkMask(width, height, radius = 70) {
  return {
    centerX: width - 100,
    centerY: height - 100,
    radius,
  };
}

export function normaliseMaskValue(value) {
  return value > 128 ? 1.0 : 0.0;
}

function imageDataToTensor(imageData) {
  const { data, width, height } = imageData;
  const channels = 3;
  const tensor = new Float32Array(width * height * channels);
  const planeSize = width * height;

  for (let index = 0; index < planeSize; index += 1) {
    const pixelOffset = index * 4;

    tensor[index] = data[pixelOffset] / 255;
    tensor[index + planeSize] = data[pixelOffset + 1] / 255;
    tensor[index + planeSize * 2] = data[pixelOffset + 2] / 255;
  }

  return tensor;
}

function maskImageDataToTensor(imageData) {
  const { data, width, height } = imageData;
  const tensor = new Float32Array(width * height);

  for (let index = 0; index < width * height; index += 1) {
    tensor[index] = normaliseMaskValue(data[index * 4]);
  }

  return tensor;
}

function tensorToImageData(tensor, width, height) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  const planeSize = width * height;

  // Detect if output is 0-1 or 0-255 by checking max value
  let maxValue = 0;
  for (let i = 0; i < tensor.length; i++) {
    if (tensor[i] > maxValue) maxValue = tensor[i];
  }
  
  const isScaled0to1 = maxValue <= 1.1; // Add some margin

  for (let index = 0; index < planeSize; index += 1) {
    let r = tensor[index];
    let g = tensor[index + planeSize];
    let b = tensor[index + planeSize * 2];

    if (isScaled0to1) {
      r *= 255;
      g *= 255;
      b *= 255;
    }

    pixels[index * 4] = Math.round(Math.min(255, Math.max(0, r)));
    pixels[index * 4 + 1] = Math.round(Math.min(255, Math.max(0, g)));
    pixels[index * 4 + 2] = Math.round(Math.min(255, Math.max(0, b)));
    pixels[index * 4 + 3] = 255;
  }

  return new ImageData(pixels, width, height);
}

function createSquareCanvas(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

function drawContainedBitmap(bitmap, targetSize, smoothing = true) {
  const canvas = createSquareCanvas(targetSize);
  const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
  const rect = computeContainRect(bitmap.width, bitmap.height, targetSize);
  context.imageSmoothingEnabled = smoothing;
  context.drawImage(bitmap, rect.offsetX, rect.offsetY, rect.drawWidth, rect.drawHeight);

  return {
    canvas,
    context,
    rect,
  };
}

function createMaskCanvasFromSource(maskCanvas, rect, targetSize, blurRadius) {
  const output = createSquareCanvas(targetSize);
  const outputContext = output.getContext("2d", { alpha: false, willReadFrequently: true });
  outputContext.fillStyle = "black";
  outputContext.fillRect(0, 0, targetSize, targetSize);
  outputContext.filter = blurRadius > 0 ? `blur(${blurRadius}px)` : "none";
  outputContext.drawImage(maskCanvas, rect.offsetX, rect.offsetY, rect.drawWidth, rect.drawHeight);
  outputContext.filter = "none";

  return output;
}

export async function createSession(modelBuffer, executionProvider) {
  const ort = getOrt();
  return ort.InferenceSession.create(modelBuffer, {
    executionProviders: [normaliseExecutionProvider(executionProvider)],
    graphOptimizationLevel: "all",
  });
}

export async function runLamaModel({
  file,
  session,
  executionProvider,
  blurRadius = 15,
}) {
  const ort = getOrt();
  const bitmap = await createImageBitmap(file);

  try {
    const inputNames = new Set(session.inputNames);

    if (!inputNames.has("image") || !inputNames.has("mask")) {
      throw new Error(
        `Expected LaMa-style inputs named image and mask. Loaded inputs: ${session.inputNames.join(", ")}.`,
      );
    }

    // Create full-res mask (grayscale: black bg, white hole)
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = bitmap.width;
    maskCanvas.height = bitmap.height;
    const maskContext = maskCanvas.getContext("2d", { alpha: false });
    
    maskContext.fillStyle = "black";
    maskContext.fillRect(0, 0, bitmap.width, bitmap.height);

    const maskInfo = computeWatermarkMask(bitmap.width, bitmap.height);
    maskContext.fillStyle = "white";
    maskContext.beginPath();
    maskContext.arc(maskInfo.centerX, maskInfo.centerY, maskInfo.radius, 0, Math.PI * 2);
    maskContext.fill();

    const modelSize = 512;
    const { canvas: imageCanvas, context: imageContext, rect } = drawContainedBitmap(
      bitmap,
      modelSize,
    );
    const imageTensorData = imageDataToTensor(
      imageContext.getImageData(0, 0, modelSize, modelSize),
    );

    // Create 512x512 mask directly
    const resizedMaskCanvas = createSquareCanvas(modelSize);
    const resizedMaskContext = resizedMaskCanvas.getContext("2d", { alpha: false });
    resizedMaskContext.fillStyle = "black";
    resizedMaskContext.fillRect(0, 0, modelSize, modelSize);
    
    const scale = rect.scale;
    const centerX = rect.offsetX + (maskInfo.centerX * scale);
    const centerY = rect.offsetY + (maskInfo.centerY * scale);
    const radius = maskInfo.radius * scale;
    const scaledBlur = blurRadius * scale;

    resizedMaskContext.save();
    if (scaledBlur > 0) {
      resizedMaskContext.filter = `blur(${scaledBlur}px)`;
    }
    resizedMaskContext.fillStyle = "white";
    resizedMaskContext.beginPath();
    resizedMaskContext.arc(centerX, centerY, radius, 0, Math.PI * 2);
    resizedMaskContext.fill();
    resizedMaskContext.restore();

    const maskTensorData = maskImageDataToTensor(
      resizedMaskContext.getImageData(0, 0, modelSize, modelSize),
    );

    const feeds = {
      image: new ort.Tensor("float32", imageTensorData, [1, 3, modelSize, modelSize]),
      mask: new ort.Tensor("float32", maskTensorData, [1, 1, modelSize, modelSize]),
    };

    const outputName = session.outputNames[0];
    const results = await session.run(feeds);
    const outputTensor = results[outputName];

    if (!outputTensor) {
      throw new Error("The model did not return an output tensor.");
    }

    const outputImageData = tensorToImageData(outputTensor.data, modelSize, modelSize);
    const inpaintCanvas = createSquareCanvas(modelSize);
    const inpaintContext = inpaintCanvas.getContext("2d", { alpha: true });
    inpaintContext.putImageData(outputImageData, 0, 0);

    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = bitmap.width;
    compositeCanvas.height = bitmap.height;
    const compositeContext = compositeCanvas.getContext("2d", { alpha: true });

    compositeContext.save();
    compositeContext.drawImage(
      inpaintCanvas,
      rect.offsetX,
      rect.offsetY,
      rect.drawWidth,
      rect.drawHeight,
      0,
      0,
      bitmap.width,
      bitmap.height,
    );
    compositeContext.restore();

    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = bitmap.width;
    finalCanvas.height = bitmap.height;
    const finalContext = finalCanvas.getContext("2d", { alpha: true });
    finalContext.drawImage(bitmap, 0, 0);
    finalContext.save();
    finalContext.drawImage(compositeCanvas, 0, 0);
    finalContext.restore();

    return {
      canvas: finalCanvas,
      executionProvider: normaliseExecutionProvider(executionProvider),
      inputSize: `${modelSize}×${modelSize}`,
      outputSize: `${modelSize}×${modelSize}`,
      outputName,
      maskCanvas: resizedMaskCanvas,
    };
  } finally {
    bitmap.close();
  }
}
