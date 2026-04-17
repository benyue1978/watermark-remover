const FORMAT_CONFIG = {
  webp: {
    format: "webp",
    mimeType: "image/webp",
    extension: "webp",
  },
  png: {
    format: "png",
    mimeType: "image/png",
    extension: "png",
  },
  jpeg: {
    format: "jpeg",
    mimeType: "image/jpeg",
    extension: "jpeg",
  },
};

export function getFormatConfig(format) {
  return FORMAT_CONFIG[format] ?? FORMAT_CONFIG.webp;
}

export function buildDownloadName(filename, format) {
  const baseName = filename.replace(/\.[^.]+$/, "") || "processed-image";
  return `${baseName}.${getFormatConfig(format).extension}`;
}

export function clampQuality(value) {
  return Math.min(1, Math.max(0.1, Number(value) || 0.92));
}

export async function fileToImageBitmap(file) {
  return createImageBitmap(file);
}

export async function canvasToBlob(canvas, options = {}) {
  const { mimeType } = getFormatConfig(options.format);
  const quality = clampQuality(options.quality);

  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      (result) => {
        if (!result) {
          reject(new Error("Unable to export image."));
          return;
        }

        resolve(result);
      },
      mimeType,
      mimeType === "image/png" ? undefined : quality,
    );
  });

  return {
    blob,
    width: canvas.width,
    height: canvas.height,
    mimeType,
  };
}

export async function renderImageToBlob(file, options = {}) {
  const bitmap = await fileToImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;

  const context = canvas.getContext("2d", { alpha: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();

  return canvasToBlob(canvas, options);
}
