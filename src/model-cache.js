const DATABASE_NAME = "local-onnx-model-cache";
const STORE_NAME = "models";
const DATABASE_VERSION = 1;

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "url" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readRecord(url) {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const request = transaction.objectStore(STORE_NAME).get(url);

    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

async function writeRecord(record) {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(record);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

export async function getCachedModel(url) {
  const record = await readRecord(url);

  if (!record?.buffer) {
    return null;
  }

  return {
    buffer: record.buffer,
    updatedAt: record.updatedAt,
    size: record.size,
  };
}

export async function fetchAndCacheModel(url, onProgress = () => {}) {
  const cached = await getCachedModel(url);

  if (cached) {
    onProgress({ phase: "cache-hit", loaded: cached.size, total: cached.size });
    return cached.buffer;
  }

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Model download failed with ${response.status}.`);
  }

  const contentLength = Number(response.headers.get("content-length")) || 0;

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    await writeRecord({
      url,
      buffer,
      size: buffer.byteLength,
      updatedAt: Date.now(),
    });
    onProgress({ phase: "done", loaded: buffer.byteLength, total: buffer.byteLength });
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ phase: "download", loaded, total: contentLength });
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;

  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  await writeRecord({
    url,
    buffer: buffer.buffer,
    size: loaded,
    updatedAt: Date.now(),
  });

  onProgress({ phase: "done", loaded, total: contentLength || loaded });
  return buffer.buffer;
}
