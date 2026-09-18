// Service worker for offline capability. Precaches the offline ViT model +
// ONNX Runtime WASM binary (large, stable-path assets) on install, and
// caches everything else same-origin as it's requested (so a page visited
// once online keeps working offline on repeat visits).
const CACHE_NAME = "deepsyndrome-offline-v1";
const PRECACHE = [
  "/",
  "/models/vit_s16.onnx",
  "/ort/ort-wasm-simd-threaded.wasm",
  "/ort/ort-wasm-simd-threaded.mjs",
  "/ort/ort.wasm.min.mjs",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    // add() per file (not addAll): a missing file - e.g. the large model files
    // when they aren't deployed - must not fail the whole service worker install.
    caches.open(CACHE_NAME).then((cache) =>
      Promise.all(PRECACHE.map((url) => cache.add(url).catch(() => {})))
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // never cache the HF Space / cross-origin calls

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return res;
        })
        .catch(() => cached); // offline and not cached - let it fail naturally
    })
  );
});
