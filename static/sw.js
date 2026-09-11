// DeepSyndrome service worker
// Scope is "/" (served from root as /sw.js) so it can control navigation requests.
const CACHE_NAME = "deepsyndrome-shell-v1";
const CORE_ASSETS = [
  "/",
  "/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

// Never let the service worker touch these — they're live inference calls,
// never safe to serve from cache.
const NETWORK_ONLY = ["/predict", "/health", "/api", "/docs", "/openapi.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return; // let POST /predict pass through untouched

  const url = new URL(request.url);
  if (NETWORK_ONLY.some((p) => url.pathname.startsWith(p))) return;

  // Stale-while-revalidate for the app shell / icons.
  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(request);
      const network = fetch(request)
        .then((res) => {
          if (res.ok) cache.put(request, res.clone());
          return res;
        })
        .catch(() => cached); // offline: fall back to whatever we have cached
      return cached || network;
    })
  );
});
