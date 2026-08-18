const CACHE = 'astrostack-local-v8';
const SHELL = [
  './', './index.html', './styles.css', './app.js', './worker.js', './python_worker.js',
  './manifest.webmanifest', './icon.svg',
  './vendor/libraw/index.js', './vendor/libraw/worker.js', './vendor/libraw/libraw.js', './vendor/libraw/libraw.wasm',
  './pycore/astrostack/__init__.py', './pycore/astrostack/alignment.py', './pycore/astrostack/calibration.py',
  './pycore/astrostack/io.py', './pycore/astrostack/models.py', './pycore/astrostack/pipeline.py',
  './pycore/astrostack/postprocess.py', './pycore/astrostack/stacking.py', './pycore/astrostack/streaming.py', './pycore/astrostack/web.py',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))).then(() => self.clients.claim()));
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || !event.request.url.startsWith(self.location.origin)) return;
  const refreshable = event.request.mode === 'navigate' || /\.(?:html|js|css|webmanifest)$/.test(new URL(event.request.url).pathname);
  event.respondWith((refreshable ? fetch(event.request).then((response) => {
    const copy = response.clone();
    caches.open(CACHE).then((cache) => cache.put(event.request, copy));
    return response;
  }).catch(() => caches.match(event.request)) : caches.match(event.request).then((cached) => cached || fetch(event.request).then((response) => {
    const copy = response.clone();
    caches.open(CACHE).then((cache) => cache.put(event.request, copy));
    return response;
  }))).catch(() => caches.match('./index.html')));
});
