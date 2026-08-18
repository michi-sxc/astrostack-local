const $ = (id) => document.getElementById(id);

const state = {
  lights: [],
  darks: [],
  mask: null,
  worker: null,
  lastBlob: null,
  installEvent: null,
  running: false,
};

const defaults = {
  mode: 'sigma',
  autoBrightness: true,
  protectForeground: true,
  lightPollution: false,
  hdr: true,
  starEnhancement: true,
  dynamicDenoise: true,
  chromaticAberration: true,
  distortion: true,
  resolution: 'full',
  coverage: 98,
};

function setStatus(label, tone = 'ready') {
  const badge = $('statusBadge');
  badge.textContent = label;
  badge.className = `status-badge ${tone}`;
}

function log(message, tone = '') {
  const panel = $('logPanel');
  if (panel.querySelector('.log-muted')) panel.textContent = '';
  const line = document.createElement('div');
  line.className = tone ? `log-line ${tone}` : 'log-line';
  line.textContent = message;
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
}

function settings() {
  const out = { ...defaults };
  document.querySelectorAll('input[type="checkbox"]').forEach((input) => { out[input.id] = input.checked; });
  out.mode = document.querySelector('input[name="mode"]:checked')?.value || defaults.mode;
  out.resolution = $('resolution').value;
  out.coverage = Number($('coverage').value);
  return out;
}

function saveSettings() {
  try { localStorage.setItem('astrostack-settings-v1', JSON.stringify(settings())); } catch (_) { /* private mode */ }
}

function restoreSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('astrostack-settings-v1') || '{}'); } catch (_) { saved = {}; }
  const merged = { ...defaults, ...saved };
  document.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = Boolean(merged[input.id]); });
  const radio = document.querySelector(`input[name="mode"][value="${merged.mode}"]`);
  if (radio) radio.checked = true;
  if (['full', 'mobile', 'preview'].includes(merged.resolution)) $('resolution').value = merged.resolution;
  $('coverage').value = Math.min(100, Math.max(75, Number(merged.coverage) || defaults.coverage));
  $('coverageValue').textContent = `${$('coverage').value}%`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function isRaw(file) {
  return /\.(dng|nef|cr2|cr3|arw|rw2|orf|raf|raw)$/i.test(file.name);
}

function renderFiles() {
  const total = state.lights.length + state.darks.length;
  $('lightsCount').textContent = state.lights.length ? `${state.lights.length} selected` : 'Choose files';
  $('darksCount').textContent = state.darks.length ? `${state.darks.length} selected` : 'Choose files';
  $('fileSummary').textContent = total ? `${state.lights.length} light${state.lights.length === 1 ? '' : 's'} · ${state.darks.length} dark${state.darks.length === 1 ? '' : 's'}` : 'No frames loaded';
  const list = $('fileList');
  list.textContent = '';
  const files = [...state.lights.map((file) => ({ file, kind: 'Light' })), ...state.darks.map((file) => ({ file, kind: 'Dark' }))];
  files.slice(0, 8).forEach(({ file, kind }) => {
    const chip = document.createElement('span');
    chip.className = `file-chip ${kind.toLowerCase()}`;
    chip.textContent = `${kind}: ${file.name} · ${formatBytes(file.size)}`;
    list.appendChild(chip);
  });
  if (files.length > 8) {
    const more = document.createElement('span');
    more.className = 'file-chip';
    more.textContent = `+${files.length - 8} more`;
    list.appendChild(more);
  }
  const hasRaw = files.some(({ file }) => isRaw(file));
  const notice = $('formatNotice');
  notice.hidden = !hasRaw;
  if (hasRaw) notice.textContent = 'RAW/DNG inputs are decoded locally with the bundled LibRaw WASM engine. Large native-resolution sets can use substantial phone memory.';
}

function addFiles(target, files) {
  const incoming = [...files];
  state[target] = [...state[target], ...incoming];
  renderFiles();
}

function clearFiles() {
  state.lights = [];
  state.darks = [];
  state.mask = null;
  $('lightsInput').value = '';
  $('darksInput').value = '';
  $('maskInput').value = '';
  renderFiles();
  setStatus('Ready');
  log('Frame set cleared.');
}

function resetFinish() {
  Object.entries(defaults).forEach(([key, value]) => {
    const checkbox = $(key);
    if (checkbox && checkbox.type === 'checkbox') checkbox.checked = value;
  });
  const radio = document.querySelector(`input[name="mode"][value="${defaults.mode}"]`);
  if (radio) radio.checked = true;
  $('resolution').value = defaults.resolution;
  $('coverage').value = defaults.coverage;
  $('coverageValue').textContent = `${defaults.coverage}%`;
  saveSettings();
}

function updateMemoryHint() {
  const mode = $('resolution').value;
  $('memoryHint').textContent = mode === 'full'
    ? 'Native mode keeps source dimensions. On a phone, use Mobile-safe if the browser warns about memory.'
    : mode === 'mobile'
      ? 'Mobile-safe caps the long edge at 1800 px while keeping the full processing pipeline.'
      : 'Quick preview caps the long edge at 1100 px for fast tuning before a native export.';
}

function appendLog(message) { log(message); }

function ensureWorker() {
  if (!state.worker) {
    state.worker = new Worker('./worker.js', { type: 'module' });
    state.worker.onmessage = handleWorkerMessage;
    state.worker.onerror = (event) => {
      state.running = false;
      $('processButton').disabled = false;
      $('progressWrap').hidden = true;
      setStatus('Error', 'error');
      log(event.message || 'The processing worker stopped unexpectedly.', 'error');
    };
  }
  return state.worker;
}

function handleWorkerMessage(event) {
  const message = event.data || {};
  if (message.type === 'progress') {
    $('progressWrap').hidden = false;
    $('progressLabel').textContent = message.label || 'Processing';
    const percent = Math.max(0, Math.min(100, Math.round(message.percent || 0)));
    $('progressPercent').textContent = `${percent}%`;
    $('progressBar').style.width = `${percent}%`;
  } else if (message.type === 'log') {
    appendLog(message.message);
  } else if (message.type === 'done') {
    const started = state.startedAt || performance.now();
    const pixels = new Uint8ClampedArray(message.buffer);
    const image = new ImageData(pixels, message.width, message.height);
    const canvas = $('previewCanvas');
    canvas.width = message.width;
    canvas.height = message.height;
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.putImageData(image, 0, 0);
    canvas.hidden = false;
    $('emptyPreview').hidden = true;
    $('previewOverlay').hidden = false;
    $('previewResolution').textContent = `${message.width} × ${message.height}`;
    $('previewTime').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
    canvas.toBlob((blob) => { state.lastBlob = blob; $('downloadButton').disabled = !blob; }, 'image/png');
    state.running = false;
    $('processButton').disabled = false;
    $('progressBar').style.width = '100%';
    $('progressPercent').textContent = '100%';
    setStatus('Complete', 'done');
    log('Finished locally. Your source frames were not uploaded.', 'success');
  } else if (message.type === 'error') {
    state.running = false;
    $('processButton').disabled = false;
    $('progressWrap').hidden = true;
    setStatus('Error', 'error');
    log(message.message || 'Processing failed.', 'error');
  }
}

function processStack() {
  if (state.running) return;
  if (!state.lights.length) {
    setStatus('Add lights', 'error');
    log('Choose at least one light frame before processing.', 'error');
    return;
  }
  state.running = true;
  state.startedAt = performance.now();
  state.lastBlob = null;
  $('downloadButton').disabled = true;
  $('processButton').disabled = true;
  $('progressWrap').hidden = false;
  $('progressBar').style.width = '0%';
  $('progressPercent').textContent = '0%';
  $('logPanel').textContent = '';
  setStatus('Processing', 'running');
  const config = settings();
  saveSettings();
  ensureWorker().postMessage({
    type: 'process',
    lights: state.lights,
    darks: state.darks,
    mask: state.mask,
    settings: config,
  });
}

function downloadResult() {
  if (!state.lastBlob) return;
  const url = URL.createObjectURL(state.lastBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `astrostack-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function installPrompt() {
  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault();
    state.installEvent = event;
    $('installButton').hidden = false;
  });
  $('installButton').addEventListener('click', async () => {
    if (!state.installEvent) return;
    state.installEvent.prompt();
    await state.installEvent.userChoice;
    state.installEvent = null;
    $('installButton').hidden = true;
  });
}

function wire() {
  $('lightsInput').addEventListener('change', (event) => addFiles('lights', event.target.files));
  $('darksInput').addEventListener('change', (event) => addFiles('darks', event.target.files));
  $('maskInput').addEventListener('change', (event) => { state.mask = event.target.files[0] || null; log(state.mask ? `Using foreground mask: ${state.mask.name}` : 'Foreground mask removed.'); });
  $('clearFilesButton').addEventListener('click', clearFiles);
  $('resetButton').addEventListener('click', resetFinish);
  $('processButton').addEventListener('click', processStack);
  $('downloadButton').addEventListener('click', downloadResult);
  $('coverage').addEventListener('input', () => { $('coverageValue').textContent = `${$('coverage').value}%`; saveSettings(); });
  $('resolution').addEventListener('change', () => { updateMemoryHint(); saveSettings(); });
  document.querySelectorAll('input[type="checkbox"], input[name="mode"]').forEach((input) => input.addEventListener('change', saveSettings));
  restoreSettings();
  updateMemoryHint();
  renderFiles();
  installPrompt();
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
}

wire();
