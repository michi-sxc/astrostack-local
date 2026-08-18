/* Pyodide worker: runs the same Python pipeline as the desktop app. */
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/pyodide.mjs';

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
const PY_FILES = [
  '__init__.py', 'alignment.py', 'calibration.py', 'io.py', 'models.py',
  'pipeline.py', 'postprocess.py', 'stacking.py', 'streaming.py', 'web.py',
];

let runtimePromise;

function send(message, transfer = []) { self.postMessage(message, transfer); }

async function runtime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      send({ type: 'log', message: 'Loading Python stacker and OpenCV in the browser (first run downloads the local runtime).' });
      const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
      await pyodide.loadPackage(['scipy', 'opencv-python']);
      pyodide.FS.mkdirTree('/astrostack');
      for (const name of PY_FILES) {
        const response = await fetch(new URL(`./pycore/astrostack/${name}?v=stream312`, import.meta.url));
        if (!response.ok) throw new Error(`Could not load Python module ${name}`);
        pyodide.FS.writeFile(`/astrostack/${name}`, await response.text());
      }
      pyodide.runPython("import sys; sys.path.insert(0, '/'); import astrostack.web");
      return pyodide;
    })();
  }
  return runtimePromise;
}

function writeFrames(pyodide, entries, prefix, channels = 3) {
  const paths = [];
  entries.forEach((entry, index) => {
    const path = `/tmp/${prefix}-${index}.f32`;
    pyodide.FS.writeFile(path, new Uint8Array(entry.buffer));
    entry.buffer = null;
    paths.push({ path, w: entry.w, h: entry.h, channels, dtype: entry.dtype || 'float32' });
  });
  return paths;
}

async function process(job) {
  const pyodide = await runtime();
  const lights = writeFrames(pyodide, job.lights, 'light');
  const darks = writeFrames(pyodide, job.darks || [], 'dark');
  const mask = job.mask ? writeFrames(pyodide, [job.mask], 'mask', 1)[0] : null;
  const settings = JSON.stringify(job.settings || {});

  pyodide.globals.set('web_settings_json', settings);
  pyodide.globals.set('web_lights_json', JSON.stringify(lights));
  pyodide.globals.set('web_darks_json', JSON.stringify(darks));
  pyodide.globals.set('web_mask_json', JSON.stringify(mask));
  pyodide.globals.set('emit_progress', (stage, current, total) => {
    const percent = Math.max(1, Math.min(95, Math.round((Number(current) / Math.max(1, Number(total))) * 95)));
    send({ type: 'progress', label: String(stage), percent });
  });
  pyodide.globals.set('emit_log', (message) => send({ type: 'log', message: String(message) }));

  await pyodide.runPythonAsync(`
import json
from astrostack.web import stack_decoded_sources

mask_item = json.loads(web_mask_json)

def _progress(stage, current, total):
    try:
        emit_progress(stage, current, total)
    except Exception:
        pass

def _log(message):
    try:
        emit_log(message)
    except Exception:
        pass

result = stack_decoded_sources(
    json.loads(web_lights_json),
    json.loads(web_darks_json),
    mask_item,
    json.loads(web_settings_json),
    _progress,
    _log,
)
result.image.astype(np.float32).tofile('/tmp/astrostack-result.f32')
with open('/tmp/astrostack-result.json', 'w', encoding='utf-8') as handle:
    json.dump({'width': int(result.image.shape[1]), 'height': int(result.image.shape[0])}, handle)
`);

  const info = JSON.parse(new TextDecoder().decode(pyodide.FS.readFile('/tmp/astrostack-result.json')));
  const data = pyodide.FS.readFile('/tmp/astrostack-result.f32');
  const buffer = data.slice().buffer;
  send({ type: 'pythonDone', width: info.width, height: info.height, buffer }, [buffer]);
}

function writeIncoming(pyodide, entry, path) {
  pyodide.FS.writeFile(path, new Uint8Array(entry.buffer));
  entry.buffer = null;
  return { path, w: entry.w, h: entry.h, channels: entry.channels || 3, dtype: entry.dtype || 'float32' };
}

function setCallbacks(pyodide) {
  pyodide.globals.set('emit_progress', (stage, current, total) => {
    const percent = Math.max(1, Math.min(95, Math.round((Number(current) / Math.max(1, Number(total))) * 95)));
    send({ type: 'progress', label: String(stage), percent });
  });
  pyodide.globals.set('emit_log', (message) => send({ type: 'log', message: String(message) }));
}

async function streamStart(job) {
  const pyodide = await runtime();
  setCallbacks(pyodide);
  const reference = writeIncoming(pyodide, job.reference, '/tmp/stream-reference.bin');
  const darks = (job.darks || []).map((entry, index) => writeIncoming(pyodide, entry, `/tmp/stream-dark-${index}.bin`));
  const mask = job.mask ? writeIncoming(pyodide, job.mask, '/tmp/stream-mask.bin') : null;
  pyodide.globals.set('stream_settings_json', JSON.stringify(job.settings || {}));
  pyodide.globals.set('stream_reference_json', JSON.stringify(reference));
  pyodide.globals.set('stream_darks_json', JSON.stringify(darks));
  pyodide.globals.set('stream_mask_json', JSON.stringify(mask));
  pyodide.globals.set('stream_count', Number(job.count));
  pyodide.globals.set('stream_reference_index', Number(job.referenceIndex));
  await pyodide.runPythonAsync(`
import json
import numpy as np
from astrostack.web import _options
from astrostack.streaming import StreamingStackSession

def _stream_load(item):
    shape = (int(item['h']), int(item['w']), int(item['channels']))
    data = np.fromfile(item['path'], dtype=np.dtype(item['dtype'])).reshape(shape)
    return data.astype(np.float32) / 65535.0 if data.dtype != np.float32 else data

_stream_ref = _stream_load(json.loads(stream_reference_json))
_stream_darks = [_stream_load(item) for item in json.loads(stream_darks_json)]
_stream_mask_item = json.loads(stream_mask_json)
_stream_mask = None if _stream_mask_item is None else _stream_load(_stream_mask_item)[..., 0]
_stream_session = StreamingStackSession(
    int(stream_count),
    int(stream_reference_index),
    _stream_ref,
    _stream_darks,
    _stream_mask,
    _options(json.loads(stream_settings_json)),
    lambda stage, current, total: emit_progress(stage, current, total),
    lambda message: emit_log(message),
)
del _stream_ref, _stream_darks, _stream_mask
`);
  send({ type: 'streamReady' });
}

async function streamFrame(message, stage) {
  const pyodide = await runtime();
  const entry = writeIncoming(pyodide, message.frame, '/tmp/stream-current.bin');
  pyodide.globals.set('stream_current_json', JSON.stringify(entry));
  pyodide.globals.set('stream_index', Number(message.index));
  await pyodide.runPythonAsync(`
import json
import numpy as np

item = json.loads(stream_current_json)
shape = (int(item['h']), int(item['w']), int(item['channels']))
frame = np.fromfile(item['path'], dtype=np.dtype(item['dtype'])).reshape(shape)
if frame.dtype != np.float32:
    frame = frame.astype(np.float32) / 65535.0
if '${stage}' == 'first':
    _stream_session.add_first(int(stream_index), frame)
elif '${stage}' == 'probe':
    _stream_session.add_probe(int(stream_index), frame)
else:
    _stream_session.add_second(int(stream_index), frame)
del frame
`);
  send({ type: 'streamReady' });
}

async function streamFirstDone() {
  const pyodide = await runtime();
  await pyodide.runPythonAsync(`
import json
probe_indices = _stream_session.finish_first()
accepted_indices = [int(path.stem.rsplit('_', 1)[-1]) for path in _stream_session.accepted]
with open('/tmp/stream-ready.json', 'w', encoding='utf-8') as handle:
    json.dump({'indices': probe_indices, 'accepted': accepted_indices}, handle)
`);
  const data = pyodide.FS.readFile('/tmp/stream-ready.json');
  send({ type: 'streamFirstReady', ...JSON.parse(new TextDecoder().decode(data)) });
}

async function streamProbeDone() {
  const pyodide = await runtime();
  await pyodide.runPythonAsync("_stream_session.finish_probe()");
  send({ type: 'streamReady' });
}

async function streamFinish() {
  const pyodide = await runtime();
  await pyodide.runPythonAsync(`
result = _stream_session.finish()
result.image.astype(np.float32).tofile('/tmp/astrostack-result.f32')
with open('/tmp/astrostack-result.json', 'w', encoding='utf-8') as handle:
    json.dump({'width': int(result.image.shape[1]), 'height': int(result.image.shape[0])}, handle)
`);
  const info = JSON.parse(new TextDecoder().decode(pyodide.FS.readFile('/tmp/astrostack-result.json')));
  const data = pyodide.FS.readFile('/tmp/astrostack-result.f32');
  const buffer = data.slice().buffer;
  send({ type: 'pythonDone', width: info.width, height: info.height, buffer }, [buffer]);
}

self.onmessage = async (event) => {
  const type = event.data?.type;
  if (!['processDecoded', 'streamStart', 'streamFrame', 'streamFirstDone', 'streamProbeDone', 'streamFinish'].includes(type)) return;
  try {
    if (type === 'processDecoded') await process(event.data);
    else if (type === 'streamStart') await streamStart(event.data);
    else if (type === 'streamFrame') await streamFrame(event.data, event.data.stage);
    else if (type === 'streamFirstDone') await streamFirstDone();
    else if (type === 'streamProbeDone') await streamProbeDone();
    else await streamFinish();
  } catch (error) {
    const detail = error?.message || String(error);
    const message = /ArrayMemoryError|Unable to allocate|out of memory/i.test(detail)
      ? 'The browser ran out of working memory. Choose Mobile-safe or Quick preview, then run the stack again.'
      : detail;
    send({ type: 'error', message });
  }
};
