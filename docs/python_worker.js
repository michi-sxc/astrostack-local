/* Pyodide worker: runs the same Python pipeline as the desktop app. */
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/pyodide.mjs';

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
const PY_FILES = [
  '__init__.py', 'alignment.py', 'calibration.py', 'io.py', 'models.py',
  'pipeline.py', 'postprocess.py', 'stacking.py', 'web.py',
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
        const response = await fetch(new URL(`./pycore/astrostack/${name}`, import.meta.url));
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
    paths.push({ path, w: entry.w, h: entry.h, channels });
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
import numpy as np
from astrostack.web import stack_decoded_frames

def _load(item):
    shape = (int(item['h']), int(item['w']), int(item['channels']))
    return np.fromfile(item['path'], dtype=np.float32).reshape(shape)

lights = [_load(item) for item in json.loads(web_lights_json)]
darks = [_load(item) for item in json.loads(web_darks_json)]
mask_item = json.loads(web_mask_json)
mask = None if mask_item is None else _load(mask_item)[..., 0]

def _progress(stage, current, total):
    emit_progress(stage, current, total)

def _log(message):
    emit_log(message)

result = stack_decoded_frames(lights, darks, mask, json.loads(web_settings_json), _progress, _log)
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
  if (event.data?.type !== 'processDecoded') return;
  try {
    await process(event.data);
  } catch (error) {
    send({ type: 'error', message: error?.message || String(error) });
  }
};
