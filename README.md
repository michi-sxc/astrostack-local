# AstroStack

AstroStack is a local astrophotography stacker. The Python package provides the processing pipeline and a Windows GUI. The `docs/` folder contains the browser build for GitHub Pages.

## Desktop app

Run `run_astrostack.bat`. The first run creates `.venv` and installs the dependencies.

1. Select light frames and optional dark frames.
2. Choose a stacking mode. Sigma is the default; Sum and Average are also available.
3. Select a foreground mask if the frame contains trees, buildings, or another fixed object. White marks foreground and black marks sky.
4. Choose an output path and start the stack.

TIFF and PNG output use 16-bit precision. The stack also writes a `_coverage.png` map showing how many aligned samples contributed to each pixel.

## Processing

- Per-pixel coverage normalization prevents drifting edges from being divided by missing samples.
- A middle frame is used as the registration reference to reduce total drift.
- Coarse star matching is followed by native-resolution subpixel refinement.
- Registration quality, spatial coverage, edge residuals, and transform scale are checked per frame.
- Sigma uses a second rejection pass. Sum and Average use the same coverage-aware accumulators.
- Dark frames are combined into a master dark and used for defect correction.
- Foreground protection uses a separate camera-fixed stack and a temporal mask when no mask file is supplied.
- Optional finishing stages provide background-gradient removal, HDR tone mapping, star contrast, chroma/luminance denoise, auto brightness, and chromatic-aberration correction.

Default settings are native resolution, 98% common coverage, Sigma stacking, auto brightness, foreground protection, HDR, star enhancement, chromatic-aberration correction, registration correction, and dynamic denoise. Light-pollution reduction is off by default.

## Command line

```powershell
.\.venv\Scripts\python.exe -m astrostack `
  --lights test-dataset\lights `
  --darks test-dataset\darks `
  --output output\astrostack.tiff `
  --mode sigma
```

Use `python -m astrostack --help` for all switches. Add `--half-size` only for a preview.

## Web app

The `docs/` folder is a static GitHub Pages site. It runs the same Python/OpenCV/SciPy pipeline through Pyodide in a Web Worker.

- RAW/DNG/CR2/CR3/NEF/ARW-family files are decoded in the browser with LibRaw WASM.
- Decoding produces 16-bit RGB with a linear tone curve; PNG conversion is not required.
- Files are not uploaded to a server.
- Native mode keeps the source dimensions. Mobile-safe and Quick preview reduce the working dimensions after RAW decoding.
- The first run downloads and caches the Python runtime and packages.

To publish with GitHub Pages, select **Settings -> Pages -> Deploy from a branch**, choose the default branch, and set the folder to `/docs`.

## Project map

- `astrostack/alignment.py` - star detection, matching, RANSAC, and warping
- `astrostack/stacking.py` - coverage-aware accumulators and rejection
- `astrostack/calibration.py` - dark calibration and bad-pixel repair
- `astrostack/postprocess.py` - gradient removal, HDR, denoise, enhancement, and stretch
- `astrostack/pipeline.py` - frame preparation and orchestration
- `astrostack/gui.py` - Tk desktop interface
- `docs/` - browser interface and worker bridge
- `tests/` - registration, coverage, rejection, and finishing tests

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Decompiled material

The files in `sources/` are JADX output from the supplied APK. They are not a buildable Android project. See [DECOMPILATION_NOTES.md](DECOMPILATION_NOTES.md) for the evidence and the requirements for an Android port.
