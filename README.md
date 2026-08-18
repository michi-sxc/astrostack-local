# AstroStack Studio

A clean, runnable astrophotography stacker built from the supplied DNG workflow. The files in `sources/` are decompiler output, not a buildable Android project, so this implementation is independent and intentionally keeps the processing core separate from the desktop UI.

## Run the app

Double-click `run_astrostack.bat`.

The launcher creates `.venv` and installs the four dependencies on its first run. In the window:

1. Choose all light frames and optional dark frames.
2. Keep **Sigma** selected for the best default result.
3. Leave the fast half-resolution preview disabled for the full RAW resolution.
4. Optionally choose a foreground mask, then choose a TIFF output and click **Stack images**. The stack is written immediately; **Save current result** re-exports the last completed stack if you change the destination.

TIFF and PNG are written at 16-bit precision. The app also writes a `_coverage.png` map; brighter pixels received more aligned samples.

## What changed

- Per-pixel coverage normalization fixes the drifting-edge defect. A missing warped pixel is never treated as a black exposure.
- The chronological middle frame is the reference, reducing maximum drift on both sides.
- Defect-resistant coarse registration followed by native-resolution subpixel refinement prevents RAW hot pixels from winning the solve while keeping edge stars precise.
- Registration residuals are measured in native pixels across the full field. Bad edge residuals, weak spatial coverage, and implausible transforms reject a frame.
- **Average**, robust coverage-compensated **Sum**, and two-pass **Sigma** stacking modes are available.
- Sum and Sigma run an exact second rejection pass, so fixed defects cannot become colored arcs after sky alignment.
- Dark calibration finds defects repeated across the supplied darks and subtracts their shared fixed pattern.
- Foreground protection makes a separate two-pass camera-fixed stack. The automatic mask uses the full sequence plus opposite drift directions, stays ground-connected, preserves sky holes, and saves beside the result for inspection.
- Optional light-pollution gradient removal, HDR tone compression, star enhancement, halo-safe adaptive luminance/chroma denoising, auto brightness, and common-coverage cropping are included.
- RAW decoding stays linear until the finishing stage, and master dark calibration happens before alignment.

The safe defaults are full resolution, 98% common coverage, robust stacking, auto brightness, and mild dynamic denoise. Light-pollution removal, HDR, and star enhancement are opt-in because combining aggressive finishing stages can amplify noise and edge gradients. Half resolution is only a fast preview option.

## Command line

```powershell
.\.venv\Scripts\python.exe -m astrostack `
  --lights test-dataset\lights `
  --darks test-dataset\darks `
  --output output\astrostack_improved.tiff `
  --mode sigma
```

Use `python -m astrostack --help` for individual feature switches. Add `--half-size` only for a fast preview. A custom foreground mask is a grayscale image where white means camera-fixed foreground and black means sky; passing `--mask` enables the separate foreground stack.

## Project map

- `astrostack/alignment.py`: star detection, asterism matching, RANSAC, and warping
- `astrostack/stacking.py`: coverage-aware accumulators, two-pass rejection, masking, and cropping
- `astrostack/calibration.py`: dark calibration and bad-pixel repair
- `astrostack/postprocess.py`: gradient removal, HDR, denoise, star enhancement, and stretch
- `astrostack/pipeline.py`: memory-bounded orchestration and frame rejection
- `astrostack/gui.py`: threaded Tk desktop app
- `tests/`: synthetic registration, edge-coverage, rejection, and finishing tests

## Supplied-dataset validation

The included 42-frame set completes in Sigma mode with spatially verified registration and strict common-coverage cropping. Full-resolution validation output and its coverage map are written in `output/`.

Run the regression suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Android note

This is currently a Windows desktop app. The processing modules are the usable reference implementation for an Android port, but rebuilding the friend's exact Android app still requires the original Gradle project and its native ABI libraries. See `DECOMPILATION_NOTES.md` for the evidence.

## Mobile web app

The `docs/` folder is a static, offline-first browser build for GitHub Pages. It keeps the stack local to the phone, uses a Web Worker for alignment and finishing, caches the app shell, and exports a PNG without a server.

To publish it, push the repository to GitHub and choose **Settings -> Pages -> Deploy from a branch**, then select the default branch and the `/docs` folder. Open the resulting Pages URL on the phone and use **Install** when the browser offers it.

The web build includes a bundled LibRaw WebAssembly decoder for DNG/CR2/NEF/ARW-family inputs. RAW files are decoded locally to 16-bit RGB with automatic brightness disabled and a linear tone curve before stacking, so conversion to PNG is not required. The browser now runs the same Python/OpenCV/SciPy pipeline from `astrostack/` through Pyodide; the old JavaScript approximation is no longer used for registration, stacking, masking, or finishing. The first stack downloads and caches the Python runtime and packages. Large native-resolution sets can still exceed a phone's memory; Mobile-safe mode reduces the working dimensions after RAW decoding.
