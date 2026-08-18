# AstroStack Local web app

This is a static GitHub Pages build. It processes browser-readable image files locally and caches the app shell with a service worker.

The browser pipeline includes sigma/sum/average stacking, dark-frame calibration, full-field star registration, common-coverage cropping, foreground protection, auto brightness, light-pollution reduction, HDR tone mapping, star enhancement, adaptive chroma/luminance denoise, and chromatic-aberration correction. Processing runs in a Web Worker so the mobile UI stays responsive. RAW/DNG/CR2/NEF/ARW-family inputs are decoded in-browser with the bundled LibRaw WebAssembly build before the same Python/OpenCV/SciPy pipeline used by the desktop app runs through Pyodide. The first stack downloads the Python runtime and packages; the browser cache keeps them for subsequent use.

## Publish with GitHub Pages

Set **Settings → Pages → Deploy from a branch**, choose the repository's default branch, and choose `/docs` as the folder. The app uses relative URLs so it works from a project subpath.

## Input note

RAW/DNG/CR2/NEF/ARW-family files are decoded locally with LibRaw WASM. The decoder demosaics to 16-bit RGB with camera white balance, no automatic brightness, and a linear tone curve before the stacker applies its own finishing controls. Large RAW sets can use substantial phone memory; Mobile-safe or Quick preview reduces the working dimensions after decoding, without requiring a pre-converted file.
