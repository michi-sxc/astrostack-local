# AstroStack web app

Static GitHub Pages build for local astrophotography stacking. The interface runs the Python/OpenCV/SciPy pipeline in a Web Worker and caches the app shell.

## Supported input

JPG, PNG, DNG, CR2, CR3, NEF, ARW, RW2, ORF, RAF, and other LibRaw-supported camera files can be selected directly. RAW files are decoded in the browser with LibRaw WASM to 16-bit RGB with a linear tone curve. No pre-conversion to PNG is required.

## Processing options

Sigma, Sum, and Average stacking; dark calibration; star registration; common-coverage cropping; foreground protection; auto brightness; light-pollution reduction; HDR tone mapping; star enhancement; dynamic denoise; chromatic-aberration correction; and registration-distortion correction.

Native resolution keeps the decoded source dimensions and streams one RAW frame at a time. Mobile-safe and Quick preview reduce the working dimensions after decoding. Native-resolution sets can require substantial phone memory.

## Deploy

In GitHub, open **Settings -> Pages**, choose **Deploy from a branch**, select the default branch, and set the folder to `/docs`. The app uses relative URLs, so it works from a project subpath.

The first stack downloads the Pyodide runtime and packages. Later runs use the browser cache. Frames and results stay on the device; there is no upload endpoint.
