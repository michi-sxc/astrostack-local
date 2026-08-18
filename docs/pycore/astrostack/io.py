from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
try:
    import rawpy
except ImportError:  # web build supplies decoded arrays through StackRequest
    rawpy = None  # type: ignore[assignment]


SUPPORTED_EXTENSIONS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".raf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RAW_EXTENSIONS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".raf"}
_CAPTURE_STAMP = re.compile(r"(\d{8})[_-]?(\d{6})(?!\d)")


def acquisition_sort_key(path: Path) -> tuple[str, str]:
    match = _CAPTURE_STAMP.search(path.stem)
    return ((match.group(1) + match.group(2)) if match else path.name.lower(), path.name.lower())


def discover_images(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            found.extend(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return sorted(dict.fromkeys(p.resolve() for p in found), key=acquisition_sort_key)


def read_linear_rgb(path: str | Path, half_size: bool = False) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        if rawpy is None:
            raise RuntimeError("RAW decoding is provided by the browser LibRaw adapter")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                gamma=(1.0, 1.0),
                output_bps=16,
                half_size=half_size,
            )
        return rgb.astype(np.float32) / 65535.0

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"unsupported or damaged image: {path.name}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    scale = float(np.iinfo(bgr.dtype).max) if np.issubdtype(bgr.dtype, np.integer) else 1.0
    return cv2.cvtColor(bgr.astype(np.float32) / scale, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read foreground mask: {path}")
    if mask.shape[::-1] != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32) / 255.0


def write_image(path: str | Path, image: np.ndarray) -> Path:
    raw_path = str(path).strip()
    path = Path(raw_path).resolve()
    # Accept a folder too; the old writer silently created output.tiff beside it.
    if path.is_dir() or raw_path.endswith(("/", "\\")):
        path /= "astrostack.tiff"
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        path = path.with_suffix(".tiff")
        ext = ".tiff"
    clipped = np.clip(image, 0.0, 1.0)
    max_value = 255.0 if ext in {".jpg", ".jpeg"} else 65535.0
    dtype = np.uint8 if max_value == 255.0 else np.uint16
    encoded = np.rint(clipped * max_value).astype(dtype)
    ok = cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
    if not ok or not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"failed to write {path}")
    return path


def write_stack_outputs(
    path: str | Path,
    image: np.ndarray,
    coverage: np.ndarray,
    foreground_mask: np.ndarray | None = None,
) -> Path:
    output = write_image(path, image)
    write_coverage(output.with_name(f"{output.stem}_coverage.png"), coverage)
    if foreground_mask is not None:
        write_coverage(output.with_name(f"{output.stem}_foreground_mask.png"), foreground_mask)
    return output


def write_coverage(path: str | Path, coverage: np.ndarray) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = coverage / max(float(coverage.max()), 1.0)
    ok = cv2.imwrite(str(path), np.rint(normalized * 65535.0).astype(np.uint16))
    if not ok or not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"failed to write {path}")
    return path
