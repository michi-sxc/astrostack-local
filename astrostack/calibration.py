from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from .io import read_linear_rgb


@dataclass(slots=True)
class CalibrationData:
    master_dark: np.ndarray | None = None
    bad_pixels: np.ndarray | None = None
    dark_count: int = 0
    dark_noise: float = 0.0


def build_master_dark(
    paths: Sequence[Path],
    half_size: bool,
    target_size: tuple[int, int],
    progress: Callable[[str, int, int], None] | None = None,
) -> CalibrationData:
    if not paths:
        return CalibrationData()
    frames: list[np.ndarray] = []
    for index, path in enumerate(paths, 1):
        frame = read_linear_rgb(path, half_size)
        if frame.shape[1::-1] != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if progress:
            progress("Calibrating darks", index, len(paths))
    master = np.median(np.stack(frames), axis=0).astype(np.float32)
    local = cv2.medianBlur(master, 3)
    residual = np.max(master - local, axis=2)
    median = float(np.median(residual))
    noise = float(np.median(np.abs(residual - median))) * 1.4826
    threshold = max(median + 10.0 * noise, float(np.percentile(residual, 99.99)))
    bad_pixels = residual > threshold
    if len(frames) >= 2:
        stable = np.ones(master.shape[:2], bool)
        for frame in frames:
            detail = np.max(np.abs(frame - cv2.medianBlur(frame, 3)), axis=2)
            center = float(np.median(detail))
            spread = float(np.median(np.abs(detail - center))) * 1.4826
            stable &= detail > center + max(3.5 * spread, 1e-4)
        # Intersection keeps this wider defect map safe even with just two darks.
        bad_pixels |= stable
    bad_pixels = bad_pixels.astype(np.uint8)
    bad_pixels = cv2.dilate(bad_pixels, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
    # Dont stamp two-frame shot noise into every light; repair defects, subtract smooth dark signal.
    clean_master = master.copy()
    local_master = cv2.medianBlur(master, 3)
    clean_master[bad_pixels] = local_master[bad_pixels]
    dark_signal = cv2.GaussianBlur(clean_master, (0, 0), 0.8) if len(frames) >= 2 else None
    # dark chroma sets a useful floor for foreground color denoise
    dark_detail = clean_master - cv2.GaussianBlur(clean_master, (0, 0), 1.1)
    dark_chroma = dark_detail - np.dot(dark_detail, np.array((0.2126, 0.7152, 0.0722), np.float32))[..., None]
    center = float(np.median(dark_chroma))
    dark_noise = float(np.median(np.abs(dark_chroma - center))) * 1.4826
    return CalibrationData(dark_signal, bad_pixels, len(frames), dark_noise)


def calibrate(frame: np.ndarray, calibration: CalibrationData) -> np.ndarray:
    if calibration.master_dark is not None:
        np.maximum(frame - calibration.master_dark, 0.0, out=frame)
    if calibration.bad_pixels is not None and np.any(calibration.bad_pixels):
        repaired = cv2.medianBlur(frame, 3)
        frame[calibration.bad_pixels] = repaired[calibration.bad_pixels]
    return frame
