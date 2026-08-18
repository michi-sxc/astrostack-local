"""Small in-memory adapter used by the Pyodide browser build."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .models import ProcessingOptions, StackMode, StackRequest, StackResult
from .pipeline import stack_images


def stack_decoded_frames(
    lights: Sequence[np.ndarray],
    darks: Sequence[np.ndarray] = (),
    foreground_mask: np.ndarray | None = None,
    settings: Mapping[str, object] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> StackResult:
    """Run the desktop pipeline on browser-decoded linear RGB frames."""
    config = settings or {}
    mode = StackMode(str(config.get("mode", "sigma")))
    lights_paths = [Path(f"/astrostack-web/light_{index:05d}.frame") for index in range(len(lights))]
    dark_paths = [Path(f"/astrostack-web/dark_{index:05d}.frame") for index in range(len(darks))]
    frame_data = {
        **{path: np.asarray(frame, dtype=np.float32) for path, frame in zip(lights_paths, lights)},
        **{path: np.asarray(frame, dtype=np.float32) for path, frame in zip(dark_paths, darks)},
    }
    options = ProcessingOptions(
        mode=mode,
        min_coverage=float(config.get("coverage", 98)) / 100.0,
        auto_crop=True,
        auto_brightness=bool(config.get("autoBrightness", True)),
        protect_foreground=bool(config.get("protectForeground", True)),
        reduce_light_pollution=bool(config.get("lightPollution", False)),
        hdr=bool(config.get("hdr", True)),
        enhance_stars=bool(config.get("starEnhancement", True)),
        dynamic_denoise=bool(config.get("dynamicDenoise", True)),
        correct_distortion=bool(config.get("distortion", True)),
        correct_chromatic_aberration=bool(config.get("chromaticAberration", True)),
        max_alignment_side=4096,
    )
    request = StackRequest(
        lights_paths,
        dark_paths,
        options,
        progress,
        log,
        frame_data,
        foreground_mask,
    )
    return stack_images(request)
