"""Small in-memory adapter used by the Pyodide browser build."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Callable, Sequence

import numpy as np

from .models import ProcessingOptions, StackMode, StackRequest, StackResult
from .pipeline import stack_images


class _FrameStore(Mapping[Path, np.ndarray]):
    """Lazy file-backed frames; only the frame in the current pipeline stage is resident."""

    def __init__(self, sources: Mapping[Path, tuple[str, tuple[int, int, int], str]]) -> None:
        self.sources = dict(sources)

    def __getitem__(self, path: Path) -> np.ndarray:
        filename, shape, dtype = self.sources[path]
        data = np.fromfile(filename, dtype=np.dtype(dtype)).reshape(shape)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / 65535.0
        return data

    def __iter__(self):
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)


def _options(settings: Mapping[str, object]) -> ProcessingOptions:
    return ProcessingOptions(
        mode=StackMode(str(settings.get("mode", "sigma"))),
        min_coverage=float(settings.get("coverage", 98)) / 100.0,
        auto_crop=True,
        auto_brightness=bool(settings.get("autoBrightness", True)),
        protect_foreground=bool(settings.get("protectForeground", True)),
        reduce_light_pollution=bool(settings.get("lightPollution", False)),
        hdr=bool(settings.get("hdr", True)),
        enhance_stars=bool(settings.get("starEnhancement", True)),
        dynamic_denoise=bool(settings.get("dynamicDenoise", True)),
        correct_distortion=bool(settings.get("distortion", True)),
        correct_chromatic_aberration=bool(settings.get("chromaticAberration", True)),
        max_alignment_side=4096,
    )


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
    lights_paths = [Path(f"/astrostack-web/light_{index:05d}.frame") for index in range(len(lights))]
    dark_paths = [Path(f"/astrostack-web/dark_{index:05d}.frame") for index in range(len(darks))]
    frame_data = {
        **{path: np.asarray(frame, dtype=np.float32) for path, frame in zip(lights_paths, lights)},
        **{path: np.asarray(frame, dtype=np.float32) for path, frame in zip(dark_paths, darks)},
    }
    options = _options(config)
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


def stack_decoded_sources(
    lights: Sequence[Mapping[str, object]],
    darks: Sequence[Mapping[str, object]] = (),
    foreground_mask: Mapping[str, object] | None = None,
    settings: Mapping[str, object] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> StackResult:
    """Run the stacker from browser-owned files without building a frame cube."""
    config = settings or {}
    light_paths = [Path(f"/astrostack-web/light_{index:05d}.frame") for index in range(len(lights))]
    dark_paths = [Path(f"/astrostack-web/dark_{index:05d}.frame") for index in range(len(darks))]

    def source_map(paths: Sequence[Path], entries: Sequence[Mapping[str, object]]) -> dict[Path, tuple[str, tuple[int, int, int], str]]:
        return {
            path: (
                str(entry["path"]),
                (int(entry["h"]), int(entry["w"]), int(entry["channels"])),
                str(entry.get("dtype", "float32")),
            )
            for path, entry in zip(paths, entries)
        }

    mask = None
    if foreground_mask is not None:
        shape = (int(foreground_mask["h"]), int(foreground_mask["w"]))
        mask = np.fromfile(str(foreground_mask["path"]), dtype=np.dtype(str(foreground_mask.get("dtype", "float32")))).reshape(shape)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32) / 65535.0
    request = StackRequest(
        light_paths,
        dark_paths,
        _options(config),
        progress,
        log,
        _FrameStore({**source_map(light_paths, lights), **source_map(dark_paths, darks)}),
        mask,
    )
    return stack_images(request)
