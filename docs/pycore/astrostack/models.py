from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


class StackMode(str, Enum):
    AVERAGE = "average"
    SUM = "sum"
    SIGMA_CLIPPED = "sigma"


ProgressCallback = Callable[[str, int, int], None]
LogCallback = Callable[[str], None]


@dataclass(slots=True)
class ProcessingOptions:
    mode: StackMode = StackMode.SIGMA_CLIPPED
    half_size: bool = False
    sigma: float = 3.0
    min_coverage: float = 0.98
    auto_crop: bool = True
    auto_brightness: bool = True
    protect_foreground: bool = False
    foreground_mask: Path | None = None
    reduce_light_pollution: bool = False
    hdr: bool = False
    enhance_stars: bool = False
    dynamic_denoise: bool = True
    correct_distortion: bool = True
    correct_chromatic_aberration: bool = True
    save_coverage: bool = True
    max_alignment_side: int = 4096


@dataclass(slots=True)
class AlignmentStats:
    stars: int = 0
    matches: int = 0
    inliers: int = 0
    rms: float = 0.0
    p90: float = 0.0
    edge_p90: float = 0.0
    field_coverage: float = 0.0
    rotation_degrees: float = 0.0
    scale: float = 1.0
    distortion_improvement: float = 0.0
    fallback: str | None = None


@dataclass(slots=True)
class StackResult:
    image: np.ndarray
    coverage: np.ndarray
    accepted: list[Path] = field(default_factory=list)
    rejected: dict[Path, str] = field(default_factory=dict)
    crop: tuple[int, int, int, int] | None = None
    alignment: dict[Path, AlignmentStats] = field(default_factory=dict)
    foreground_mask: np.ndarray | None = None


@dataclass(slots=True)
class StackRequest:
    lights: Sequence[Path]
    darks: Sequence[Path] = ()
    options: ProcessingOptions = field(default_factory=ProcessingOptions)
    progress: ProgressCallback | None = None
    log: LogCallback | None = None
    # browser adapter keeps decoded frames in memory instead of fake files
    frame_data: Mapping[Path, np.ndarray] | None = None
    mask_data: np.ndarray | None = None
