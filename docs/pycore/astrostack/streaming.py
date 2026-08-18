"""Native-resolution, two-pass orchestration for the browser bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .alignment import SpatialTransform, estimate_transform, luminance, warp_frame
from .calibration import build_master_dark, calibrate
from .models import AlignmentStats, ProcessingOptions, StackMode, StackRequest, StackResult
from .pipeline import (
    _bottom_connected,
    _geometric_coverage,
    _neutralize_foreground,
    _prepare_frame,
)
from .postprocess import process_image
from .stacking import (
    ClippedAccumulator,
    CoverageAccumulator,
    MomentAccumulator,
    estimate_foreground_mask,
    estimate_temporal_foreground_mask,
    largest_coverage_crop,
)


class StreamingStackSession:
    """Keep only accumulators and the reference; source frames arrive one at a time."""

    def __init__(
        self,
        light_count: int,
        reference_index: int,
        reference: np.ndarray,
        darks: Sequence[np.ndarray],
        mask: np.ndarray | None,
        options: ProcessingOptions,
        progress: Callable[[str, int, int], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.options = options
        self.progress = progress
        self.log = log
        self.light_paths = [Path(f"/astrostack-web/light_{index:05d}.frame") for index in range(light_count)]
        self.dark_paths = [Path(f"/astrostack-web/dark_{index:05d}.frame") for index in range(len(darks))]
        self.reference_path = self.light_paths[reference_index]
        self.reference_index = reference_index
        self.reference = np.asarray(reference, dtype=np.float32)
        self.size = (self.reference.shape[1], self.reference.shape[0])
        frame_data = {path: frame for path, frame in zip(self.dark_paths, darks)}
        frame_data[self.reference_path] = self.reference
        self.request = StackRequest(
            self.light_paths,
            self.dark_paths,
            options,
            progress,
            log,
            frame_data,
            mask,
        )
        self.calibration = build_master_dark(
            self.dark_paths,
            options.half_size,
            self.size,
            progress,
            frame_data,
        )
        calibrate(self.reference, self.calibration)
        for path in self.dark_paths:
            frame_data.pop(path, None)

        self.foreground = None if mask is None else np.asarray(mask, dtype=np.float32)
        self.auto_foreground = options.protect_foreground and mask is None
        self.normalization_mask = self.foreground
        self.source_validity = np.ones(self.reference.shape[:2], np.float32)
        self.robust_sky = options.mode in {StackMode.SIGMA_CLIPPED, StackMode.SUM}
        self.accumulator = (
            MomentAccumulator(self.reference.shape)
            if self.robust_sky or self.auto_foreground
            else CoverageAccumulator(self.reference.shape, StackMode.AVERAGE)
        )
        self.accumulator.add(self.reference, self.source_validity)
        self.foreground_moments = MomentAccumulator(self.reference.shape) if self.foreground is not None or self.auto_foreground else None
        if self.foreground_moments is not None:
            self.foreground_moments.add(self.reference, self.source_validity)
        self.geometric_coverage = np.ones(self.reference.shape[:2], np.float32)
        self.accepted = [self.reference_path]
        self.rejected: dict[Path, str] = {}
        self.alignment: dict[Path, AlignmentStats] = {self.reference_path: AlignmentStats()}
        self.transforms: dict[Path, SpatialTransform] = {self.reference_path: SpatialTransform(np.eye(3, dtype=np.float64))}
        self.sky_mean = self.sky_deviation = self.sky_target = None
        self.foreground_mean = self.foreground_deviation = self.foreground_target = None
        self.base_stack = None
        self.sky_clipped = self.foreground_clipped = None
        self.probe_candidates: list[np.ndarray] = []
        self._first_done = False

    def _log(self, message: str) -> None:
        if self.log:
            self.log(message)

    def _progress(self, stage: str, current: int, total: int) -> None:
        if self.progress:
            self.progress(stage, current, total)

    def add_first(self, index: int, frame: np.ndarray) -> None:
        path = self.light_paths[index]
        if index == self.reference_index:
            return
        self._progress("Aligning and stacking", len(self.accepted), len(self.light_paths) - 1)
        self.request.frame_data[path] = frame
        try:
            prepared = _prepare_frame(path, self.request, self.reference, self.calibration, self.size, self.normalization_mask)
            transform, stats = estimate_transform(
                prepared,
                self.reference,
                self.options.max_alignment_side,
                self.normalization_mask,
                self.options.correct_distortion,
            )
            if stats.rms > 1.15:
                raise ValueError(f"star residual too high ({stats.rms:.2f} px)")
            if abs(stats.rotation_degrees) > 1.35:
                raise ValueError(f"rotation span too large ({stats.rotation_degrees:+.2f} deg)")
            if stats.p90 > 0.42 or stats.edge_p90 > 0.48:
                raise ValueError(f"registration residual too high ({stats.p90:.2f}/{stats.edge_p90:.2f} px)")
            if not 0.95 <= stats.scale <= 1.05:
                raise ValueError(f"implausible frame scale ({stats.scale:.3f})")
            aligned, validity = warp_frame(prepared, transform, self.size, self.source_validity)
            self.accumulator.add(aligned, validity)
            if self.foreground_moments is not None:
                self.foreground_moments.add(prepared, self.source_validity)
            self.geometric_coverage += _geometric_coverage(transform, prepared.shape[:2], self.size)
            self.accepted.append(path)
            self.alignment[path] = stats
            self.transforms[path] = transform
            self._log(
                f"Accepted {path.name}: {stats.inliers} verified stars over {stats.field_coverage:.0%} of field, "
                f"{stats.rotation_degrees:+.2f} deg, rms {stats.rms:.2f} px, p90 {stats.p90:.2f} px, "
                f"edge p90 {stats.edge_p90:.2f} px"
                + (f", fallback {stats.fallback}" if stats.fallback else "")
            )
        except Exception as error:
            self.rejected[path] = str(error) or error.__class__.__name__
            self._log(f"Rejected {path.name}: {self.rejected[path]}")
        finally:
            self.request.frame_data.pop(path, None)

    def finish_first(self) -> list[int]:
        if isinstance(self.accumulator, MomentAccumulator):
            self.sky_mean, self.sky_deviation, self.sky_target = self.accumulator.statistics()
            if not self.robust_sky:
                self.base_stack = self.sky_mean
        else:
            self.base_stack = self.accumulator.finish()
        # first-pass moments are no longer needed once clip thresholds exist
        self.accumulator = None
        if self.foreground_moments is not None:
            self.foreground_mean, self.foreground_deviation, self.foreground_target = self.foreground_moments.statistics()
            if not self.auto_foreground:
                self._make_clippers()
            self.foreground_moments = None
        if self.auto_foreground:
            self.foreground = estimate_temporal_foreground_mask(self.sky_deviation, self.foreground_deviation)
            self._first_done = True
            candidates = [path for path in self.accepted if path != self.reference_path]
            probes = []
            if candidates:
                probes = [min(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[-1])), max(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))]
            return list(dict.fromkeys(int(path.stem.rsplit("_", 1)[-1]) for path in probes))
        self._make_sky_clipper()
        self._first_done = True
        return []

    def add_probe(self, index: int, frame: np.ndarray) -> None:
        if not self.auto_foreground or index == self.reference_index:
            return
        path = self.light_paths[index]
        if path not in self.transforms:
            return
        self.request.frame_data[path] = frame
        try:
            prepared = _prepare_frame(path, self.request, self.reference, self.calibration, self.size, None)
            aligned, _ = warp_frame(prepared, self.transforms[path], self.size)
            self.probe_candidates.append(estimate_foreground_mask(self.reference, prepared, aligned))
        finally:
            self.request.frame_data.pop(path, None)

    def finish_probe(self) -> None:
        if self.auto_foreground:
            probe = np.mean(self.probe_candidates, axis=0) if self.probe_candidates else np.zeros_like(self.foreground)
            self.foreground = _bottom_connected(self.foreground * 0.55 + probe * 0.45)
            if not 0.001 < float(self.foreground.mean()) < 0.45:
                raise ValueError("automatic foreground mask failed; supply a white-foreground mask")
            self._log(f"Automatic foreground mask: {self.foreground.mean():.1%} of frame")
            self._make_clippers()
        self._make_sky_clipper()

    def _make_sky_clipper(self) -> None:
        if self.robust_sky and self.sky_clipped is None:
            self.sky_clipped = ClippedAccumulator(self.sky_mean, self.sky_deviation, self.options.sigma, self.sky_target)

    def _make_clippers(self) -> None:
        if self.foreground_moments is not None and self.foreground_clipped is None:
            self.foreground_clipped = ClippedAccumulator(
                self.foreground_mean,
                self.foreground_deviation,
                min(self.options.sigma, 2.4),
                self.foreground_target,
            )

    def add_second(self, index: int, frame: np.ndarray) -> None:
        if not self._first_done or index == self.reference_index:
            return
        path = self.light_paths[index]
        if path not in self.transforms:
            return
        self.request.frame_data[path] = frame
        try:
            prepared = _prepare_frame(path, self.request, self.reference, self.calibration, self.size, self.normalization_mask)
            aligned, validity = warp_frame(prepared, self.transforms[path], self.size, self.source_validity)
            if self.sky_clipped is not None:
                self.sky_clipped.add(aligned, validity)
            if self.foreground_clipped is not None:
                self.foreground_clipped.add(prepared, self.source_validity)
        finally:
            self.request.frame_data.pop(path, None)

    def finish(self) -> StackResult:
        # Reference must enter the second-pass clipper without another decode.
        if self.sky_clipped is not None:
            self.sky_clipped.add(self.reference, self.source_validity)
        if self.foreground_clipped is not None:
            self.foreground_clipped.add(self.reference, self.source_validity)
        if self.sky_clipped is not None:
            stacked = self.sky_clipped.finish(self.options.mode)
            output_scale = self.sky_clipped.output_scale
            self._log(f"Robust sky downweighting: {self.sky_clipped.rejected_fraction:.2%} of covered samples")
        else:
            stacked = self.base_stack
            output_scale = 1.0
        if self.foreground is not None and self.foreground_clipped is not None:
            foreground_stack = self.foreground_clipped.finish(StackMode.AVERAGE) * output_scale
            foreground_stack = _neutralize_foreground(foreground_stack, self.foreground)
            reference_stack = self.reference * output_scale
            static_stack = foreground_stack + (luminance(reference_stack) - luminance(foreground_stack))[..., None]
            self._log(f"Robust foreground downweighting: {self.foreground_clipped.rejected_fraction:.2%} of samples")
            stacked = stacked * (1.0 - self.foreground[..., None]) + static_stack * self.foreground[..., None]

        crop = None
        if self.options.auto_crop and len(self.accepted) > 1:
            crop = largest_coverage_crop(self.geometric_coverage, self.options.min_coverage)
            if crop:
                x0, y0, x1, y1 = crop
                stacked = stacked[y0:y1, x0:x1]
                self.geometric_coverage = self.geometric_coverage[y0:y1, x0:x1]
                if self.foreground is not None:
                    self.foreground = self.foreground[y0:y1, x0:x1]
                self._log(f"Coverage crop: {x1 - x0} x {y1 - y0} at {self.options.min_coverage:.0%} minimum")
        self._progress("Finishing image", 1, 1)
        image = process_image(stacked, self.options, self.foreground, self.calibration.dark_noise)
        self._log(f"Finished: {len(self.accepted)} accepted, {len(self.rejected)} rejected")
        return StackResult(image, self.geometric_coverage, self.accepted, self.rejected, crop, self.alignment, self.foreground)
