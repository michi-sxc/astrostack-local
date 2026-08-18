from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np

from .alignment import SpatialTransform, estimate_transform, luminance, prepare_reference, registration_confidence, warp_frame, warp_mask
from .calibration import build_master_dark, calibrate
from .io import acquisition_sort_key, read_linear_rgb, read_mask
from .models import AlignmentStats, StackMode, StackRequest, StackResult
from .postprocess import process_image
from .stacking import (
    ClippedAccumulator,
    CoverageAccumulator,
    MomentAccumulator,
    estimate_foreground_mask,
    estimate_temporal_foreground_mask,
    largest_coverage_crop,
)


def _notify(request: StackRequest, stage: str, current: int, total: int) -> None:
    if request.progress:
        request.progress(stage, current, total)


def _log(request: StackRequest, message: str) -> None:
    if request.log:
        request.log(message)


def _read_source(path: Path, request: StackRequest) -> np.ndarray:
    if request.frame_data is not None and path in request.frame_data:
        return request.frame_data[path].copy().astype(np.float32, copy=False)
    return read_linear_rgb(path, request.options.half_size)


def _match_exposure(
    frame: np.ndarray,
    reference: np.ndarray,
    foreground: np.ndarray | None,
    reference_luminance: np.ndarray | None = None,
    reference_sample: np.ndarray | None = None,
) -> np.ndarray:
    cur_lum = luminance(frame)
    valid = np.ones(frame.shape[:2], bool) if foreground is None else foreground < 0.25
    # Sparse sampling is plenty for robust photometric normalization.
    stride = max(1, int(np.sqrt(valid.size / 250_000)))
    if reference_sample is None:
        ref_lum = luminance(reference) if reference_luminance is None else reference_luminance
        reference_sample = ref_lum[::stride, ::stride][valid[::stride, ::stride]]
    cur_sample = cur_lum[::stride, ::stride][valid[::stride, ::stride]]
    if len(ref_sample) < 100:
        return frame
    ref_low, ref_high = np.percentile(ref_sample, (20.0, 92.0))
    cur_low, cur_high = np.percentile(cur_sample, (20.0, 92.0))
    gain = np.clip((ref_high - ref_low) / max(float(cur_high - cur_low), 1e-6), 0.55, 1.8)
    return np.maximum((frame - float(cur_low)) * gain + float(ref_low), 0.0).astype(np.float32)


def _prepare_frame(
    path: Path,
    request: StackRequest,
    reference: np.ndarray,
    calibration,
    size: tuple[int, int],
    foreground: np.ndarray | None,
    reference_luminance: np.ndarray | None = None,
    reference_sample: np.ndarray | None = None,
) -> np.ndarray:
    frame = _read_source(path, request)
    if frame.shape != reference.shape:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return _match_exposure(calibrate(frame, calibration), reference, foreground, reference_luminance, reference_sample)


def _exposure_reference_sample(reference: np.ndarray, foreground: np.ndarray | None) -> np.ndarray:
    luminance_plane = luminance(reference)
    valid = np.ones(luminance_plane.shape, bool) if foreground is None else foreground < 0.25
    stride = max(1, int(np.sqrt(valid.size / 250_000)))
    return luminance_plane[::stride, ::stride][valid[::stride, ::stride]]


def _geometric_coverage(transform: SpatialTransform, source_shape: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    return warp_mask(np.ones(source_shape, np.float32), transform, size)


def _bottom_connected(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.15).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    bottom_labels = np.unique(labels[-max(5, mask.shape[0] // 200) :])
    minimum_area = max(100, mask.size // 5000)
    for label in bottom_labels:
        if 0 < label < count and stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            keep[labels == label] = 1
    # guard band catches wind/camera drift at the silhouette without unioning whole masks
    guard = max(3, int(min(mask.shape) * 0.018))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1)))
    # narrow feather keeps sky stars out of the camera layer
    expanded = cv2.dilate(mask.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1)))
    return expanded * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), max(1.2, min(mask.shape) * 0.00055))


def _probe_foreground_mask(
    request: StackRequest,
    accepted: list[Path],
    transforms: dict[Path, SpatialTransform],
    reference_path: Path,
    reference: np.ndarray,
    calibration,
    size: tuple[int, int],
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    probes = [path for path in (min(accepted, key=acquisition_sort_key), max(accepted, key=acquisition_sort_key)) if path != reference_path]
    for path in dict.fromkeys(probes):
        frame = _prepare_frame(path, request, reference, calibration, size, None)
        aligned, _ = warp_frame(frame, transforms[path], size)
        candidates.append(estimate_foreground_mask(reference, frame, aligned))
    if not candidates:
        return np.zeros(reference.shape[:2], np.float32)
    # average keeps a centered boundary without unioning endpoint offsets
    return _bottom_connected(np.mean(candidates, axis=0))


def _neutralize_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = mask > 0.65
    if selected.sum() < 100:
        return image
    samples = image[selected]
    luminance_samples = samples @ np.array((0.2126, 0.7152, 0.0722), np.float32)
    low, high = np.percentile(luminance_samples, (15.0, 75.0))
    samples = samples[(luminance_samples >= low) & (luminance_samples <= high)]
    if len(samples) < 100:
        return image
    color = np.median(samples, axis=0)
    neutral = float(np.mean(color))
    # Mild gray-world balance cools lamp cast without swinging dark foliage blue.
    gains = np.clip(np.power(neutral / np.maximum(color, 1e-6), 0.30), 0.85, 1.18)
    return (image * gains).astype(np.float32)


def stack_images(request: StackRequest) -> StackResult:
    virtual = request.frame_data or {}

    def canonical(path: Path) -> Path:
        if path in virtual:
            return path
        return path.resolve()

    lights = sorted((canonical(Path(path)) for path in request.lights), key=acquisition_sort_key)
    darks = sorted((canonical(Path(path)) for path in request.darks), key=acquisition_sort_key)
    if not lights:
        raise ValueError("select at least one light frame")
    missing = [path for path in (*lights, *darks) if path not in virtual and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input: {missing[0]}")

    reference_index = len(lights) // 2
    reference_path = lights[reference_index]
    _log(request, f"Reference: {reference_path.name} (middle frame minimizes edge drift)")
    _notify(request, "Decoding reference", 0, len(lights))
    reference = _read_source(reference_path, request)
    size = reference.shape[1], reference.shape[0]
    calibration = build_master_dark(darks, request.options.half_size, size, request.progress, request.frame_data)
    reference = calibrate(reference, calibration)
    if calibration.dark_count:
        strategy = "dark subtraction and bad-pixel repair" if calibration.master_dark is not None else "bad-pixel repair"
        _log(request, f"Calibration: {strategy} from {calibration.dark_count} dark frame(s); chroma floor {calibration.dark_noise:.5f}")

    foreground: np.ndarray | None = None
    auto_foreground = request.options.protect_foreground and request.options.foreground_mask is None and request.mask_data is None
    if request.options.protect_foreground:
        if request.mask_data is not None or request.options.foreground_mask:
            foreground = request.mask_data.copy() if request.mask_data is not None else read_mask(request.options.foreground_mask, size)
            _log(request, "Foreground protection: custom mask")
        else:
            _log(request, "Foreground protection: temporal automatic mask")
    normalization_mask = foreground
    reference_sample = _exposure_reference_sample(reference, normalization_mask)
    reference_field = prepare_reference(reference, request.options.max_alignment_side, normalization_mask)

    # Keep a complete sky estimate behind the foreground; alpha errors then reveal sky, not black holes.
    source_validity = np.ones(reference.shape[:2], np.float32)
    robust_sky = request.options.mode in {StackMode.SIGMA_CLIPPED, StackMode.SUM}
    accumulator = MomentAccumulator(reference.shape) if robust_sky or auto_foreground else CoverageAccumulator(reference.shape, StackMode.AVERAGE)
    accumulator.add(reference, source_validity)
    foreground_moments = None
    if foreground is not None or auto_foreground:
        foreground_moments = MomentAccumulator(reference.shape)
        foreground_moments.add(reference, source_validity)
    geometric_coverage = np.ones(reference.shape[:2], np.float32)
    accepted = [reference_path]
    rejected: dict[Path, str] = {}
    alignments: dict[Path, AlignmentStats] = {reference_path: AlignmentStats(stars=0, matches=0, inliers=0)}
    transforms: dict[Path, SpatialTransform] = {reference_path: SpatialTransform(np.eye(3, dtype=np.float64))}
    frame_confidence: dict[Path, float] = {reference_path: 1.0}

    # Nearest frames bootstrap robust statistics before drift/outliers reach the stack.
    ordered = sorted(
        ((index, path) for index, path in enumerate(lights) if index != reference_index),
        key=lambda item: abs(item[0] - reference_index),
    )
    for completed, (_, path) in enumerate(ordered, 1):
        _notify(request, "Aligning and stacking", completed, len(ordered))
        try:
            frame = _prepare_frame(path, request, reference, calibration, size, normalization_mask, None, reference_sample)
            transform, stats = estimate_transform(
                frame,
                reference,
                request.options.max_alignment_side,
                normalization_mask,
                request.options.correct_distortion,
                reference_field,
            )
            if stats.rms > 1.15:
                raise ValueError(f"star residual too high ({stats.rms:.2f} px)")
            # rotation is expected for an untracked camera; estimate_transform
            # already rejects transforms whose field residuals are unsafe
            if not 0.95 <= stats.scale <= 1.05:
                raise ValueError(f"implausible frame scale ({stats.scale:.3f})")
            aligned, validity = warp_frame(frame, transform, size, source_validity)
            confidence = registration_confidence(stats)
            validity *= confidence
            accumulator.add(aligned, validity)
            if foreground_moments is not None:
                foreground_moments.add(frame, source_validity)
            geometric_coverage += _geometric_coverage(transform, frame.shape[:2], size) * confidence
            accepted.append(path)
            alignments[path] = stats
            transforms[path] = transform
            frame_confidence[path] = confidence
            _log(
                request,
                f"Accepted {path.name}: {stats.inliers} verified stars over {stats.field_coverage:.0%} of field, "
                f"{stats.rotation_degrees:+.2f} deg, rms {stats.rms:.2f} px, p90 {stats.p90:.2f} px, "
                f"edge p90 {stats.edge_p90:.2f} px"
                + (f", fallback {stats.fallback}" if stats.fallback else ""),
            )
        except Exception as error:
            reason = str(error) or error.__class__.__name__
            rejected[path] = reason
            _log(request, f"Rejected {path.name}: {reason}")

    if not accepted:
        raise RuntimeError("no frames could be stacked")
    sky_clipped = None
    sky_mean = sky_deviation = None
    if robust_sky or auto_foreground:
        sky_mean, sky_deviation, sky_target = accumulator.statistics()
        if robust_sky:
            sky_clipped = ClippedAccumulator(sky_mean, sky_deviation, request.options.sigma, sky_target)
        else:
            stacked = sky_mean
    else:
        stacked = accumulator.finish()
    del accumulator

    foreground_clipped = None
    if foreground_moments is not None:
        foreground_mean, foreground_deviation, foreground_target = foreground_moments.statistics()
        if auto_foreground:
            temporal_mask = estimate_temporal_foreground_mask(sky_deviation, foreground_deviation)  # type: ignore[arg-type]
            probe_mask = _probe_foreground_mask(
                request,
                accepted,
                transforms,
                reference_path,
                reference,
                calibration,
                size,
            )
            # center sequence and endpoint evidence at the boundary
            foreground = _bottom_connected(temporal_mask * 0.55 + probe_mask * 0.45)
            if not 0.001 < float(foreground.mean()) < 0.45:
                raise ValueError("automatic foreground mask failed; supply a white-foreground mask")
            _log(request, f"Automatic foreground mask: {foreground.mean():.1%} of frame")
        # stricter clipping rejects moving stars while averaging camera-fixed color noise
        foreground_clipped = ClippedAccumulator(
            foreground_mean,
            foreground_deviation,
            min(request.options.sigma, 2.4),
            foreground_target,
        )
        del foreground_moments
    gc.collect()

    if sky_clipped is not None or foreground_clipped is not None:
        for completed, path in enumerate(accepted, 1):
            _notify(request, "Robust rejection pass", completed, len(accepted))
            frame = reference if path == reference_path else _prepare_frame(path, request, reference, calibration, size, normalization_mask, None, reference_sample)
            if sky_clipped is not None:
                aligned, validity = warp_frame(frame, transforms[path], size, source_validity)
                validity *= frame_confidence[path]
                sky_clipped.add(aligned, validity)
            if foreground_clipped is not None:
                # Stars move here, so the second pass rejects them from the camera-fixed layer.
                foreground_clipped.add(frame, source_validity)

    output_scale = 1.0
    if sky_clipped is not None:
        stacked = sky_clipped.finish(request.options.mode)
        output_scale = sky_clipped.output_scale
        _log(request, f"Robust sky downweighting: {sky_clipped.rejected_fraction:.2%} of covered samples")
        del sky_clipped
    if foreground is not None and foreground_clipped is not None:
        # average foreground chroma across raw camera coordinates, then restore ref luminance
        # this cuts shot/chroma noise without averaging the silhouette into motion blur
        foreground_stack = foreground_clipped.finish(StackMode.AVERAGE) * output_scale
        foreground_stack = _neutralize_foreground(foreground_stack, foreground)
        reference_stack = reference * output_scale
        static_stack = foreground_stack + (
            luminance(reference_stack) - luminance(foreground_stack)
        )[..., None]
        _log(request, f"Robust foreground downweighting: {foreground_clipped.rejected_fraction:.2%} of samples")
        del foreground_clipped
        stacked = stacked * (1.0 - foreground[..., None]) + static_stack * foreground[..., None]
        del foreground_stack, reference_stack, static_stack
    gc.collect()

    crop = None
    if request.options.auto_crop and len(accepted) > 1:
        crop = largest_coverage_crop(geometric_coverage, request.options.min_coverage)
        if crop:
            x0, y0, x1, y1 = crop
            stacked = stacked[y0:y1, x0:x1]
            geometric_coverage = geometric_coverage[y0:y1, x0:x1]
            if foreground is not None:
                foreground = foreground[y0:y1, x0:x1]
            _log(request, f"Coverage crop: {x1 - x0} x {y1 - y0} at {request.options.min_coverage:.0%} minimum")

    _notify(request, "Finishing image", 1, 1)
    image = process_image(stacked, request.options, foreground, calibration.dark_noise)
    _log(request, f"Finished: {len(accepted)} accepted, {len(rejected)} rejected")
    return StackResult(image, geometric_coverage, accepted, rejected, crop, alignments, foreground)
