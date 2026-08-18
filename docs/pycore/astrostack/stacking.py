from __future__ import annotations

import cv2
import numpy as np

from .models import StackMode


class MomentAccumulator:
    """First pass for exact sigma rejection without retaining the RAW frame cube."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.total = np.zeros(shape, np.float32)
        self.square_total = np.zeros(shape, np.float32)
        self.weight = np.zeros(shape[:2], np.float32)

    def add(self, frame: np.ndarray, validity: np.ndarray) -> None:
        valid = np.clip(validity.astype(np.float32), 0.0, 1.0)
        sample = frame.astype(np.float32, copy=False)
        self.total += sample * valid[..., None]
        self.square_total += sample * sample * valid[..., None]
        self.weight += valid

    def statistics(self) -> tuple[np.ndarray, np.ndarray, float]:
        safe_weight = np.maximum(self.weight, 1e-6)[..., None]
        mean = self.total / safe_weight
        variance = np.maximum(self.square_total / safe_weight - mean * mean, 0.0)
        # Small floor avoids rejecting harmless quantization differences in flat areas.
        deviation = np.sqrt(variance + 2.5e-7).astype(np.float32)
        valid_weight = self.weight[self.weight > 0]
        target = max(float(np.percentile(valid_weight, 90.0)), 1.0) if len(valid_weight) else 1.0
        return mean.astype(np.float32), deviation, target


class ClippedAccumulator:
    """Second pass; rejects defects jointly across RGB so they cannot leave color flecks."""

    def __init__(self, mean: np.ndarray, deviation: np.ndarray, sigma: float, target_weight: float) -> None:
        self.mean = mean
        self.deviation = deviation
        self.sigma = max(float(sigma), 2.0)
        self.target_weight = max(float(target_weight), 1.0)
        self.total = np.zeros_like(mean, np.float32)
        self.weight = np.zeros(mean.shape[:2], np.float32)
        self.seen_weight = 0.0
        self.kept_weight = 0.0
        self.output_scale = 1.0

    def add(self, frame: np.ndarray, validity: np.ndarray) -> None:
        valid = np.clip(validity.astype(np.float32), 0.0, 1.0)
        sample = frame.astype(np.float32, copy=False)
        z_score = np.max(np.abs(sample - self.mean) / self.deviation, axis=2)
        # Taper weak walking-noise outliers too; a hard cutoff leaves curved sensor trails.
        taper_start = max(1.25, self.sigma * 0.45)
        taper = np.clip((z_score - taper_start) / max(self.sigma - taper_start, 1e-3), 0.0, 1.0)
        robust_weight = np.square(1.0 - np.square(taper))
        robust_weight[z_score >= self.sigma] = 0.0
        accepted = valid * robust_weight
        self.total += sample * accepted[..., None]
        self.weight += accepted
        self.seen_weight += float(valid.sum())
        self.kept_weight += float(accepted.sum())

    @property
    def rejected_fraction(self) -> float:
        return 1.0 - self.kept_weight / max(self.seen_weight, 1.0)

    def finish(self, mode: StackMode) -> np.ndarray:
        result = self.total / np.maximum(self.weight, 1e-6)[..., None]
        missing = self.weight <= 1e-3
        result[missing] = self.mean[missing]
        if mode is StackMode.SUM:
            self.output_scale = self.target_weight
            result *= self.output_scale
        return result.astype(np.float32)

class CoverageAccumulator:
    """Streaming per-pixel stack; missing warp pixels never become black samples."""

    def __init__(self, shape: tuple[int, int, int], mode: StackMode, sigma: float = 3.0) -> None:
        self.mode = mode
        self.sigma = max(float(sigma), 1.0)
        self.total = np.zeros(shape, np.float32)
        self.square_total = np.zeros(shape, np.float32) if mode is StackMode.SIGMA_CLIPPED else None
        self.weight = np.zeros(shape[:2], np.float32)
        self.minimum = np.full(shape, np.inf, np.float16) if mode is StackMode.SIGMA_CLIPPED else None
        self.maximum = np.full(shape, -np.inf, np.float16) if mode is StackMode.SIGMA_CLIPPED else None
        self.full_count = np.zeros(shape[:2], np.uint16) if mode is StackMode.SIGMA_CLIPPED else None
        self.output_scale = 1.0

    def add(self, frame: np.ndarray, validity: np.ndarray) -> None:
        valid = np.clip(validity.astype(np.float32), 0.0, 1.0)
        sample = frame.astype(np.float32, copy=False)
        if self.square_total is not None:
            warm = self.weight >= 4.0
            if np.any(warm):
                safe_weight = np.maximum(self.weight, 1e-6)[..., None]
                mean = self.total / safe_weight
                variance = np.maximum(self.square_total / safe_weight - mean * mean, 0.0)
                deviation = np.sqrt(variance + 1e-8)
                lower = mean - self.sigma * deviation
                upper = mean + self.sigma * deviation
                # Winsorization stays streaming and rejects trails/hot pixels without a multi-GB frame cube.
                sample = np.where(warm[..., None], np.clip(sample, lower, upper), sample)
            self.square_total += sample * sample * valid[..., None]
            full = valid >= 0.999
            np.minimum(self.minimum, sample, out=self.minimum, where=full[..., None])
            np.maximum(self.maximum, sample, out=self.maximum, where=full[..., None])
            self.full_count += full
        self.total += sample * valid[..., None]
        self.weight += valid

    def finish(self) -> np.ndarray:
        mean = self.total / np.maximum(self.weight, 1e-6)[..., None]
        mean[self.weight <= 1e-3] = 0.0
        if self.minimum is not None and self.maximum is not None:
            # Trim both ends so early outliers cannot poison the streaming warmup.
            trim = (self.full_count >= 5) & (np.abs(self.weight - self.full_count) < 1e-3)
            if np.any(trim):
                denominator = np.maximum(self.weight[trim] - 2.0, 1.0)[:, None]
                mean[trim] = (
                    self.total[trim] - self.minimum[trim].astype(np.float32) - self.maximum[trim].astype(np.float32)
                ) / denominator
        if self.mode is StackMode.SUM:
            # Coverage compensation keeps integrated exposure constant at drifting edges.
            target = max(float(np.percentile(self.weight[self.weight > 0], 90.0)), 1.0)
            self.output_scale = target
            return mean * target
        return mean


def largest_coverage_crop(coverage: np.ndarray, fraction: float) -> tuple[int, int, int, int] | None:
    peak = float(coverage.max())
    if peak <= 0:
        return None
    binary = coverage >= peak * float(np.clip(fraction, 0.05, 1.0))
    step = max(1, int(np.ceil(max(binary.shape) / 700)))
    reduced = cv2.resize(
        binary.astype(np.uint8),
        (max(1, binary.shape[1] // step), max(1, binary.shape[0] // step)),
        interpolation=cv2.INTER_AREA,
    ) >= 0.999
    rect = _largest_rectangle(reduced)
    if rect is None:
        return None
    x, y, width, height = rect
    x0, y0 = x * step, y * step
    x1 = min(binary.shape[1], (x + width) * step)
    y1 = min(binary.shape[0], (y + height) * step)
    if step > 1 and x1 - x0 > step * 4 and y1 - y0 > step * 4:
        # Coarse resize can leak a few sub-threshold corner pixels, inset one cell.
        x0, y0, x1, y1 = x0 + step, y0 + step, x1 - step, y1 - step
    if (x1 - x0) * (y1 - y0) < binary.size * 0.10:
        return None
    return x0, y0, x1, y1


def _largest_rectangle(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    heights = np.zeros(binary.shape[1], np.int32)
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for row, values in enumerate(binary):
        heights = np.where(values, heights + 1, 0)
        stack: list[int] = []
        for column in range(len(heights) + 1):
            height = int(heights[column]) if column < len(heights) else 0
            while stack and height < heights[stack[-1]]:
                top = stack.pop()
                rect_height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = column - left
                area = rect_height * width
                if area > best_area:
                    best_area = area
                    best = (left, row - rect_height + 1, width, rect_height)
            stack.append(column)
    return best


def estimate_foreground_mask(
    reference: np.ndarray,
    comparison: np.ndarray | None = None,
    sky_aligned_comparison: np.ndarray | None = None,
) -> np.ndarray:
    if comparison is None or sky_aligned_comparison is None:
        return np.zeros(reference.shape[:2], np.float32)

    def gray_small(image: np.ndarray, scale: float) -> np.ndarray:
        gray = np.dot(image[..., :3], np.array((0.2126, 0.7152, 0.0722), np.float32))
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    scale = min(1.0, 1400.0 / max(reference.shape[:2]))
    ref = gray_small(reference, scale)
    static_candidate = gray_small(comparison, scale)
    sky_candidate = gray_small(sky_aligned_comparison, scale)
    sky_valid = sky_candidate > 1e-7
    sky_valid = cv2.erode(sky_valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ref_low, ref_high = np.percentile(ref, (10.0, 90.0))
    for candidate in (static_candidate, sky_candidate):
        low, high = np.percentile(candidate, (10.0, 90.0))
        candidate -= low
        candidate *= (ref_high - ref_low) / max(float(high - low), 1e-6)
        candidate += ref_low

    # Wide support ignores stationary hot pixels and keeps only coherent objects.
    static_error = cv2.GaussianBlur(np.abs(ref - static_candidate), (0, 0), 4.5)
    sky_error = cv2.GaussianBlur(np.abs(ref - sky_candidate), (0, 0), 4.5)
    advantage = cv2.GaussianBlur(sky_error - static_error, (0, 0), 3.0)
    threshold = max(float(np.percentile(advantage, 86.0)), 0.004)
    # Foreground matches before sky alignment and mismatches after it.
    candidate = (
        (advantage > threshold)
        & (static_error < np.percentile(static_error, 60.0))
        & sky_valid
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate)
    # Typical foreground enters from the horizon/frame edges; ignore top-edge warp artifacts.
    border_labels = np.unique(np.concatenate((labels[-4:].ravel(), labels[:, -4:].ravel(), labels[:, :4].ravel())))
    for label in border_labels:
        if 0 < label < count and stats[label, cv2.CC_STAT_AREA] > 100:
            keep[labels == label] = 1
    # Keep canopy/branch holes open; contour filling wrongly marks visible sky as foreground.
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.resize(keep.astype(np.float32), reference.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    feather = max(1.4, min(reference.shape[:2]) * 0.0008)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), feather), 0.0, 1.0)


def estimate_temporal_foreground_mask(sky_deviation: np.ndarray, camera_deviation: np.ndarray) -> np.ndarray:
    """Find coherent camera-fixed regions from the full registered sequence."""
    scale = min(1.0, 900.0 / max(sky_deviation.shape[:2]))
    weights = np.array((0.2126, 0.7152, 0.0722), np.float32)
    sky = cv2.resize(sky_deviation @ weights, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    camera = cv2.resize(camera_deviation @ weights, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    score = cv2.GaussianBlur(sky - camera, (0, 0), 4.0)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) * 1.4826
    # favor recall for connected foreground; bottom connectivity rejects sky speckles later
    threshold = max(float(np.percentile(score, 75.0)), median + 2.0 * mad, 0.0005)
    candidate = ((score > threshold) & (camera < sky * 0.98)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate)
    border_labels = np.unique(np.concatenate((labels[-5:].ravel(), labels[:, :5].ravel(), labels[:, -5:].ravel())))
    for label in border_labels:
        if 0 < label < count and stats[label, cv2.CC_STAT_AREA] > 120:
            keep[labels == label] = 1
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.resize(keep.astype(np.float32), sky_deviation.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    feather = max(1.4, min(sky_deviation.shape[:2]) * 0.0008)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), feather), 0.0, 1.0)
