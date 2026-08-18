from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .models import AlignmentStats


@dataclass(slots=True)
class StarField:
    points: np.ndarray
    shape: tuple[int, int]
    scale: float
    detail: np.ndarray


@dataclass(slots=True)
class AlignmentReference:
    """Reference detections reused for every light in a sequence."""

    coarse: StarField
    fine: StarField
    max_side: int


@dataclass(slots=True)
class SpatialTransform:
    matrix: np.ndarray
    # smooth source-space residual, fixes lens distortion left after homography
    correction: np.ndarray | None = None


def luminance(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3], np.array((0.2126, 0.7152, 0.0722), np.float32))


def _alignment_image(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    gray = luminance(image)
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(gray, (2.0, 99.8))
    gray = np.clip((gray - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    background = cv2.GaussianBlur(gray, (0, 0), 7.0)
    detail = np.maximum(gray - background, 0.0)
    hi_detail = float(np.percentile(detail, 99.95))
    if hi_detail > 0:
        detail = np.clip(detail / hi_detail, 0.0, 1.0)
    return detail.astype(np.float32), scale


def detect_stars(
    image: np.ndarray,
    max_side: int = 4096,
    exclude_mask: np.ndarray | None = None,
    max_stars: int = 1200,
) -> StarField:
    detail, scale = _alignment_image(image, max_side)
    median = float(np.median(detail))
    mad = float(np.median(np.abs(detail - median)))
    threshold = max(float(np.percentile(detail, 97.5)), median + 5.0 * 1.4826 * mad, 0.035)
    local_max = detail >= cv2.dilate(detail, np.ones((5, 5), np.uint8))
    candidates = local_max & (detail >= threshold)

    if exclude_mask is not None:
        mask = cv2.resize(exclude_mask, detail.shape[::-1], interpolation=cv2.INTER_AREA)
        candidates &= mask < 0.25

    ys, xs = np.nonzero(candidates)
    if not len(xs):
        return StarField(np.empty((0, 2), np.float32), detail.shape, scale, detail)
    strengths = detail[ys, xs]
    order = np.argsort(strengths)[::-1][:max_stars]
    points = np.column_stack((xs[order], ys[order])).astype(np.float32)
    border = 8
    keep = (
        (points[:, 0] >= border)
        & (points[:, 1] >= border)
        & (points[:, 0] < detail.shape[1] - border)
        & (points[:, 1] < detail.shape[0] - border)
    )
    points = points[keep]
    refined = np.empty_like(points)
    for index, (x_value, y_value) in enumerate(points):
        x, y = int(x_value), int(y_value)
        patch = detail[y - 2 : y + 3, x - 2 : x + 3]
        weights = np.maximum(patch - float(np.percentile(patch, 20.0)), 0.0)
        total = float(weights.sum())
        if total <= 1e-8:
            refined[index] = (x_value, y_value)
            continue
        yy, xx = np.mgrid[y - 2 : y + 3, x - 2 : x + 3]
        refined[index] = (float((xx * weights).sum() / total), float((yy * weights).sum() / total))
    return StarField(refined, detail.shape, scale, detail)


def prepare_reference(
    image: np.ndarray,
    max_side: int = 4096,
    exclude_mask: np.ndarray | None = None,
) -> AlignmentReference:
    coarse_side = min(max_side, 1200)
    coarse = detect_stars(image, coarse_side, exclude_mask, max_stars=700)
    fine = coarse if max_side <= coarse_side else detect_stars(image, max_side, exclude_mask, max_stars=2000)
    if fine is not coarse:
        # fine matching only needs points; keep coarse detail for rare phase fallback
        fine = StarField(fine.points, fine.shape, fine.scale, np.empty(0, np.float32))
    return AlignmentReference(coarse, fine, max_side)


def registration_confidence(stats: AlignmentStats) -> float:
    """Soft weight for usable but less-covered edge frames."""
    residual = np.clip((0.82 - stats.p90) / 0.42, 0.0, 1.0)
    edge = np.clip((0.92 - stats.edge_p90) / 0.42, 0.0, 1.0)
    field = np.clip((stats.field_coverage - 0.35) / 0.35, 0.0, 1.0)
    return float(max(0.35, min(residual, edge, field)))


def _mutual_matches(
    current: np.ndarray,
    reference: np.ndarray,
    matrix: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = cv2.perspectiveTransform(current.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    distance, nearest = cKDTree(reference).query(predicted, k=1)
    reverse = cKDTree(predicted).query(reference, k=1)[1]
    indices = np.arange(len(predicted))
    good = (distance < max_distance) & (reverse[nearest] == indices)
    return current[good], reference[nearest[good]], distance[good]


def _triangle_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 6:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.int32)
    try:
        triangles = Delaunay(points).simplices
    except Exception:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.int32)

    vertices = points[triangles]
    opposite = np.stack(
        (
            np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1),
        ),
        axis=1,
    )
    longest = opposite.max(axis=1)
    ordered_sides = np.sort(opposite, axis=1)
    limit = min(350.0, float(np.max(np.ptp(points, axis=0))) * 0.35)
    keep = (longest >= 8.0) & (longest <= limit) & (ordered_sides[:, 0] / np.maximum(longest, 1e-6) >= 0.12)
    descriptors = (ordered_sides[keep, :2] / longest[keep, None]).astype(np.float32)
    # Opposite-side ordering maps the same physical vertex across rotation/scale.
    canonical = np.take_along_axis(triangles, np.argsort(opposite, axis=1), axis=1)[keep]
    return descriptors, canonical.astype(np.int32, copy=False)


def _triangle_correspondences(current: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    cur_desc, cur_triangles = _triangle_features(current)
    ref_desc, ref_triangles = _triangle_features(reference)
    if not len(cur_desc) or len(ref_desc) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), 0

    distances, indices = cKDTree(ref_desc).query(cur_desc, k=2)
    accepted = (distances[:, 0] < 0.018) & (distances[:, 0] < distances[:, 1] * 0.82)
    matched_cur = cur_triangles[accepted]
    matched_ref = ref_triangles[indices[accepted, 0]]
    if not len(matched_cur):
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), 0
    return current[matched_cur.reshape(-1)], reference[matched_ref.reshape(-1)], len(matched_cur)


def _phase_translation(current: StarField, reference: StarField) -> np.ndarray:
    shift, response = cv2.phaseCorrelate(current.detail, reference.detail)
    if not np.isfinite(shift).all() or response < 0.02:
        raise ValueError("not enough consistent stars")
    # phaseCorrelate(current, reference) yields current-to-reference shift.
    return np.array(((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1]), (0.0, 0.0, 1.0)), np.float64)


def _validate_transform(matrix: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    affine = matrix[:2, :2]
    scale = math.sqrt(abs(float(np.linalg.det(affine))))
    rotation = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    height, width = shape
    corners = np.float32((((0, 0), (width, 0), (width, height), (0, height)),)).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, matrix.astype(np.float64)).reshape(-1, 2)
    span = np.ptp(warped, axis=0)
    if not np.isfinite(matrix).all() or not 0.82 <= scale <= 1.18 or abs(rotation) > 20.0:
        raise ValueError("implausible star transform")
    if np.any(span < np.array((width, height)) * 0.55) or np.any(span > np.array((width, height)) * 1.55):
        raise ValueError("transform distorts the frame too much")
    return scale, rotation


def _refine_homography(
    current: StarField,
    reference: StarField,
    matrix: np.ndarray,
) -> tuple[np.ndarray, int]:
    refine_src, refine_dst, _ = _mutual_matches(current.points, reference.points, matrix, 5.0)
    if len(refine_src) < 10:
        return matrix, 0
    homography, inliers = cv2.findHomography(
        refine_src,
        refine_dst,
        cv2.RANSAC,
        2.5,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or inliers is None or inliers.sum() < 8:
        return matrix, 0
    try:
        _validate_transform(homography, current.shape)
    except ValueError:
        return matrix, 0
    return homography, int(inliers.sum())


def _residual_basis(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = points[:, 0] / max(width * 0.5, 1.0) - 1.0
    y = points[:, 1] / max(height * 0.5, 1.0) - 1.0
    # lens residuals are mostly radial; fewer terms avoid cubic edge wiggles
    radius2 = x * x + y * y
    return np.column_stack((np.ones(len(x)), x, y, x * radius2, y * radius2, x * radius2**2, y * radius2**2))


def _robust_polynomial_fit(basis: np.ndarray, residual: np.ndarray) -> np.ndarray:
    weights = np.ones(len(basis), np.float64)
    coefficients = np.zeros((basis.shape[1], 2), np.float64)
    for _ in range(5):
        weighted = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(basis * weighted, residual * weighted, rcond=1e-5)[0]
        error = np.linalg.norm(residual - basis @ coefficients, axis=1)
        scale = max(float(np.median(error)) * 2.5, 0.05)
        weights = 1.0 / (1.0 + np.square(error / scale))
    return coefficients


def _fit_spatial_correction(
    predicted: np.ndarray,
    destination: np.ndarray,
    shape: tuple[int, int],
    max_correction: float,
) -> tuple[np.ndarray | None, float]:
    if len(predicted) < 28:
        return None, 0.0
    basis = _residual_basis(predicted, shape)
    residual = destination - predicted
    holdout = np.arange(len(basis)) % 5 == 0
    if holdout.sum() < 7 or (~holdout).sum() < basis.shape[1] * 2:
        return None, 0.0
    trial = _robust_polynomial_fit(basis[~holdout], residual[~holdout])
    before = np.linalg.norm(residual[holdout], axis=1)
    after = np.linalg.norm(residual[holdout] - basis[holdout] @ trial, axis=1)
    before_p90 = float(np.percentile(before, 90.0))
    after_p90 = float(np.percentile(after, 90.0))
    if after_p90 >= before_p90 * 0.92 or before_p90 - after_p90 < 0.08:
        return None, 0.0

    height, width = shape
    normalized = np.column_stack(
        (
            np.abs(predicted[holdout, 0] / max(width * 0.5, 1.0) - 1.0),
            np.abs(predicted[holdout, 1] / max(height * 0.5, 1.0) - 1.0),
        )
    )
    edge = np.max(normalized, axis=1) > 0.68
    if edge.sum() >= 6 and np.percentile(after[edge], 90.0) >= np.percentile(before[edge], 90.0):
        return None, 0.0

    coefficients = _robust_polynomial_fit(basis, residual)
    xx, yy = np.meshgrid(np.linspace(0, width, 9), np.linspace(0, height, 7))
    grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    grid_correction = float(np.max(np.linalg.norm(_residual_basis(grid, shape) @ coefficients, axis=1)))
    if not np.isfinite(coefficients).all() or grid_correction > max_correction:
        return None, 0.0
    return coefficients, 1.0 - after_p90 / max(before_p90, 1e-6)


def estimate_transform(
    current_image: np.ndarray,
    reference_image: np.ndarray,
    max_side: int = 4096,
    exclude_mask: np.ndarray | None = None,
    correct_distortion: bool = True,
    reference_field: AlignmentReference | None = None,
) -> tuple[SpatialTransform, AlignmentStats]:
    # coarse pass ignores RAW defects, native pass adds precision
    coarse_side = min(max_side, 1200)
    current = detect_stars(current_image, coarse_side, exclude_mask, max_stars=700)
    cached = reference_field if reference_field is not None and reference_field.max_side == max_side else None
    reference = cached.coarse if cached is not None else detect_stars(reference_image, coarse_side, exclude_mask, max_stars=700)
    stats = AlignmentStats(stars=len(current.points))
    src, dst, triangle_matches = _triangle_correspondences(current.points, reference.points)

    matrix_small: np.ndarray
    inlier_mask: np.ndarray | None = None
    if len(src) >= 9:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=15_000,
            confidence=0.999,
            refineIters=25,
        )
        if affine is None:
            matrix_small = _phase_translation(current, reference)
            stats.fallback = "phase correlation"
        else:
            matrix_small = np.vstack((affine, (0.0, 0.0, 1.0))).astype(np.float64)
    else:
        matrix_small = _phase_translation(current, reference)
        stats.fallback = "phase correlation"

    stats.matches = triangle_matches * 3
    stats.inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if correct_distortion and len(current.points) >= 12 and stats.fallback is None:
        matrix_small, refined_inliers = _refine_homography(current, reference, matrix_small)
        stats.inliers = refined_inliers or stats.inliers

    if max_side > coarse_side:
        fine_current = detect_stars(current_image, max_side, exclude_mask, max_stars=2000)
        fine_reference = cached.fine if cached is not None else detect_stars(reference_image, max_side, exclude_mask, max_stars=2000)
        ratio = fine_current.scale / current.scale
        scale_up = np.diag((ratio, ratio, 1.0))
        matrix_small = scale_up @ matrix_small @ np.linalg.inv(scale_up)
        current, reference = fine_current, fine_reference
        stats.stars = len(current.points)

    # Refine only mutual star pairs; one-way nearest matches bend dense edge fields.
    if correct_distortion and len(current.points) >= 12 and stats.fallback is None:
        matrix_small, refined_inliers = _refine_homography(current, reference, matrix_small)
        stats.inliers = refined_inliers or stats.inliers

    scale, rotation = _validate_transform(matrix_small, current.shape)
    stats.scale = scale
    stats.rotation_degrees = rotation

    quality_src, quality_dst, residual = _mutual_matches(current.points, reference.points, matrix_small, 3.5)
    if len(residual) < 12:
        raise ValueError("not enough verified star matches")
    predicted = cv2.perspectiveTransform(quality_src.reshape(-1, 1, 2), matrix_small).reshape(-1, 2)
    correction_small = None
    if correct_distortion:
        # cap in native px, avoids edge warp
        correction_small, stats.distortion_improvement = _fit_spatial_correction(
            predicted,
            quality_dst,
            current.shape,
            2.5 * current.scale,
        )
        if correction_small is not None:
            residual = np.linalg.norm(quality_dst - predicted - _residual_basis(predicted, current.shape) @ correction_small, axis=1)
    stats.inliers = len(residual)
    # keep quality gates independent of solve scale
    stats.rms = float(np.sqrt(np.mean(np.square(residual))) / current.scale)
    stats.p90 = float(np.percentile(residual, 90.0) / current.scale)
    height, width = current.shape
    normalized = np.column_stack(
        (
            np.abs(quality_dst[:, 0] / max(width * 0.5, 1.0) - 1.0),
            np.abs(quality_dst[:, 1] / max(height * 0.5, 1.0) - 1.0),
        )
    )
    edge = np.max(normalized, axis=1) > 0.68
    stats.edge_p90 = float(np.percentile(residual[edge], 90.0) / current.scale) if edge.sum() >= 8 else stats.p90
    hull = cv2.convexHull(quality_dst.astype(np.float32))
    stats.field_coverage = float(cv2.contourArea(hull) / max(current.shape[0] * current.shape[1], 1))
    if stats.field_coverage < 0.35:
        raise ValueError(f"star matches cover too little of the frame ({stats.field_coverage:.0%})")
    # keep only frames that stay tight across the whole field
    if stats.p90 > 0.82:
        raise ValueError(f"outer-field star residual too high ({stats.p90:.2f} px)")
    if stats.edge_p90 > 0.92:
        raise ValueError(f"edge star residual too high ({stats.edge_p90:.2f} px)")

    scale_matrix = np.diag((current.scale, current.scale, 1.0))
    matrix_full = np.linalg.inv(scale_matrix) @ matrix_small @ scale_matrix
    correction_full = None if correction_small is None else correction_small / current.scale
    return SpatialTransform(matrix_full, correction_full), stats


def estimate_channel_alignment(
    current_channel: np.ndarray,
    reference_channel: np.ndarray,
    max_side: int = 1400,
) -> np.ndarray | None:
    current_rgb = np.repeat(current_channel[..., None], 3, axis=2)
    reference_rgb = np.repeat(reference_channel[..., None], 3, axis=2)
    current = detect_stars(current_rgb, max_side=max_side, max_stars=1000)
    reference = detect_stars(reference_rgb, max_side=max_side, max_stars=1000)
    if len(current.points) < 30 or len(reference.points) < 30:
        return None
    src, dst, before = _mutual_matches(current.points, reference.points, np.eye(3, dtype=np.float64), 5.0)
    if len(src) < 25:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=10_000,
        confidence=0.999,
        refineIters=30,
    )
    if affine is None or inliers is None or inliers.sum() < 20:
        return None
    matrix_small = np.vstack((affine, (0.0, 0.0, 1.0))).astype(np.float64)
    scale = math.sqrt(abs(float(np.linalg.det(matrix_small[:2, :2]))))
    rotation = abs(math.degrees(math.atan2(float(matrix_small[1, 0]), float(matrix_small[0, 0]))))
    if not 0.994 <= scale <= 1.006 or rotation > 0.2 or np.linalg.norm(matrix_small[:2, 2]) > 5.0:
        return None
    predicted = cv2.transform(src.reshape(-1, 1, 2), affine).reshape(-1, 2)
    selected = inliers.ravel().astype(bool)
    after = np.linalg.norm(dst[selected] - predicted[selected], axis=1)
    before_p90 = float(np.percentile(before[selected], 90.0))
    after_p90 = float(np.percentile(after, 90.0))
    if after_p90 >= before_p90 * 0.9 or before_p90 - after_p90 < 0.025:
        return None
    scale_matrix = np.diag((current.scale, current.scale, 1.0))
    return (np.linalg.inv(scale_matrix) @ matrix_small @ scale_matrix)[:2]


def _evaluate_correction(x: np.ndarray, y: np.ndarray, correction: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    xn = x / max(width * 0.5, 1.0) - 1.0
    yn = y / max(height * 0.5, 1.0) - 1.0
    radius2 = xn * xn + yn * yn
    basis = np.stack((np.ones_like(xn), xn, yn, xn * radius2, yn * radius2, xn * radius2**2, yn * radius2**2), axis=-1)
    delta = basis @ correction.astype(np.float32)
    return delta[..., 0], delta[..., 1]


def _warp_spatial(
    source: np.ndarray,
    transform: SpatialTransform,
    size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    width, height = size
    output_shape = (height, width) + source.shape[2:]
    output = np.zeros(output_shape, source.dtype)
    inverse = np.linalg.inv(transform.matrix).astype(np.float64)
    x = np.arange(width, dtype=np.float32)[None, :]
    # Tile the maps; full-frame x/y maps waste hundreds of MB on RAWs.
    for y0 in range(0, height, 384):
        y1 = min(y0 + 384, height)
        target_x = np.broadcast_to(x, (y1 - y0, width))
        target_y = np.broadcast_to(np.arange(y0, y1, dtype=np.float32)[:, None], target_x.shape)
        # correction is fitted as source->target residual; solve its inverse
        denominator = inverse[2, 0] * target_x + inverse[2, 1] * target_y + inverse[2, 2]
        base_x = (inverse[0, 0] * target_x + inverse[0, 1] * target_y + inverse[0, 2]) / denominator
        base_y = (inverse[1, 0] * target_x + inverse[1, 1] * target_y + inverse[1, 2]) / denominator
        for _ in range(2):
            dx, dy = _evaluate_correction(base_x, base_y, transform.correction, size)  # type: ignore[arg-type]
            corrected_x, corrected_y = target_x - dx, target_y - dy
            denominator = inverse[2, 0] * corrected_x + inverse[2, 1] * corrected_y + inverse[2, 2]
            base_x = (inverse[0, 0] * corrected_x + inverse[0, 1] * corrected_y + inverse[0, 2]) / denominator
            base_y = (inverse[1, 0] * corrected_x + inverse[1, 1] * corrected_y + inverse[1, 2]) / denominator
        map_x = base_x.astype(np.float32)
        map_y = base_y.astype(np.float32)
        output[y0:y1] = cv2.remap(source, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return output


def warp_mask(mask: np.ndarray, transform: SpatialTransform | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if isinstance(transform, np.ndarray):
        transform = SpatialTransform(transform)
    if transform.correction is None:
        return cv2.warpPerspective(mask, transform.matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _warp_spatial(mask.astype(np.float32, copy=False), transform, size, cv2.INTER_LINEAR)


def warp_frame(
    frame: np.ndarray,
    transform: SpatialTransform | np.ndarray,
    size: tuple[int, int],
    source_validity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(transform, np.ndarray):
        transform = SpatialTransform(transform)
    width, height = size
    if transform.correction is None:
        aligned = cv2.warpPerspective(
            frame,
            transform.matrix,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        aligned = _warp_spatial(frame, transform, size, cv2.INTER_LANCZOS4)
    validity = np.ones(frame.shape[:2], np.float32) if source_validity is None else source_validity.astype(np.float32)
    coverage = warp_mask(validity, transform, size)
    coverage[coverage < 1e-3] = 0.0
    return aligned, np.clip(coverage, 0.0, 1.0)
