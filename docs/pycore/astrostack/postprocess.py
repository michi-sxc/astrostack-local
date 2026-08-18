from __future__ import annotations

import cv2
import numpy as np

from .alignment import estimate_channel_alignment, luminance
from .models import ProcessingOptions, StackMode


def _fit_background(image: np.ndarray, exclude_mask: np.ndarray | None = None) -> np.ndarray:
    height, width = image.shape[:2]
    grid_y, grid_x = 16, 20
    samples_xy: list[tuple[float, float]] = []
    samples_rgb: list[np.ndarray] = []
    for iy in range(grid_y):
        y0, y1 = height * iy // grid_y, height * (iy + 1) // grid_y
        for ix in range(grid_x):
            x0, x1 = width * ix // grid_x, width * (ix + 1) // grid_x
            if exclude_mask is not None and float(exclude_mask[y0:y1, x0:x1].mean()) > 0.20:
                continue
            block = image[y0:y1, x0:x1].reshape(-1, 3)
            if len(block):
                samples_xy.append(((x0 + x1) / (2.0 * width), (y0 + y1) / (2.0 * height)))
                samples_rgb.append(np.percentile(block, 25.0, axis=0))
    if len(samples_xy) < 12:
        return np.zeros_like(image)

    xy = np.asarray(samples_xy, np.float64)
    values = np.asarray(samples_rgb, np.float64)
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    keep = np.ones(len(x), bool)
    coefficients = np.zeros((6, 3), np.float64)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = np.linalg.norm(values - design @ coefficients, axis=1)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median)) * 1.4826
        keep = residual <= median + max(2.8 * mad, 1e-5)
        if keep.sum() < 10:
            keep[:] = True
            break

    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx.astype(np.float32) / max(width - 1, 1)
    yy = yy.astype(np.float32) / max(height - 1, 1)
    basis = (np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy)
    background = sum(term[..., None] * coefficients[index].astype(np.float32) for index, term in enumerate(basis))
    sample_values = values[keep] if np.any(keep) else values
    lower, upper = np.percentile(sample_values, (2.0, 98.0), axis=0)
    # Dont let the polynomial invent a bright rim outside its sky samples.
    background = np.clip(background, lower, upper)
    return background.astype(np.float32)


def remove_light_pollution(image: np.ndarray, exclude_mask: np.ndarray | None = None) -> np.ndarray:
    background = _fit_background(image, exclude_mask)
    background_luminance = luminance(background)
    base_luminance = float(np.percentile(background_luminance, 25.0))
    # Full subtraction exposes curved walking noise when the capture has little rotation.
    corrected = np.maximum(image - (background_luminance - base_luminance)[..., None] * 0.62, 0.0)
    base_color = np.percentile(background.reshape(-1, 3), 25.0, axis=0)
    neutral = float(np.mean(base_color))
    # Partial balance avoids swapping a mild lamp cast for magenta shadows.
    gains = np.clip(np.power(neutral / np.maximum(base_color, 1e-6), 0.38), 0.86, 1.18)
    return (corrected * gains).astype(np.float32)


def _star_mask(image: np.ndarray) -> np.ndarray:
    gray = luminance(image)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
    detail = np.maximum(gray - blur, 0.0)
    median = float(np.median(detail))
    noise = float(np.median(np.abs(detail - median))) * 1.4826
    threshold = max(median + 2.7 * noise, float(np.percentile(detail, 97.8)))
    span = max(float(np.percentile(detail, 99.8)) - threshold, noise * 3.0, 1e-6)
    signal = np.clip((detail - threshold) / span, 0.0, 1.0)
    local_peak = detail >= cv2.dilate(detail, np.ones((3, 3), np.uint8)) * 0.92
    # keep faint point sources soft, but dont turn grain into a star mask
    signal *= 0.45 + 0.55 * local_peak.astype(np.float32)
    return np.clip(cv2.GaussianBlur(signal, (0, 0), 1.35), 0.0, 1.0)


def _weighted_gaussian_blur(image: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    numerator = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma)
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    # dont bleed star color into nearby sky
    return numerator / np.maximum(denominator[..., None], 1e-4)


def _masked_gaussian_blur(image: np.ndarray, protected: np.ndarray, sigma: float) -> np.ndarray:
    return _weighted_gaussian_blur(image, 1.0 - protected, sigma)


def _foreground_color_alpha(mask: np.ndarray) -> np.ndarray:
    """Wide alpha for color work; keep the actual edge untouched in luminance."""
    foreground = np.clip(mask.astype(np.float32), 0.0, 1.0)
    # broad chroma feather only; a hard luminance silhouette is still retained
    sigma = max(8.0, min(foreground.shape[:2]) * 0.012)
    return np.clip(cv2.GaussianBlur(foreground, (0, 0), sigma), 0.0, 1.0)


def _smooth_detail(detail: np.ndarray, noise: float, strength: float) -> np.ndarray:
    threshold = max(noise * strength, 1e-6)
    normalized = np.abs(detail) / threshold
    keep = 1.0 - np.exp(-0.65 * normalized * normalized)
    return detail * keep


def dynamic_denoise(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    gray = luminance(image)
    fine_base = cv2.GaussianBlur(gray, (0, 0), 1.0)
    coarse_base = cv2.GaussianBlur(fine_base, (0, 0), 2.4)
    fine_detail = gray - fine_base
    coarse_detail = fine_base - coarse_base
    background = star_mask < 0.08
    fine_sample = fine_detail[background] if np.any(background) else fine_detail.ravel()
    coarse_sample = coarse_detail[background] if np.any(background) else coarse_detail.ravel()
    fine_noise = float(np.median(np.abs(fine_sample - np.median(fine_sample)))) * 1.4826
    coarse_noise = float(np.median(np.abs(coarse_sample - np.median(coarse_sample)))) * 1.4826
    fine_clean = _smooth_detail(fine_detail, fine_noise, 2.4)
    coarse_clean = _smooth_detail(coarse_detail, coarse_noise, 1.5)
    clean_luminance = coarse_base + coarse_clean + fine_clean
    luminance_clean = image + (clean_luminance - gray)[..., None]

    # ratios near black create colored star halos
    chroma = image - gray[..., None]
    chroma_blur = _masked_gaussian_blur(chroma, star_mask, 2.2)
    denoised = luminance_clean + chroma_blur
    protect = np.clip(star_mask[..., None], 0.0, 1.0)
    return (denoised * (1.0 - protect) + image * protect).astype(np.float32)


def denoise_foreground_color(image: np.ndarray, foreground_mask: np.ndarray, dark_noise: float = 0.0) -> np.ndarray:
    """Denoise camera-layer color and replace the mask seam with a soft field."""
    foreground = np.clip(foreground_mask.astype(np.float32), 0.0, 1.0)
    if float(foreground.max()) < 0.1:
        return image
    gray = luminance(image)
    chroma = image - gray[..., None]
    alpha = _foreground_color_alpha(foreground)
    # wider fields fix the low-frequency color offset, not just grain
    fg_chroma = _weighted_gaussian_blur(chroma, foreground, 4.5)
    sky_chroma = _weighted_gaussian_blur(chroma, 1.0 - foreground, 4.5)
    local = chroma - _weighted_gaussian_blur(chroma, foreground, 1.3)
    selected = foreground > 0.75
    sample = local[selected]
    if len(sample):
        center = float(np.median(sample))
        noise = float(np.median(np.abs(sample - center))) * 1.4826
    else:
        noise = 0.0
    floor = max(float(dark_noise), 1e-5)
    # dark floor is a confidence cue: stronger calibrated noise allows stronger color cleanup
    strength = 0.90 + 0.08 * np.clip(noise / max(noise + floor, 1e-6), 0.0, 1.0)
    target = fg_chroma * alpha[..., None] + sky_chroma * (1.0 - alpha[..., None])
    # also replace the transition band on the sky side; this is what removes the seam
    edge_weight = 3.0 * alpha * (1.0 - alpha)
    mix = np.clip(alpha * strength + edge_weight, 0.0, 1.0)[..., None]
    return (gray[..., None] + chroma * (1.0 - mix) + target * mix).astype(np.float32)


def correct_chromatic_aberration(image: np.ndarray) -> np.ndarray:
    corrected = image.copy()
    height, width = image.shape[:2]
    for channel in (0, 2):
        transform = estimate_channel_alignment(image[..., channel], image[..., 1])
        if transform is not None:
            corrected[..., channel] = cv2.warpAffine(
                image[..., channel],
                transform,
                (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT,
            )
    return corrected


def enhance_stars(image: np.ndarray, star_mask: np.ndarray, strength: float = 0.18) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    positive_detail = np.maximum(image - blurred, 0.0)
    return (image + positive_detail * star_mask[..., None] * strength).astype(np.float32)


def suppress_foreground_speckles(image: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    # only repair isolated chroma hits deep in the camera layer
    # broad smoothing here turns a stable tree edge into motion blur
    local = cv2.medianBlur(image.astype(np.float32), 3)
    residual = np.max(np.abs(image - local), axis=2)
    chroma_range = np.ptp(image, axis=2)
    sample = residual[foreground_mask > 0.75]
    if not len(sample):
        return image
    median = float(np.median(sample))
    noise = float(np.median(np.abs(sample - median))) * 1.4826
    threshold = max(median + 7.0 * noise, 0.018)
    speckles = (foreground_mask > 0.75) & (residual > threshold) & (chroma_range > 0.055)
    if not np.any(speckles):
        return image
    cleaned = image.copy()
    cleaned[speckles] = local[speckles]
    return cleaned


def hdr_tone_map(image: np.ndarray, strength: float = 2.0) -> np.ndarray:
    gray = luminance(image)
    mapped = np.log1p(np.maximum(gray, 0.0) * strength) / np.log1p(strength)
    local_base = cv2.GaussianBlur(mapped, (0, 0), max(8.0, min(gray.shape) * 0.012))
    detail = mapped - local_base
    mapped = np.maximum(local_base + detail * 1.05, 0.0)
    return (image * (mapped / np.maximum(gray, 1e-6))[..., None]).astype(np.float32)


def auto_stretch(image: np.ndarray, exclude_mask: np.ndarray | None = None, strength: float = 3.5) -> np.ndarray:
    gray = luminance(image)
    sample = gray[exclude_mask < 0.5] if exclude_mask is not None and np.any(exclude_mask < 0.5) else gray.ravel()
    black, white = np.percentile(sample, (4.0, 99.95))
    stretched = np.clip((image - float(black)) / max(float(white - black), 1e-6), 0.0, None)
    return (np.arcsinh(stretched * strength) / np.arcsinh(strength)).astype(np.float32)


def process_image(
    image: np.ndarray,
    options: ProcessingOptions,
    foreground_mask: np.ndarray | None = None,
    dark_noise: float = 0.0,
) -> np.ndarray:
    result = np.maximum(image.astype(np.float32), 0.0)
    if options.auto_brightness:
        sample_mask = foreground_mask < 0.5 if foreground_mask is not None else np.ones(result.shape[:2], bool)
        sample = luminance(result)[sample_mask]
        # Normalize exposure before nonlinear tools so Sum and Mean finish identically.
        result /= max(float(np.percentile(sample, 99.8)), 1e-6)
    if options.correct_chromatic_aberration:
        result = correct_chromatic_aberration(result)
    if options.reduce_light_pollution:
        result = remove_light_pollution(result, foreground_mask)
    stars = _star_mask(result)
    if foreground_mask is not None:
        # Leaves and roof texture are not stars, let foreground denoise touch them.
        stars *= 1.0 - foreground_mask
    if options.dynamic_denoise:
        original = result
        result = dynamic_denoise(result, stars)
        if foreground_mask is not None:
            # sky-only denoise; keep camera-fixed texture sharp
            fg = np.clip(foreground_mask, 0.0, 1.0)[..., None]
            result = result * (1.0 - fg) + original * fg
    if options.enhance_stars:
        result = enhance_stars(result, stars)
    if options.hdr:
        result = hdr_tone_map(result)
    if options.auto_brightness:
        result = auto_stretch(result, foreground_mask, 3.0)
    elif options.mode is StackMode.SUM:
        result /= max(float(np.percentile(result, 99.9)), 1e-6)
    gray = luminance(result)
    if options.dynamic_denoise:
        protected_stars = _star_mask(result)
        if foreground_mask is not None:
            protected_stars *= 1.0 - foreground_mask
        smooth = _masked_gaussian_blur(result, protected_stars, 0.85)
        shadow = np.clip((0.38 - gray) / 0.28, 0.0, 1.0)
        shadow *= 1.0 - protected_stars
        if foreground_mask is not None:
            shadow *= 1.0 - foreground_mask
        result = result * (1.0 - shadow[..., None] * 0.12) + smooth * shadow[..., None] * 0.12
        gray = luminance(result)
    # Smooth only opponent color in non-stars; kills residual RGB bands without softening detail.
    final_stars = _star_mask(result)
    chroma_delta = result - gray[..., None]
    smooth_chroma = _masked_gaussian_blur(chroma_delta, final_stars, 3.0)
    chroma_mix = (1.0 - final_stars)[..., None] * 0.75
    if foreground_mask is not None:
        chroma_mix *= (1.0 - foreground_mask)[..., None]
    result = gray[..., None] + chroma_delta * (1.0 - chroma_mix) + smooth_chroma * chroma_mix
    gray = luminance(result)
    if foreground_mask is not None:
        result = denoise_foreground_color(result, foreground_mask, dark_noise)
        gray = luminance(result)
    chroma = result / np.maximum(gray[..., None], 1e-5)
    saturation = np.full(gray.shape, 0.52, np.float32)
    saturation *= 0.65 + 0.35 * np.clip(gray / 0.25, 0.0, 1.0)
    if foreground_mask is not None:
        # never use the hard mask for color, it recreates the seam we just removed
        saturation *= 1.0 - _foreground_color_alpha(foreground_mask) * 0.22
    # Dark night chroma gets wild after stretching, pull it toward neutral most in shadows.
    result = gray[..., None] * (1.0 + (chroma - 1.0) * saturation[..., None])
    if foreground_mask is not None:
        result = suppress_foreground_speckles(result, foreground_mask)
    return np.clip(result, 0.0, 1.0)
