from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from astrostack.alignment import estimate_transform, luminance, warp_frame
from astrostack.io import write_image, write_stack_outputs
from astrostack.models import ProcessingOptions, StackMode
from astrostack.postprocess import _star_mask, correct_chromatic_aberration, denoise_foreground_color, dynamic_denoise, process_image, suppress_foreground_speckles
from astrostack.stacking import ClippedAccumulator, CoverageAccumulator, MomentAccumulator, largest_coverage_crop


class CoverageAccumulatorTests(unittest.TestCase):
    def test_two_pass_sum_rejects_repeatable_sensor_defect(self) -> None:
        shape = (8, 8, 3)
        valid = np.ones(shape[:2], np.float32)
        frames = [np.full(shape, 0.2, np.float32) for _ in range(40)]
        frames[7][3, 4] = (0.8, 0.1, 0.7)
        moments = MomentAccumulator(shape)
        for frame in frames:
            moments.add(frame, valid)
        mean, deviation, target = moments.statistics()
        clipped = ClippedAccumulator(mean, deviation, 3.0, target)
        for frame in frames:
            clipped.add(frame, valid)
        result = clipped.finish(StackMode.SUM)
        self.assertTrue(np.allclose(result[3, 4], 8.0, atol=1e-4))
        self.assertAlmostEqual(clipped.output_scale, 40.0)

    def test_two_pass_downweights_walking_noise_below_hard_cutoff(self) -> None:
        shape = (3, 3, 3)
        valid = np.ones(shape[:2], np.float32)
        frames = [np.full(shape, 0.2, np.float32) for _ in range(34)]
        frames += [np.full(shape, 0.3, np.float32) for _ in range(6)]
        moments = MomentAccumulator(shape)
        for frame in frames:
            moments.add(frame, valid)
        mean, deviation, target = moments.statistics()
        clipped = ClippedAccumulator(mean, deviation, 3.0, target)
        for frame in frames:
            clipped.add(frame, valid)
        result = clipped.finish(StackMode.SIGMA_CLIPPED)
        self.assertLess(float(result.mean()), 0.21)

    def test_missing_warp_pixels_do_not_dark_edge(self) -> None:
        shape = (40, 60, 3)
        accumulator = CoverageAccumulator(shape, StackMode.AVERAGE)
        reference = np.full(shape, 0.5, np.float32)
        shifted = np.full(shape, 0.5, np.float32)
        accumulator.add(reference, np.ones(shape[:2], np.float32))
        transform = np.array(((1, 0, 12), (0, 1, 0), (0, 0, 1)), np.float64)
        aligned, validity = warp_frame(shifted, transform, (shape[1], shape[0]))
        accumulator.add(aligned, validity)
        result = accumulator.finish()
        self.assertTrue(np.allclose(result, 0.5, atol=1e-5))
        self.assertAlmostEqual(float(accumulator.weight[:, :10].mean()), 1.0, places=3)
        self.assertAlmostEqual(float(accumulator.weight[:, 20:].mean()), 2.0, places=3)

    def test_sum_compensates_coverage(self) -> None:
        shape = (24, 32, 3)
        accumulator = CoverageAccumulator(shape, StackMode.SUM)
        accumulator.add(np.full(shape, 0.2, np.float32), np.ones(shape[:2], np.float32))
        valid = np.ones(shape[:2], np.float32)
        valid[:, :8] = 0
        accumulator.add(np.full(shape, 0.2, np.float32), valid)
        result = accumulator.finish()
        self.assertTrue(np.allclose(result, 0.4, atol=1e-5))

    def test_sigma_mode_suppresses_late_hot_pixel(self) -> None:
        shape = (8, 8, 3)
        accumulator = CoverageAccumulator(shape, StackMode.SIGMA_CLIPPED, sigma=3.0)
        valid = np.ones(shape[:2], np.float32)
        for _ in range(5):
            accumulator.add(np.full(shape, 0.2, np.float32), valid)
        outlier = np.full(shape, 0.2, np.float32)
        outlier[3, 4] = 1.0
        accumulator.add(outlier, valid)
        self.assertLess(float(accumulator.finish()[3, 4, 0]), 0.21)

    def test_sigma_mode_suppresses_early_hot_pixel(self) -> None:
        shape = (8, 8, 3)
        accumulator = CoverageAccumulator(shape, StackMode.SIGMA_CLIPPED, sigma=3.0)
        valid = np.ones(shape[:2], np.float32)
        outlier = np.full(shape, 0.2, np.float32)
        outlier[3, 4] = 1.0
        accumulator.add(outlier, valid)
        for _ in range(5):
            accumulator.add(np.full(shape, 0.2, np.float32), valid)
        self.assertLess(float(accumulator.finish()[3, 4, 0]), 0.21)

    def test_sigma_mode_handles_feathered_validity(self) -> None:
        shape = (8, 8, 3)
        accumulator = CoverageAccumulator(shape, StackMode.SIGMA_CLIPPED, sigma=3.0)
        validity = np.full(shape[:2], 0.5, np.float32)
        for _ in range(6):
            accumulator.add(np.full(shape, 0.3, np.float32), validity)
        result = accumulator.finish()
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.allclose(result, 0.3, atol=1e-4))

    def test_coverage_crop_finds_shared_center(self) -> None:
        coverage = np.ones((100, 120), np.float32)
        coverage[:, 15:105] = 10
        crop = largest_coverage_crop(coverage, 0.8)
        self.assertIsNotNone(crop)
        x0, y0, x1, y1 = crop or (0, 0, 0, 0)
        self.assertGreaterEqual(x0, 14)
        self.assertLessEqual(x1, 106)
        self.assertEqual((y0, y1), (0, 100))


class AlignmentTests(unittest.TestCase):
    def test_star_triangle_alignment_recovers_rotation_and_shift(self) -> None:
        rng = np.random.default_rng(7)
        reference = np.zeros((320, 420, 3), np.float32)
        for x, y, strength in zip(rng.integers(25, 395, 90), rng.integers(25, 295, 90), rng.uniform(0.35, 1.0, 90)):
            cv2.circle(reference, (int(x), int(y)), int(rng.integers(1, 3)), (strength,) * 3, -1, cv2.LINE_AA)
        reference = cv2.GaussianBlur(reference, (0, 0), 0.7)
        center = (reference.shape[1] / 2, reference.shape[0] / 2)
        affine = cv2.getRotationMatrix2D(center, 2.6, 1.0)
        affine[:, 2] += (9.0, -6.0)
        current = cv2.warpAffine(reference, affine, reference.shape[1::-1])
        expected = np.vstack((cv2.invertAffineTransform(affine), (0.0, 0.0, 1.0)))
        # cover coarse-to-native refinement used by full RAW jobs
        estimated, stats = estimate_transform(current, reference, max_side=1500, correct_distortion=True)
        points = np.float32((((80, 70), (330, 250), (210, 160)),)).reshape(-1, 1, 2)
        expected_points = cv2.perspectiveTransform(points, expected)
        actual_points = cv2.perspectiveTransform(points, estimated.matrix)
        self.assertLess(float(np.max(np.linalg.norm(expected_points - actual_points, axis=2))), 3.0)
        self.assertGreaterEqual(stats.inliers, 8)

    def test_alignment_stats_are_native_pixel_units(self) -> None:
        rng = np.random.default_rng(17)
        reference = np.zeros((640, 840, 3), np.float32)
        for x, y in zip(rng.integers(20, 820, 180), rng.integers(20, 620, 180)):
            cv2.circle(reference, (int(x), int(y)), 2, (0.8,) * 3, -1, cv2.LINE_AA)
        reference = cv2.GaussianBlur(reference, (0, 0), 0.8)
        current = cv2.warpAffine(reference, np.float32(((1, 0, 5.3), (0, 1, -3.7))), reference.shape[1::-1])
        _, full = estimate_transform(current, reference, max_side=900)
        _, reduced = estimate_transform(current, reference, max_side=420)
        self.assertLess(abs(full.p90 - reduced.p90), 0.45)


class PostProcessingTests(unittest.TestCase):
    def test_denoise_does_not_brighten_star_halo(self) -> None:
        yy, xx = np.mgrid[0:81, 0:81]
        radius = np.hypot(xx - 40, yy - 40)
        star = 0.05 + 0.8 * np.exp(-(radius**2) / (2.0 * 1.5**2))
        image = np.repeat(star[..., None], 3, axis=2).astype(np.float32)
        result = dynamic_denoise(image, _star_mask(image))
        annulus = (radius >= 4.0) & (radius <= 9.0)
        self.assertLess(float(np.max(result[annulus] - image[annulus])), 0.002)

    def test_chromatic_aberration_alignment_reduces_channel_error(self) -> None:
        rng = np.random.default_rng(9)
        green = np.zeros((360, 480), np.float32)
        for x, y, strength in zip(rng.integers(20, 460, 150), rng.integers(20, 340, 150), rng.uniform(0.4, 1.0, 150)):
            cv2.circle(green, (int(x), int(y)), 1, float(strength), -1, cv2.LINE_AA)
        green = cv2.GaussianBlur(green, (0, 0), 0.7)
        distortion = cv2.getRotationMatrix2D((240, 180), 0.03, 1.0015)
        distortion[:, 2] += (0.7, -0.5)
        red = cv2.warpAffine(green, distortion, green.shape[::-1])
        image = np.stack((red, green, green), axis=2)
        before = float(np.mean(np.abs(image[..., 0] - green)))
        corrected = correct_chromatic_aberration(image)
        after = float(np.mean(np.abs(corrected[..., 0] - green)))
        self.assertLess(after, before * 0.75)

    def test_processing_is_finite_and_bounded(self) -> None:
        yy, xx = np.mgrid[0:160, 0:220]
        gradient = 0.03 + xx / 2200.0 + yy / 3200.0
        image = np.repeat(gradient[..., None], 3, axis=2).astype(np.float32)
        image[40, 80] = 1.0
        image[90, 160] = 0.8
        options = ProcessingOptions()
        result = process_image(image, options)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_foreground_speckle_filter_preserves_sky(self) -> None:
        image = np.full((24, 32, 3), 0.1, np.float32)
        image[8, 7] = (1.0, 0.0, 0.0)
        image[8, 24] = (1.0, 0.0, 0.0)
        mask = np.zeros(image.shape[:2], np.float32)
        mask[:, :16] = 1.0
        result = suppress_foreground_speckles(image, mask)
        self.assertTrue(np.allclose(result[8, 7], 0.1))
        self.assertTrue(np.allclose(result[8, 24], image[8, 24]))

    def test_foreground_color_denoise_preserves_luminance_and_feathers_edge(self) -> None:
        rng = np.random.default_rng(12)
        height, width = 80, 100
        image = np.full((height, width, 3), 0.16, np.float32)
        mask = np.zeros((height, width), np.float32)
        mask[:, :50] = 1.0
        noise = rng.normal(0.0, 0.035, (height, 50, 3)).astype(np.float32)
        image[:, :50] += noise - noise.mean(axis=2, keepdims=True)
        before_luminance = luminance(image)
        before_chroma = image[:, :40] - before_luminance[:, :40, None]
        result = denoise_foreground_color(image, mask, dark_noise=0.01)
        after_luminance = luminance(result)
        after_chroma = result[:, :40] - after_luminance[:, :40, None]
        self.assertTrue(np.allclose(after_luminance, before_luminance, atol=2e-6))
        self.assertLess(float(after_chroma.std()), float(before_chroma.std()) * 0.8)
        edge_jump = np.abs(result[:, 49] - result[:, 50]).mean()
        self.assertLess(float(edge_jump), 0.018)

    def test_finish_does_not_reintroduce_hard_foreground_color_seam(self) -> None:
        image = np.full((64, 96, 3), 0.20, np.float32)
        image[:, :48] += np.array((0.04, -0.01, -0.03), np.float32)
        mask = np.zeros(image.shape[:2], np.float32)
        mask[:, :48] = 1.0
        options = ProcessingOptions(
            mode=StackMode.AVERAGE,
            auto_brightness=False,
            dynamic_denoise=False,
            correct_chromatic_aberration=False,
        )
        result = process_image(image, options, mask, dark_noise=0.01)
        before = float(np.linalg.norm(image[:, 47] - image[:, 48], axis=1).mean())
        after = float(np.linalg.norm(result[:, 47] - result[:, 48], axis=1).mean())
        self.assertLess(after, before * 0.15)

    def test_auto_finish_is_exposure_scale_invariant(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.uniform(0.01, 0.2, (80, 100, 3)).astype(np.float32)
        sigma = ProcessingOptions(mode=StackMode.SIGMA_CLIPPED, reduce_light_pollution=True, hdr=True, enhance_stars=True)
        summed = ProcessingOptions(mode=StackMode.SUM, reduce_light_pollution=True, hdr=True, enhance_stars=True)
        self.assertTrue(np.allclose(process_image(image, sigma), process_image(image * 37.0, summed), atol=2e-4))


class OutputTests(unittest.TestCase):
    def test_folder_output_writes_inside_folder(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            output = write_image(folder + "\\", np.zeros((3, 4, 3), np.float32))
            self.assertEqual(output.name, "astrostack.tiff")
            self.assertTrue(output.is_file())

    def test_stack_outputs_write_image_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            output = write_stack_outputs(
                Path(folder) / "result.tiff",
                np.zeros((3, 4, 3), np.float32),
                np.ones((3, 4), np.float32),
                np.zeros((3, 4), np.float32),
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name("result_coverage.png").is_file())
            self.assertTrue(output.with_name("result_foreground_mask.png").is_file())


if __name__ == "__main__":
    unittest.main()
