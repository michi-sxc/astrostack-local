from __future__ import annotations

import argparse
from pathlib import Path

from .io import discover_images, write_stack_outputs
from .models import ProcessingOptions, StackMode, StackRequest
from .pipeline import stack_images


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coverage-aware RAW astrophotography stacker")
    parser.add_argument("--lights", nargs="+", required=True, help="light files and/or folders")
    parser.add_argument("--darks", nargs="*", default=(), help="dark files and/or folders")
    parser.add_argument("--output", required=True, help="output .tiff, .png, or .jpg")
    parser.add_argument("--mode", choices=[mode.value for mode in StackMode], default=StackMode.SIGMA_CLIPPED.value)
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--half-size", action="store_true", help="use a faster half-resolution preview")
    resolution.add_argument("--full-size", action="store_true", help="full resolution (the default)")
    parser.add_argument("--mask", help="white-foreground mask image")
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--foreground", action="store_true", help="use the explicit --mask")
    parser.add_argument("--light-pollution", action="store_true")
    parser.add_argument("--hdr", action="store_true")
    parser.add_argument("--star-enhancement", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-distortion-correction", action="store_true")
    parser.add_argument("--no-chromatic-aberration-correction", action="store_true")
    parser.add_argument("--no-auto-brightness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lights = discover_images(args.lights)
    darks = discover_images(args.darks)
    options = ProcessingOptions(
        mode=StackMode(args.mode),
        half_size=args.half_size,
        min_coverage=args.min_coverage,
        auto_crop=not args.no_crop,
        auto_brightness=not args.no_auto_brightness,
        protect_foreground=args.foreground or bool(args.mask),
        foreground_mask=Path(args.mask).resolve() if args.mask else None,
        reduce_light_pollution=args.light_pollution,
        hdr=args.hdr,
        enhance_stars=args.star_enhancement,
        dynamic_denoise=not args.no_denoise,
        correct_distortion=not args.no_distortion_correction,
        correct_chromatic_aberration=not args.no_chromatic_aberration_correction,
    )

    def progress(stage: str, current: int, total: int) -> None:
        print(f"[{stage}] {current}/{total}", flush=True)

    result = stack_images(StackRequest(lights, darks, options, progress, print))
    output = write_stack_outputs(args.output, result.image, result.coverage, result.foreground_mask)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
