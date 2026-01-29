from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .detector import ImageResult, detect_qr_in_image
from .image_io import IMAGE_EXTS, scan_images
from .visualize import save_visualization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR image recognition (OpenCV QRCodeDetector)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--image", help="single image path")
    src.add_argument("-f", "--folder", help="folder containing images")

    # No output file; results are printed to stdout
    parser.add_argument("--no-recursive", action="store_true", help="disable recursive folder scan")
    parser.add_argument("--exts", nargs="*", default=None, help="custom extensions, e.g. .jpg .png")
    parser.add_argument("--save-vis", action="store_true", help="save visualization images")
    parser.add_argument("--vis-dir", default="vis", help="visualization output directory")
    parser.add_argument("--preprocess", action="store_true", help="enable preprocessing variants")
    parser.add_argument(
        "--manual-detect",
        action="store_true",
        help="use manual detect->decode pipeline (decode still by OpenCV)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="only detect polygons/boxes without decoding text",
    )

    return parser.parse_args()


def _relative_output_path(image_path: Path, root: Path, vis_dir: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        rel = image_path.name
    return vis_dir / rel


def _handle_single(
    image_path: Path,
    save_vis: bool,
    vis_dir: Path,
    preprocess: bool,
    manual_detect: bool,
    detect_only: bool,
) -> None:
    result = detect_qr_in_image(
        str(image_path), preprocess=preprocess, manual_detect=manual_detect, detect_only=detect_only
    )
    _print_result(result)
    if save_vis:
        output_path = vis_dir / image_path.name
        save_visualization(image_path, output_path, result)


def _handle_folder(
    folder: Path,
    recursive: bool,
    exts: List[str] | None,
    save_vis: bool,
    vis_dir: Path,
    preprocess: bool,
    manual_detect: bool,
    detect_only: bool,
) -> None:
    images = scan_images(folder, recursive=recursive, exts=exts)
    for image_path in images:
        result = detect_qr_in_image(
            str(image_path), preprocess=preprocess, manual_detect=manual_detect, detect_only=detect_only
        )
        _print_result(result)
        if save_vis:
            output_path = _relative_output_path(image_path, folder, vis_dir)
            save_visualization(image_path, output_path, result)


def _to_dict(result: ImageResult) -> dict:
    return {
        "image_path": result.image_path,
        "found": result.found,
        "qrcodes": [
            {"text": qr.text, "polygon": qr.polygon, "box": qr.box}
            for qr in result.qrcodes
        ],
        "error": result.error,
    }


def _print_result(result: ImageResult) -> None:
    header = f"[OK] {result.image_path}" if result.found else f"[NONE] {result.image_path}"
    print(header)
    if result.error:
        print(f"  error: {result.error}")
        return
    for idx, qr in enumerate(result.qrcodes, start=1):
        text = qr.text if qr.text else "(not decoded)"
        print(f"  qr#{idx}: {text}")
        print(f"    polygon: {qr.polygon}")
        print(f"    box: {qr.box}")


def main() -> None:
    args = _parse_args()
    vis_dir = Path(args.vis_dir)
    exts = args.exts or None

    if args.image:
        _handle_single(
            Path(args.image),
            args.save_vis,
            vis_dir,
            args.preprocess,
            args.manual_detect,
            args.detect_only,
        )
    else:
        recursive = not args.no_recursive
        _handle_folder(
            Path(args.folder),
            recursive,
            exts,
            args.save_vis,
            vis_dir,
            args.preprocess,
            args.manual_detect,
            args.detect_only,
        )


if __name__ == "__main__":
    main()
