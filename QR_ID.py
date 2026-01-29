from __future__ import annotations

import argparse
from pathlib import Path

from qr_image_recognition.detector import ImageResult, detect_qr_in_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR ID runner (images folder)")
    parser.add_argument("--name", required=True, help="image filename under ./images")
    parser.add_argument("--preprocess", action="store_true", help="enable preprocessing variants")
    parser.add_argument("--save-vis", action="store_true", help="save visualization image")
    parser.add_argument("--vis-dir", default="vis", help="visualization output directory")
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
    image_path = Path("images") / args.name
    result = detect_qr_in_image(
        str(image_path),
        preprocess=args.preprocess,
        manual_detect=args.manual_detect,
        detect_only=args.detect_only,
    )
    _print_result(result)

    if args.save_vis:
        from qr_image_recognition.visualize import save_visualization

        output_path = Path(args.vis_dir) / image_path.name
        save_visualization(image_path, output_path, result)


if __name__ == "__main__":
    main()
