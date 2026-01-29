from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2

from qr_image_recognition.detector import ImageResult, detect_qr_in_image, preprocess_outputs
from qr_image_recognition.visualize import save_visualization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR ID runner (images folder)")
    parser.add_argument("--name", required=True, help="image filename under ./images")
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
        preprocess=True,
    )
    _print_result(result)

    output_dir = Path("output") / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, variant in preprocess_outputs(str(image_path)):
        output_path = output_dir / f"{name}.png"
        if len(variant.shape) == 2:
            cv2.imwrite(str(output_path), variant)
        else:
            cv2.imwrite(str(output_path), variant)

    vis_path = output_dir / "visualization.png"
    save_visualization(image_path, vis_path, result)

    summary_path = output_dir / "result.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        header = f"[OK] {result.image_path}" if result.found else f"[NONE] {result.image_path}"
        f.write(header + "\n")
        if result.error:
            f.write(f"error: {result.error}\n")
            return
        for idx, qr in enumerate(result.qrcodes, start=1):
            text = qr.text if qr.text else "(not decoded)"
            f.write(f"qr#{idx}: {text}\n")
            f.write(f"polygon: {qr.polygon}\n")
            f.write(f"box: {qr.box}\n")


if __name__ == "__main__":
    main()
