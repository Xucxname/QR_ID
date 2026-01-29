from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2

from qr_image_recognition.crypto_utils import DEFAULT_KEY_PATH, load_key_from_path, try_decrypt_text
from qr_image_recognition.detector import ImageResult, detect_qr_in_image, preprocess_outputs
from qr_image_recognition.qr_gen import generate_qr
from qr_image_recognition.visualize import save_visualization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR ID runner (images folder)")
    parser.add_argument("--name", required=True, help="image filename under ./images")
    parser.add_argument("--gen-text", help="generate QR code with text and save to images/")
    parser.add_argument(
        "--crypto",
        choices=["none", "fernet", "xor"],
        default="none",
        help="encryption method (default: none)",
    )
    parser.add_argument(
        "--key",
        help="path to key file (default: config/qr.key)",
    )
    return parser.parse_args()


def _print_result(result: ImageResult, method: str, key: str | None) -> None:
    header = f"[OK] {result.image_path}" if result.found else f"[NONE] {result.image_path}"
    print(header)
    if result.error:
        print(f"  error: {result.error}")
        return
    for idx, qr in enumerate(result.qrcodes, start=1):
        text = qr.text if qr.text else ""
        if not text:
            print(f"  qr#{idx}: (not decoded)")
        else:
            ok, plain = try_decrypt_text(text, method, key)
            if ok:
                print(f"  qr#{idx}: {plain}")
            else:
                print(f"  qr#{idx}: (decrypt failed)")
        print(f"    polygon: {qr.polygon}")
        print(f"    box: {qr.box}")


def main() -> None:
    args = _parse_args()
    image_path = Path("images") / args.name
    key_path = Path(args.key) if args.key else DEFAULT_KEY_PATH
    key = load_key_from_path(key_path)
    method = args.crypto if key else "none"

    if args.gen_text:
        output_path = generate_qr(args.gen_text, image_path, method=method, key=key)
        print(f"[GEN] {output_path}")
        return

    result = detect_qr_in_image(
        str(image_path),
        preprocess=True,
    )
    _print_result(result, method, key)

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
            text = qr.text if qr.text else ""
            if not text:
                f.write(f"qr#{idx}: (not decoded)\n")
            else:
                ok, plain = try_decrypt_text(text, method, key)
                if ok:
                    f.write(f"qr#{idx}: {plain}\n")
                else:
                    f.write(f"qr#{idx}: (decrypt failed)\n")
            f.write(f"polygon: {qr.polygon}\n")
            f.write(f"box: {qr.box}\n")


if __name__ == "__main__":
    main()
