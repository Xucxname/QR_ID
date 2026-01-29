from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .detector import ImageResult, detect_qr_in_image
from .image_io import IMAGE_EXTS, scan_images
from .visualize import save_visualization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR image recognition (OpenCV QRCodeDetector)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--image", help="single image path")
    src.add_argument("-f", "--folder", help="folder containing images")

    parser.add_argument("-o", "--output-jsonl", default="results.jsonl", help="output JSONL path")
    parser.add_argument("--no-recursive", action="store_true", help="disable recursive folder scan")
    parser.add_argument("--exts", nargs="*", default=None, help="custom extensions, e.g. .jpg .png")
    parser.add_argument("--save-vis", action="store_true", help="save visualization images")
    parser.add_argument("--vis-dir", default="vis", help="visualization output directory")
    parser.add_argument("--preprocess", action="store_true", help="enable preprocessing variants")

    return parser.parse_args()


def _relative_output_path(image_path: Path, root: Path, vis_dir: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        rel = image_path.name
    return vis_dir / rel


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _handle_single(
    image_path: Path,
    output_jsonl: Path,
    save_vis: bool,
    vis_dir: Path,
    preprocess: bool,
) -> None:
    result = detect_qr_in_image(str(image_path), preprocess=preprocess)
    record = _to_dict(result)
    print(json.dumps(record, ensure_ascii=False))
    _write_jsonl(output_jsonl, [record])
    if save_vis:
        output_path = vis_dir / image_path.name
        save_visualization(image_path, output_path, result)


def _handle_folder(
    folder: Path,
    output_jsonl: Path,
    recursive: bool,
    exts: List[str] | None,
    save_vis: bool,
    vis_dir: Path,
    preprocess: bool,
) -> None:
    images = scan_images(folder, recursive=recursive, exts=exts)
    records = []
    for image_path in images:
        result = detect_qr_in_image(str(image_path), preprocess=preprocess)
        record = _to_dict(result)
        print(json.dumps(record, ensure_ascii=False))
        records.append(record)
        if save_vis:
            output_path = _relative_output_path(image_path, folder, vis_dir)
            save_visualization(image_path, output_path, result)
    _write_jsonl(output_jsonl, records)


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


def main() -> None:
    args = _parse_args()
    output_jsonl = Path(args.output_jsonl)
    vis_dir = Path(args.vis_dir)
    exts = args.exts or None

    if args.image:
        _handle_single(Path(args.image), output_jsonl, args.save_vis, vis_dir, args.preprocess)
    else:
        recursive = not args.no_recursive
        _handle_folder(Path(args.folder), output_jsonl, recursive, exts, args.save_vis, vis_dir, args.preprocess)


if __name__ == "__main__":
    main()
