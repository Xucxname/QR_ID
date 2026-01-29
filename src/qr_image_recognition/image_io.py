from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path, exts: Iterable[str] | None = None) -> bool:
    allowed = {e.lower() for e in (exts or IMAGE_EXTS)}
    return path.is_file() and path.suffix.lower() in allowed


def scan_images(folder: Path, recursive: bool = True, exts: Iterable[str] | None = None) -> List[Path]:
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.glob("*")
    return [p for p in candidates if is_image_file(p, exts)]
