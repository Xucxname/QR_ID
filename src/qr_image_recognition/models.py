from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QRCodeResult:
    text: str
    polygon: List[List[float]]
    box: List[float]


@dataclass
class ImageResult:
    image_path: str
    found: bool
    qrcodes: List[QRCodeResult]
    error: Optional[str] = None
