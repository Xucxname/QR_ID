from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .detector import ImageResult


def draw_qr_results(image: np.ndarray, result: ImageResult) -> np.ndarray:
    canvas = image.copy()
    for qr in result.qrcodes:
        pts = np.array(qr.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if qr.text:
            x, y = int(qr.polygon[0][0]), int(qr.polygon[0][1]) - 6
            cv2.putText(
                canvas,
                qr.text,
                (x, max(y, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def save_visualization(
    image_path: Path,
    output_path: Path,
    result: ImageResult,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = draw_qr_results(img, result)
    cv2.imwrite(str(output_path), vis)
