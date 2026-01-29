from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np


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


def _polygon_to_box(polygon: List[List[float]]) -> List[float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _decode_multi(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[QRCodeResult]:
    ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
    results: List[QRCodeResult] = []
    if not ok or points is None or decoded_info is None:
        return results
    for text, polygon in zip(decoded_info, points):
        if not text:
            continue
        polygon_list = [[float(x), float(y)] for x, y in polygon]
        results.append(QRCodeResult(text=text, polygon=polygon_list, box=_polygon_to_box(polygon_list)))
    return results


def _decode_single(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[QRCodeResult]:
    text, points, _ = detector.detectAndDecode(img)
    if not text or points is None:
        return []
    pts = np.array(points).reshape(-1, 2)
    polygon_list = [[float(x), float(y)] for x, y in pts]
    return [QRCodeResult(text=text, polygon=polygon_list, box=_polygon_to_box(polygon_list))]


def _preprocess_variants(img: np.ndarray) -> Iterable[np.ndarray]:
    yield img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield gray

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    yield denoised

    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield thresh

    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        yield scaled


def _detect_with_variants(detector: cv2.QRCodeDetector, variants: Iterable[np.ndarray]) -> List[QRCodeResult]:
    for variant in variants:
        results = _decode_multi(detector, variant)
        if results:
            return results
        results = _decode_single(detector, variant)
        if results:
            return results
    return []


def detect_qr_in_image(image_path: str, preprocess: bool = False) -> ImageResult:
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    if img is None:
        return ImageResult(image_path=image_path, found=False, qrcodes=[], error="failed to read image")

    if preprocess:
        results = _detect_with_variants(detector, _preprocess_variants(img))
    else:
        results = _decode_multi(detector, img)
        if not results:
            results = _decode_single(detector, img)

    return ImageResult(
        image_path=image_path,
        found=len(results) > 0,
        qrcodes=results,
        error=None,
    )


def detect_qr_in_image_data(image_path: str, preprocess: bool = False) -> Dict[str, Any]:
    result = detect_qr_in_image(image_path, preprocess=preprocess)
    return {
        "image_path": result.image_path,
        "found": result.found,
        "qrcodes": [
            {"text": qr.text, "polygon": qr.polygon, "box": qr.box}
            for qr in result.qrcodes
        ],
        "error": result.error,
    }
