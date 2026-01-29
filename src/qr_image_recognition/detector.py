from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _preprocess_variants(img: np.ndarray) -> Iterable[Tuple[np.ndarray, float]]:
    yield img, 1.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield gray, 1.0

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    yield denoised, 1.0

    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield thresh, 1.0

    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        yield scaled, 2.0


def _detect_with_variants(detector: cv2.QRCodeDetector, variants: Iterable[np.ndarray]) -> List[QRCodeResult]:
    for variant in variants:
        results = _decode_multi(detector, variant)
        if results:
            return results
        results = _decode_single(detector, variant)
        if results:
            return results
    return []


def _detect_polygons(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[List[List[float]]]:
    polygons: List[List[List[float]]] = []
    try:
        ok, points = detector.detectMulti(img)
    except AttributeError:
        ok, points = False, None
    if ok and points is not None:
        for polygon in points:
            polygon_list = [[float(x), float(y)] for x, y in polygon]
            polygons.append(polygon_list)
        return polygons

    ok, points = detector.detect(img)
    if ok and points is not None:
        pts = np.array(points).reshape(-1, 2)
        polygons.append([[float(x), float(y)] for x, y in pts])
    return polygons


def _detect_polygons_with_variants(
    detector: cv2.QRCodeDetector, variants: Iterable[Tuple[np.ndarray, float]]
) -> List[List[List[float]]]:
    for variant, scale in variants:
        polygons = _detect_polygons(detector, variant)
        if polygons:
            if scale != 1.0:
                scaled = []
                for polygon in polygons:
                    scaled.append([[x / scale, y / scale] for x, y in polygon])
                return scaled
            return polygons
    return []


def _warp_from_polygon(img: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] != 4:
        return img
    width_a = np.linalg.norm(pts[0] - pts[1])
    width_b = np.linalg.norm(pts[2] - pts[3])
    height_a = np.linalg.norm(pts[1] - pts[2])
    height_b = np.linalg.norm(pts[3] - pts[0])
    side = max(int(width_a), int(width_b), int(height_a), int(height_b), 32)
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (side, side))


def _decode_from_polygons(detector: cv2.QRCodeDetector, img: np.ndarray, polygons: List[List[List[float]]]) -> List[QRCodeResult]:
    results: List[QRCodeResult] = []
    if not polygons:
        return results

    try:
        ok, decoded_info, points, _ = detector.decodeMulti(img, np.array(polygons, dtype=np.float32))
        if ok and decoded_info is not None and points is not None:
            for text, polygon in zip(decoded_info, points):
                if not text:
                    continue
                polygon_list = [[float(x), float(y)] for x, y in polygon]
                results.append(
                    QRCodeResult(text=text, polygon=polygon_list, box=_polygon_to_box(polygon_list))
                )
            if results:
                return results
    except Exception:
        pass

    for polygon in polygons:
        warped = _warp_from_polygon(img, polygon)
        text, _, _ = detector.detectAndDecode(warped)
        if not text:
            continue
        results.append(QRCodeResult(text=text, polygon=polygon, box=_polygon_to_box(polygon)))
    return results


def detect_qr_in_image(
    image_path: str,
    preprocess: bool = False,
    detect_only: bool = False,
    manual_detect: bool = False,
) -> ImageResult:
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    if img is None:
        return ImageResult(image_path=image_path, found=False, qrcodes=[], error="failed to read image")

    if manual_detect:
        variants = _preprocess_variants(img) if preprocess else [(img, 1.0)]
        polygons = _detect_polygons_with_variants(detector, variants)
        if detect_only:
            results = [
                QRCodeResult(text="", polygon=polygon, box=_polygon_to_box(polygon))
                for polygon in polygons
            ]
        else:
            results = _decode_from_polygons(detector, img, polygons)
    else:
        if preprocess:
            results = _detect_with_variants(detector, (v for v, _ in _preprocess_variants(img)))
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


def detect_qr_in_image_data(
    image_path: str,
    preprocess: bool = False,
    detect_only: bool = False,
    manual_detect: bool = False,
) -> Dict[str, Any]:
    result = detect_qr_in_image(
        image_path, preprocess=preprocess, detect_only=detect_only, manual_detect=manual_detect
    )
    return {
        "image_path": result.image_path,
        "found": result.found,
        "qrcodes": [
            {"text": qr.text, "polygon": qr.polygon, "box": qr.box}
            for qr in result.qrcodes
        ],
        "error": result.error,
    }
