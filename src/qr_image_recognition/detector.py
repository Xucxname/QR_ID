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


def _preprocess_variants(img: np.ndarray) -> Iterable[Tuple[str, np.ndarray, float]]:
    yield "original", img, 1.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield "gray", gray, 1.0

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    yield "denoised", denoised, 1.0

    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "thresh", thresh, 1.0

    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        yield "scaled", scaled, 2.0


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
    detector: cv2.QRCodeDetector, variants: Iterable[Tuple[str, np.ndarray, float]]
) -> List[List[List[float]]]:
    for _, variant, scale in variants:
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
        decoded = detector.decodeMulti(img, np.array(polygons, dtype=np.float32))
        if len(decoded) == 4:
            ok, decoded_info, points, _ = decoded
        else:
            ok, decoded_info, points = decoded[0], decoded[1], decoded[2]

        if ok and decoded_info is not None:
            if points is not None and isinstance(points, np.ndarray) and points.ndim == 3:
                source_polygons = points
            else:
                source_polygons = np.array(polygons, dtype=np.float32)

            for text, polygon in zip(decoded_info, source_polygons):
                if not text:
                    continue
                polygon_list = [[float(x), float(y)] for x, y in polygon]
                results.append(QRCodeResult(text=text, polygon=polygon_list, box=_polygon_to_box(polygon_list)))
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


def detect_qr_in_image(image_path: str, preprocess: bool = True) -> ImageResult:
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    if img is None:
        return ImageResult(image_path=image_path, found=False, qrcodes=[], error="failed to read image")

    variants = _preprocess_variants(img) if preprocess else [("original", img, 1.0)]
    polygons = _detect_polygons_with_variants(detector, variants)
    results = _decode_from_polygons(detector, img, polygons)

    return ImageResult(
        image_path=image_path,
        found=len(results) > 0,
        qrcodes=results,
        error=None,
    )


def detect_qr_in_image_data(image_path: str, preprocess: bool = True) -> Dict[str, Any]:
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


def preprocess_outputs(image_path: str) -> List[Tuple[str, np.ndarray]]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    outputs: List[Tuple[str, np.ndarray]] = []
    for name, variant, _ in _preprocess_variants(img):
        outputs.append((name, variant))
    return outputs
