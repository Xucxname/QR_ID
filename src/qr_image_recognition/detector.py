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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", clahe, 1.0

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    yield "denoised", denoised, 1.0

    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "thresh", thresh, 1.0
    adapt = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    yield "adapt", adapt, 1.0

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


def _crop_by_polygons(
    img: np.ndarray, polygons: List[List[List[float]]], pad: float = 0.35
) -> tuple[np.ndarray, tuple[int, int], bool]:
    if not polygons:
        return img, (0, 0), False
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    x_min -= w * pad
    x_max += w * pad
    y_min -= h * pad
    y_max += h * pad

    h_img, w_img = img.shape[:2]
    x0 = max(int(x_min), 0)
    y0 = max(int(y_min), 0)
    x1 = min(int(x_max), w_img - 1)
    y1 = min(int(y_max), h_img - 1)

    if x1 <= x0 or y1 <= y0:
        return img, (0, 0), False
    return img[y0 : y1 + 1, x0 : x1 + 1], (x0, y0), True


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _expand_polygon(pts: np.ndarray, scale: float = 1.08) -> np.ndarray:
    center = pts.mean(axis=0)
    return center + (pts - center) * scale


def _warp_from_polygon(img: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] != 4:
        return img
    pts = _order_points(pts)
    pts = _expand_polygon(pts, scale=1.08)
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


def detect_qr_with_steps(
    image_path: str, preprocess: bool = True
) -> tuple[ImageResult, List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    if img is None:
        return (
            ImageResult(image_path=image_path, found=False, qrcodes=[], error="failed to read image"),
            [],
            [],
        )

    base_polygons = _detect_polygons(detector, img)
    if not base_polygons:
        base_polygons = _detect_polygons_with_variants(detector, _preprocess_variants(img))
    crop_img, (offset_x, offset_y), cropped = _crop_by_polygons(img, base_polygons)

    variants = _preprocess_variants(crop_img) if preprocess else [("original", crop_img, 1.0)]
    polygons = _detect_polygons_with_variants(detector, variants)
    results = _decode_from_polygons(detector, crop_img, polygons)

    if offset_x or offset_y:
        for qr in results:
            qr.polygon = [[x + offset_x, y + offset_y] for x, y in qr.polygon]
            qr.box = _polygon_to_box(qr.polygon)

    outputs: List[Tuple[str, np.ndarray]] = []
    if cropped:
        outputs.append(("crop", crop_img))
        outputs.extend([(f"crop_{name}", variant) for name, variant, _ in variants])
    else:
        outputs.extend([(name, variant) for name, variant, _ in variants])

    warps: List[Tuple[str, np.ndarray]] = []
    source_polygons = [qr.polygon for qr in results] if results else polygons
    adapt_img = None
    for name, variant, _ in variants:
        if name == "adapt":
            adapt_img = variant
            break

    for idx, polygon in enumerate(source_polygons, start=1):
        warped = _warp_from_polygon(crop_img, polygon)
        warps.append((f"warp_{idx}", warped))
        if adapt_img is not None:
            warped_adapt = _warp_from_polygon(adapt_img, polygon)
            warps.append((f"warp_adapt_{idx}", warped_adapt))

    return (
        ImageResult(
            image_path=image_path,
            found=len(results) > 0,
            qrcodes=results,
            error=None,
        ),
        outputs,
        warps,
    )


def detect_qr_in_image(image_path: str, preprocess: bool = True) -> ImageResult:
    result, _, _ = detect_qr_with_steps(image_path, preprocess=preprocess)
    return result


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
