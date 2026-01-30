from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .geometry import polygon_to_box, warp_from_polygon
from .models import QRCodeResult


def detect_polygons(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[List[List[float]]]:
    # OpenCV 检测入口，用于定位二维码多边形。
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


def detect_polygons_with_variants(
    detector: cv2.QRCodeDetector, variants: Iterable[Tuple[str, np.ndarray, float]]
) -> List[List[List[float]]]:
    # 依次尝试预处理版本，直到找到多边形。
    for _, variant, scale in variants:
        polygons = detect_polygons(detector, variant)
        if polygons:
            if scale != 1.0:
                scaled = []
                for polygon in polygons:
                    scaled.append([[x / scale, y / scale] for x, y in polygon])
                return scaled
            return polygons
    return []


def decode_from_polygons(
    detector: cv2.QRCodeDetector, img: np.ndarray, polygons: List[List[List[float]]]
) -> List[QRCodeResult]:
    # 使用多边形辅助解码，失败则透视矫正后再解码。
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
                results.append(
                    QRCodeResult(
                        text=text, polygon=polygon_list, box=polygon_to_box(polygon_list)
                    )
                )
            if results:
                return results
    except Exception:
        pass

    for polygon in polygons:
        warped = warp_from_polygon(img, polygon)
        text, _, _ = detector.detectAndDecode(warped)
        if not text:
            continue
        results.append(QRCodeResult(text=text, polygon=polygon, box=polygon_to_box(polygon)))
    return results
