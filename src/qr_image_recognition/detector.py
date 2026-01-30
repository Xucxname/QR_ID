from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .detect_core import (
    decode_from_polygons as _decode_from_polygons,
    detect_polygons as _detect_polygons,
    detect_polygons_with_variants as _detect_polygons_with_variants,
)
from .geometry import (
    crop_by_polygons as _crop_by_polygons,
    polygon_to_box as _polygon_to_box,
    warp_from_polygon as _warp_from_polygon,
)
from .models import ImageResult, QRCodeResult
from .preprocess import adaptive_threshold as _adaptive_threshold, preprocess_variants as _preprocess_variants


def _detect_base_polygons(detector: cv2.QRCodeDetector, img: np.ndarray) -> List[List[List[float]]]:
    # 先在全图粗检测，用于指导后续裁剪。
    polygons = _detect_polygons(detector, img)
    if polygons:
        return polygons
    return _detect_polygons_with_variants(detector, _preprocess_variants(img))


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_polygons(
    primary: List[List[List[float]]], secondary: List[List[List[float]]]
) -> List[List[List[float]]]:
    if not primary:
        return secondary
    merged = list(primary)
    for polygon in secondary:
        box = _polygon_to_box(polygon)
        if any(_box_iou(box, _polygon_to_box(existing)) > 0.3 for existing in merged):
            continue
        merged.append(polygon)
    return merged


def _crop_with_polygons(
    img: np.ndarray, polygons: List[List[List[float]]]
) -> tuple[np.ndarray, tuple[int, int], bool, List[List[List[float]]]]:
    # 裁剪缩小搜索范围，并保留坐标偏移关系。
    crop_img, (offset_x, offset_y), cropped = _crop_by_polygons(img, polygons)
    crop_polygons: List[List[List[float]]] = []
    if polygons:
        if cropped:
            crop_polygons = [
                [[x - offset_x, y - offset_y] for x, y in polygon]
                for polygon in polygons
            ]
        else:
            crop_polygons = polygons
    return crop_img, (offset_x, offset_y), cropped, crop_polygons


def _prepare_variants(
    img: np.ndarray, preprocess: bool
) -> List[Tuple[str, np.ndarray, float]]:
    # 物化预处理结果，便于复用与输出。
    return list(_preprocess_variants(img)) if preprocess else [("original", img, 1.0)]


def _apply_offset(results: List[QRCodeResult], offset_x: int, offset_y: int) -> None:
    # 将多边形坐标映射回原图坐标系。
    if not (offset_x or offset_y):
        return
    for qr in results:
        qr.polygon = [[x + offset_x, y + offset_y] for x, y in qr.polygon]
        qr.box = _polygon_to_box(qr.polygon)


def _merge_results(
    primary: List[QRCodeResult], secondary: List[QRCodeResult]
) -> List[QRCodeResult]:
    if not primary:
        return secondary
    merged = list(primary)
    for qr in secondary:
        if any(_box_iou(qr.box, existing.box) > 0.3 for existing in merged):
            continue
        merged.append(qr)
    return merged


def _collect_outputs(
    cropped: bool, crop_img: np.ndarray, variants: List[Tuple[str, np.ndarray, float]]
) -> List[Tuple[str, np.ndarray]]:
    # 输出中间结果便于检查。
    outputs: List[Tuple[str, np.ndarray]] = []
    if cropped:
        outputs.append(("crop", crop_img))
        outputs.extend([(f"crop_{name}", variant) for name, variant, _ in variants])
    else:
        outputs.extend([(name, variant) for name, variant, _ in variants])
    return outputs


def _collect_warps(
    crop_img: np.ndarray,
    polygons: List[List[List[float]]],
    variants: List[Tuple[str, np.ndarray, float]],
) -> List[Tuple[str, np.ndarray]]:
    # 输出透视矫正结果便于检查。
    if not polygons:
        return []
    adapt_enabled = any(name == "adapt" for name, _, _ in variants)
    warps: List[Tuple[str, np.ndarray]] = []
    # 保存透视矫正结果便于检查。
    for idx, polygon in enumerate(polygons, start=1):
        warped = _warp_from_polygon(crop_img, polygon)
        if adapt_enabled:
            warped = _adaptive_threshold(warped)
            warps.append((f"warp_adapt_{idx}", warped))
        else:
            warps.append((f"warp_{idx}", warped))
    return warps


def detect_qr_with_steps(
    image_path: str, preprocess: bool = True
) -> tuple[ImageResult, List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    # 高层流程：检测 -> 裁剪 -> 细化检测 -> 解码 -> 输出中间结果。
    detector = cv2.QRCodeDetector()
    img = cv2.imread(image_path)
    if img is None:
        return (
            ImageResult(image_path=image_path, found=False, qrcodes=[], error="failed to read image"),
            [],
            [],
        )

    base_polygons = _detect_base_polygons(detector, img)
    if preprocess:
        alt_polygons = _detect_polygons_with_variants(detector, _prepare_variants(img, True))
        base_polygons = _merge_polygons(base_polygons, alt_polygons)
    full_results = _decode_from_polygons(detector, img, base_polygons) if base_polygons else []
    crop_img, (offset_x, offset_y), cropped, crop_base_polygons = _crop_with_polygons(
        img, base_polygons
    )

    variants = _prepare_variants(crop_img, preprocess)
    polygons = _detect_polygons_with_variants(detector, variants)
    # Decode on the cropped region using detected polygons.
    results = _decode_from_polygons(detector, crop_img, polygons)
    warp_polygons = _merge_polygons(polygons, crop_base_polygons)
    if results:
        warp_polygons = _merge_polygons(warp_polygons, [qr.polygon for qr in results])
    _apply_offset(results, offset_x, offset_y)
    results = _merge_results(results, full_results)

    outputs = _collect_outputs(cropped, crop_img, variants)
    warps = _collect_warps(crop_img, warp_polygons, variants)

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
