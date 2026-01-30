from __future__ import annotations

from typing import List

import cv2
import numpy as np


def polygon_to_box(polygon: List[List[float]]) -> List[float]:
    # 根据多边形点计算轴对齐外接框。
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def crop_by_polygons(
    img: np.ndarray, polygons: List[List[List[float]]], pad: float = 0.35
) -> tuple[np.ndarray, tuple[int, int], bool]:
    # 围绕检测到的二维码裁剪，并预留静区。
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


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def expand_polygon(pts: np.ndarray, scale: float = 1.08) -> np.ndarray:
    center = pts.mean(axis=0)
    return center + (pts - center) * scale


def warp_from_polygon(img: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
    # 透视矫正为正方形区域，便于解码/可视化。
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] != 4:
        return img
    pts = order_points(pts)
    pts = expand_polygon(pts, scale=1.08)
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
