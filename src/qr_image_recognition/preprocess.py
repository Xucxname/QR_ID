from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    # 自适应二值化，增强二维码在不同对比度下的可读性。
    gray = img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Stable binarization for low-contrast QR patterns.
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )


def preprocess_variants(img: np.ndarray) -> Iterable[Tuple[str, np.ndarray, float]]:
    # 生成用于检测/解码的预处理版本。
    gray = img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield "adapt", adaptive_threshold(gray), 1.0
