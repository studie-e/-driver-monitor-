"""
face_roi.py - phat hien khuon mat + khoanh vung ROI tran.

Port NGUYEN XI tu rppg_core.py (nguoi dung tu viet): dung Haar cascade co san
trong OpenCV (khong can tai model, chay offline duoc ngay) de tim bbox khuon
mat, roi khoanh vung tran theo ty le co dinh so voi bbox do.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from app.driver_monitor.landmark_provider import MediaPipeLandmarkProvider
    _landmarker = MediaPipeLandmarkProvider()
except (FileNotFoundError, ImportError):
    _landmarker = None

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_box(frame_bgr: np.ndarray, gray_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Phat hien khuon mat lon nhat. Uu tien MediaPipe (.task), fallback Haar cascade."""
    if _landmarker is not None:
        result = _landmarker.detect(frame_bgr)
        if result is not None:
            return result[0]
        return None

    faces = _face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def get_forehead_roi_from_face_box(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Tu bounding box khuon mat (Haar cascade tra ve), khoanh vung tran:
    tran nam o phan tren cua mat, thu hep theo chieu ngang de tranh lan toc mai/tai.
    Day la cach lam pho bien trong cac pipeline rPPG khi khong co landmark chi tiet.
    """
    x1 = x + int(w * 0.30)
    x2 = x + int(w * 0.70)
    y1 = y + int(h * 0.08)
    y2 = y + int(h * 0.28)
    return x1, y1, x2, y2
