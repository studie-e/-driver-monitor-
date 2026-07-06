"""
Các hàm đo tín hiệu khuôn mặt (EAR/MAR/head-offset).

Port trực tiếp từ:
  driver_safety/vision/metrics.py  (repo: Inferensys/ai-driver-safety)

Các hàm gốc (eye_aspect_ratio, mouth_aspect_ratio, horizontal_head_offset)
được giữ nguyên công thức toán học 100% - chỉ thêm docstring tiếng Việt.
Đầu vào "eye"/"mouth" là danh sách 6 điểm landmark (x, y), tương thích với
chuẩn 68-điểm của dlib hoặc có thể map từ MediaPipe FaceMesh.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> float:
    """EAR (Eye Aspect Ratio) - Soukupová & Čech (2016).

    EAR giảm mạnh khi mắt nhắm -> dùng để phát hiện nhắm mắt / ngủ gật.
    `eye` là 6 điểm landmark mắt theo thứ tự chuẩn 68-point (p1..p6).
    """
    if len(eye) < 6:
        raise ValueError("eye_aspect_ratio expects six eye landmarks")
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth: Sequence[Point]) -> float:
    """MAR (Mouth Aspect Ratio) - dùng để phát hiện ngáp (yawning)."""
    if len(mouth) < 6:
        raise ValueError("mouth_aspect_ratio expects at least six mouth landmarks")
    width = distance(mouth[0], mouth[3])
    vertical_a = distance(mouth[1], mouth[5])
    vertical_b = distance(mouth[2], mouth[4])
    if width == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * width)


def horizontal_head_offset(face_bbox: Tuple[int, int, int, int], frame_width: int) -> float:
    """Độ lệch ngang của đầu so với tâm khung hình, chuẩn hoá 0..1.

    Dùng để phát hiện tài xế quay đầu / mất tập trung (distracted).
    """
    x, _, w, _ = face_bbox
    center_x = x + w / 2.0
    frame_center = frame_width / 2.0
    if frame_center == 0:
        return 0.0
    return abs(center_x - frame_center) / frame_center
