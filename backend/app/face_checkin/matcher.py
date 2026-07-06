"""
FaceMatcher - doi chieu vector dac trung khuon mat (face descriptor) hoc sinh.

Face detection + landmark + 128-D face descriptor duoc chay o PHIA TRINH DUYET
bang thu vien goc face-api.js (repo: justadudewhohacks/face-api.js), vi day
la thu vien JS/tensorflow.js chay tren camera truc tiep (xem
frontend/face_checkin.html). Backend Python nay chi nhan vector 128 chieu
(list[float]) da duoc trinh duyet tinh san qua:

    const result = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor()

roi port lai chinh xac cong thuc so khop cua `faceapi.FaceMatcher`
(euclidean distance + nguong 0.6 mac dinh cua face-api.js) de quyet dinh
hoc sinh nao trong danh sach la nguoi trung khop gan nhat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


# Nguong mac dinh cua faceapi.FaceMatcher (distanceThreshold = 0.6)
DEFAULT_DISTANCE_THRESHOLD = 0.6


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Port cong thuc euclideanDistance dung ben trong faceapi.FaceMatcher."""
    if len(a) != len(b):
        raise ValueError("Hai vector face descriptor phai cung so chieu (128)")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class LabeledDescriptor:
    """Tuong duong faceapi.LabeledFaceDescriptors: 1 hoc sinh <-> nhieu descriptor mau."""
    student_id: str
    full_name: str
    descriptors: List[List[float]]


@dataclass
class MatchResult:
    student_id: Optional[str]
    full_name: Optional[str]
    distance: float
    is_match: bool


class FaceMatcher:
    """Port logic cua faceapi.FaceMatcher.findBestMatch sang Python."""

    def __init__(self, labeled_descriptors: List[LabeledDescriptor],
                 distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> None:
        self.labeled_descriptors = labeled_descriptors
        self.distance_threshold = distance_threshold

    def find_best_match(self, query_descriptor: Sequence[float]) -> MatchResult:
        best_student: Optional[LabeledDescriptor] = None
        best_distance = float("inf")

        for entry in self.labeled_descriptors:
            # Voi moi hoc sinh, lay khoang cach NHO NHAT trong cac descriptor mau
            # (giong faceapi tinh mean/hoac min tuy cau hinh - o day dung min
            # de ben vung hon voi anh sang/goc mat khac nhau khi dang ky).
            min_d_for_student = min(
                euclidean_distance(query_descriptor, sample) for sample in entry.descriptors
            )
            if min_d_for_student < best_distance:
                best_distance = min_d_for_student
                best_student = entry

        is_match = best_student is not None and best_distance <= self.distance_threshold
        return MatchResult(
            student_id=best_student.student_id if is_match else None,
            full_name=best_student.full_name if is_match else None,
            distance=round(best_distance, 4),
            is_match=is_match,
        )
