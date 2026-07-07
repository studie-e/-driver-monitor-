"""
NANG CAP (tuy chon): dung MediaPipe FaceLandmarker (478 landmark 3D that) thay
cho Haar cascade trong pipeline.py, cho EAR/MAR chinh xac hon han.

CACH DUNG:
  1. Tai model that (xem huong dan trong README hoac cau tra loi kem theo):
       wget -O face_landmarker.task \
         https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
     Dat file vao: backend/app/driver_monitor/models/face_landmarker.task

  2. Trong pipeline.py, thay dong khoi tao cascade:
        self._face_cascade = cv2.CascadeClassifier(...)
        self._eye_cascade   = cv2.CascadeClassifier(...)
        self._mouth_cascade = cv2.CascadeClassifier(...)
     bang:
        from app.driver_monitor.landmark_provider import MediaPipeLandmarkProvider
        self._landmark_provider = MediaPipeLandmarkProvider()

     Va thay toan bo khoi "gray = cv2.equalizeHist(...) ... faces = self._face_cascade..."
     trong process_frame() bang:
        result = self._landmark_provider.detect(packet.frame)
        if result is None:
            # -> nhanh missing_face nhu cu
        else:
            face_bbox, ear, mar, head_offset = result
            # -> dung truc tiep ear/mar/head_offset, KHONG can _box_to_ear_points nua
            # vi FaceLandmarker tra ve toa do that cua tung diem, khong phai bbox gia lap

  Cong thuc eye_aspect_ratio/mouth_aspect_ratio trong metrics.py KHONG doi -
  chi input dau vao chinh xac hon (diem that thay vi suy tu bbox).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from app.driver_monitor.metrics import (
    eye_aspect_ratio,
    horizontal_head_offset,
    mouth_aspect_ratio,
)

# Chi so landmark chuan cua MediaPipe FaceMesh (478 diem) dung cho EAR/MAR.
# Day la bo chi so pho bien nhat trong cong dong (khac voi 68-point dlib).
_LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
_MOUTH_IDX = [78, 82, 13, 308, 14, 312]

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")


class MediaPipeLandmarkProvider:
    """Bo do landmark chinh xac cao, thay the Haar cascade khi co model that."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Khong tim thay {model_path}. Hay tai face_landmarker.task theo huong dan "
                f"trong docstring dau file nay truoc khi dung MediaPipeLandmarkProvider."
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[tuple, float, float, float]]:
        """Tra ve (face_bbox, ear, mar, head_offset) hoac None neu khong thay mat."""
        h, w = frame_bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]  # list[NormalizedLandmark], da chuan hoa 0..1
        points = [(lm.x * w, lm.y * h) for lm in landmarks]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        face_bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

        left_eye = [points[i] for i in _LEFT_EYE_IDX]
        right_eye = [points[i] for i in _RIGHT_EYE_IDX]
        mouth = [points[i] for i in _MOUTH_IDX]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        head_offset = horizontal_head_offset(face_bbox, w)

        return face_bbox, ear, mar, head_offset
