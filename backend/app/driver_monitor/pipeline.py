"""
DriverSafetyPipeline - ham goi hoan chinh cho tinh nang "canh bao tai xe ngu gat".

Day la phan *glue code* noi cac ham da port o metrics.py / scoring.py / alerts.py
(tu repo Inferensys/ai-driver-safety) voi 1 bo do khuon mat + mat THAT chay
duoc ngay khong can tai model tu internet: dung Haar cascade co san trong
OpenCV (frontalface + eye + smile) de:
  - Tim bbox khuon mat (giong huong tiep can cua chinh webcam-pulse-detector)
  - Tim bbox 2 mat -> suy ra 6 diem gia lap chuan EAR tu bbox mat (khi Haar
    cascade KHONG con detect duoc mat trong ROI mat, coi nhu mat dang nham -
    day la tin hieu manh vi haarcascade_eye chi bat duoc mat dang mo ro).
  - Tim bbox mieng cuoi/ngap qua haarcascade_smile (mo rong bat thuong) -> suy
    ra 6 diem gia lap chuan MAR.

Ghi chu: ban goc ai-driver-safety dung MediaPipe FaceLandmarker (.task model
tai tu Google) hoac dlib 68-point cho do chinh xac EAR/MAR cao hon. Trong moi
truong khong co internet de tai model, huong tiep can Haar-cascade nay van
dung DUNG cong thuc eye_aspect_ratio/mouth_aspect_ratio goc, chi khac o buoc
lay landmark tho hon. Khi trien khai that, co the thay bo dò nay bang
MediaPipe FaceLandmarker/dlib ma KHONG can sua metrics.py/scoring.py/alerts.py.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app.driver_monitor.alerts import Alert, AlertPolicy
from app.driver_monitor.metrics import (
    eye_aspect_ratio,
    horizontal_head_offset,
    mouth_aspect_ratio,
)
from app.driver_monitor.models import (
    DetectionEvent,
    DriverState,
    FramePacket,
    ProcessedFrame,
    Severity,
)
from app.driver_monitor.scoring import RiskScorer

# Nguong mac dinh port tu driver_safety/config.py (Thresholds dataclass)
DEFAULT_THRESHOLDS = {
    "eye_aspect_ratio": 0.22,
    "mouth_aspect_ratio": 0.5,
    "head_offset": 0.42,
    "missing_face_frames": 8,
    "eye_closed_frames": 8,
    "drowsy_frames": 36,
    "yawn_frames": 6,
    "distracted_frames": 12,
}


def _box_to_ear_points(x: int, y: int, w: int, h: int):
    """Suy ra 6 diem EAR gia lap tu 1 bbox mat (Haar cascade tra ve hinh chu nhat)."""
    return [
        (x, y + h / 2.0),
        (x + w * 0.3, y),
        (x + w * 0.7, y),
        (x + w, y + h / 2.0),
        (x + w * 0.7, y + h),
        (x + w * 0.3, y + h),
    ]


def _box_to_mar_points(x: int, y: int, w: int, h: int):
    """Suy ra 6 diem MAR gia lap tu 1 bbox mieng/cuoi (Haar cascade smile)."""
    return [
        (x, y + h / 2.0),
        (x + w * 0.3, y),
        (x + w * 0.7, y),
        (x + w, y + h / 2.0),
        (x + w * 0.7, y + h),
        (x + w * 0.3, y + h),
    ]


class DriverSafetyPipeline:
    """Xu ly tung khung hinh webcam cua tai xe -> risk_score + alerts."""

    def __init__(self, thresholds: Optional[dict] = None, alert_cooldown_seconds: float = 2.0) -> None:
        cascade_dir = cv2.data.haarcascades
        self._face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt.xml")
        self._eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")
        self._mouth_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_smile.xml")

        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.scorer = RiskScorer()
        self.alert_policy = AlertPolicy(alert_cooldown_seconds)

        self._closed_counter = 0
        self._yawn_counter = 0
        self._distracted_counter = 0
        self._missing_face_counter = 0

    def process_frame(self, packet: FramePacket) -> ProcessedFrame:
        started = time.perf_counter()
        raw_signals = {
            "eyes_closed": 0.0,
            "drowsy": 0.0,
            "yawning": 0.0,
            "distracted": 0.0,
            "phone_use": 0.0,
        }
        events: list = []
        face_bbox = None
        landmarks_out: list = []

        gray = cv2.equalizeHist(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY))
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            self._missing_face_counter += 1
            if self._missing_face_counter >= self.thresholds["missing_face_frames"]:
                raw_signals["distracted"] = 1.0
                events.append(
                    self._event(
                        packet, "distracted", DriverState.DISTRACTED, 1.0, Severity.WARNING,
                        "Distracted: khong tim thay khuon mat tai xe trong khung hinh",
                    )
                )
        else:
            self._missing_face_counter = 0
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            face_bbox = (int(fx), int(fy), int(fw), int(fh))
            face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

            # --- Mat: chi tim trong nua tren cua khuon mat de tranh nham voi mui/mieng ---
            upper_half = face_roi_gray[0: int(fh * 0.6), :]
            eyes = self._eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))

            ear_points = []
            for (ex, ey, ew, eh) in eyes[:2]:
                abs_x, abs_y = fx + ex, fy + ey
                ear_points.append(_box_to_ear_points(abs_x, abs_y, ew, eh))
                landmarks_out.extend([(abs_x, abs_y), (abs_x + ew, abs_y + eh)])

            if len(ear_points) >= 1:
                ear = float(np.mean([eye_aspect_ratio(p) for p in ear_points]))
                eyes_detected = True
            else:
                # Haar cascade khong tim thay mat mo -> gia dinh mat dang NHAM
                ear = 0.0
                eyes_detected = False

            # --- Mieng: dung haarcascade_smile o nua duoi khuon mat de uoc luong MAR ---
            lower_half = face_roi_gray[int(fh * 0.55):, :]
            mouths = self._mouth_cascade.detectMultiScale(lower_half, scaleFactor=1.5, minNeighbors=15, minSize=(25, 15))
            if len(mouths) > 0:
                mx, my, mw, mh = mouths[0]
                abs_y = fy + int(fh * 0.55) + my
                mar_points = _box_to_mar_points(fx + mx, abs_y, mw, mh)
                mar = mouth_aspect_ratio(mar_points)
                landmarks_out.append((fx + mx, abs_y))
            else:
                mar = 0.0

            head_offset = horizontal_head_offset(face_bbox, packet.frame.shape[1])

            th = self.thresholds
            self._closed_counter = self._closed_counter + 1 if (not eyes_detected or ear < th["eye_aspect_ratio"]) else 0
            self._yawn_counter = self._yawn_counter + 1 if mar > th["mouth_aspect_ratio"] else 0
            self._distracted_counter = (
                self._distracted_counter + 1 if head_offset > th["head_offset"] else 0
            )

            raw_signals.update(
                {
                    "eyes_closed": min(1.0, self._closed_counter / max(1, th["eye_closed_frames"])),
                    "drowsy": min(1.0, self._closed_counter / max(1, th["drowsy_frames"])),
                    "yawning": min(1.0, self._yawn_counter / max(1, th["yawn_frames"])),
                    "distracted": min(1.0, self._distracted_counter / max(1, th["distracted_frames"])),
                    "ear": round(ear, 4),
                    "mar": round(mar, 4),
                    "head_offset": round(head_offset, 4),
                }
            )
            events.extend(self._events_from_counters(packet, raw_signals, face_bbox, landmarks_out))

        risk_score = self.scorer.score(raw_signals)
        state = self.scorer.state_from_events(events, risk_score)
        latency_ms = (time.perf_counter() - started) * 1000

        return ProcessedFrame(
            packet=packet,
            state=state,
            risk_score=risk_score,
            signals=raw_signals,
            events=events,
            latency_ms=latency_ms,
            face_bbox=face_bbox,
            landmarks=landmarks_out,
        )

    def alerts_for(self, processed: ProcessedFrame) -> list:
        """Ap AlertPolicy (cooldown) len cac event cua 1 frame da xu ly."""
        return self.alert_policy.evaluate(processed.events)

    def _events_from_counters(self, packet, signals, bbox, landmarks) -> list:
        events = []
        th = self.thresholds
        if self._closed_counter >= th["eye_closed_frames"]:
            events.append(self._event(packet, "eyes_closed", DriverState.EYES_CLOSED,
                                       signals["eyes_closed"], Severity.WARNING,
                                       "Mat nham vuot nguong cau hinh", bbox, landmarks))
        if self._closed_counter >= th["drowsy_frames"]:
            events.append(self._event(packet, "drowsy", DriverState.DROWSY,
                                       signals["drowsy"], Severity.CRITICAL,
                                       "Nham mat keo dai - dau hieu ngu gat", bbox, landmarks))
        if self._yawn_counter >= th["yawn_frames"]:
            events.append(self._event(packet, "yawning", DriverState.YAWNING,
                                       signals["yawning"], Severity.WARNING,
                                       "Phat hien ngap tu vung mieng", bbox, landmarks))
        if self._distracted_counter >= th["distracted_frames"]:
            events.append(self._event(packet, "distracted", DriverState.DISTRACTED,
                                       signals["distracted"], Severity.WARNING,
                                       "Huong dau cho thay tai xe dang nhin di noi khac", bbox, landmarks))
        return events

    @staticmethod
    def _event(packet, signal, state, score, severity, message, bbox=None, landmarks=None) -> DetectionEvent:
        return DetectionEvent(
            timestamp=packet.timestamp,
            frame_index=packet.frame_index,
            signal=signal,
            state=state,
            score=score,
            severity=severity,
            message=message,
            bbox=bbox,
            landmarks=landmarks or [],
        )
