"""
Data models cho module giám sát tài xế (drowsiness / distraction detection).

Được chuyển thể (port) từ:
  driver_safety/core/models.py  (repo: Inferensys/ai-driver-safety)
Giữ nguyên cấu trúc DriverState / Severity / DetectionEvent / SessionSummary
để tái sử dụng logic scoring & alert gốc của repo, chỉ đổi sang dùng
Pydantic-friendly dataclass cho dễ trả JSON qua FastAPI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class DriverState(str, Enum):
    ATTENTIVE = "attentive"
    EYES_CLOSED = "eyes_closed"
    DROWSY = "drowsy"
    YAWNING = "yawning"
    DISTRACTED = "distracted"
    PHONE_USE = "phone_use"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class FramePacket:
    """Một khung hình đầu vào (frame) kèm timestamp, dùng cho pipeline xử lý theo luồng video."""
    frame: Any
    timestamp: float
    frame_index: int
    source_id: str = "video"
    fps: Optional[float] = None


@dataclass(slots=True)
class DetectionEvent:
    timestamp: float
    frame_index: int
    signal: str
    state: DriverState
    score: float
    severity: Severity
    message: str
    bbox: Optional[tuple] = None
    landmarks: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True)
class ProcessedFrame:
    packet: FramePacket
    state: DriverState
    risk_score: float
    signals: dict
    events: list
    latency_ms: float
    face_bbox: Optional[tuple] = None
    landmarks: list = field(default_factory=list)
    objects: list = field(default_factory=list)


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    source: str
    duration_seconds: float
    processed_frames: int
    event_counts: dict
    risk_timeline: list
    longest_unsafe_interval_seconds: float
    confidence_distribution: dict
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)
