from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- 1. Face check-in ----------
class RegisterStudentRequest(BaseModel):
    student_id: str
    full_name: str
    descriptors: List[List[float]] = Field(..., description="1 hoac nhieu vector 128-D lay tu face-api.js")


class FaceScanRequest(BaseModel):
    bus_id: str
    student_descriptor: List[float] = Field(..., description="Vector 128-D face-api.js tinh o trinh duyet")
    event: str = Field(..., description="boarded hoac alighted")


class StartTripRequest(BaseModel):
    bus_id: str
    expected_students: List[dict]  # [{student_id, full_name}]


# ---------- 2. Driver drowsiness (thong tin frame duoc gui dang base64 qua route rieng) ----------
class DriverFrameMeta(BaseModel):
    driver_id: str
    frame_index: int
    timestamp: float


# ---------- 3. Pulse monitor ----------
class PulseFrameMeta(BaseModel):
    person_id: str


# ---------- 4. Emergency ----------
class EmergencyTriggerRequest(BaseModel):
    incident_type: str
    bus_id: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    custom_message: Optional[str] = None


# ---------- 5. Bus tracking ----------
class BusLocationUpdate(BaseModel):
    bus_id: str
    route_id: str
    lat: float
    lng: float
    traffic_level: float = 1.0


class InitRouteRequest(BaseModel):
    route_id: str
    stops: List[dict]


class PassengerETARequest(BaseModel):
    """Hành khách hỏi: 'Xe bus_id trên tuyến route_id còn bao lâu đến trạm stop_id của tôi?'"""
    route_id: str
    bus_id: str
    passenger_stop_id: int
    traffic_level: float = 1.0


class ETAResponse(BaseModel):
    """Response ETA cho hành khách."""
    bus_id: str
    passenger_stop_id: int
    stop_name: Optional[str] = None
    distance_remaining_km: float
    eta_minutes: float
    bus_distance_from_start_km: float
    stop_cumulative_distance_km: float
    direction: str


# ---------- Frame payloads (anh base64 JPEG/PNG tu camera) ----------
class DriverFramePayload(BaseModel):
    frame_index: int
    timestamp: float
    image_base64: str


class PulseFramePayload(BaseModel):
    image_base64: str
