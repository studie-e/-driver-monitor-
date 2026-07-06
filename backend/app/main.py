"""
Ung dung FastAPI trung tam - ghep 5 tinh nang thanh 1 he thong duy nhat:

  1. /api/checkin/*     - Diem danh khuon mat khi len/xuong xe (face-api.js + FaceMatcher + TripRoster)
  2. /api/driver/*       - Canh bao tai xe ngu gat (ai-driver-safety port)
  3. /api/pulse/*        - Do nhip tim tu xa (webcam-pulse-detector port)
  4. /api/emergency/*    - Nut bam khan cap (EmergencyDispatcher)
  5. /api/tracking/*     - ETA + canh bao chech huong (bustracker port)

Chay: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import base64
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.bus_tracking.tracker import BusTracker
from app.driver_monitor.models import FramePacket
from app.driver_monitor.pipeline import DriverSafetyPipeline
from app.emergency.dispatcher import EmergencyDispatcher, IncidentType
from app.face_checkin.matcher import FaceMatcher, LabeledDescriptor
from app.face_checkin.roster import BoardingEvent, TripRoster
from app.models.schemas import (
    BusLocationUpdate,
    DriverFramePayload,
    EmergencyTriggerRequest,
    FaceScanRequest,
    InitRouteRequest,
    PulseFramePayload,
    RegisterStudentRequest,
    StartTripRequest,
)
from app.pulse_monitor.pulse_detector import PulseMonitor
from app.websocket_manager import manager

app = FastAPI(title="He thong dam bao an toan xe dua don hoc sinh bang AI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------------- Trang thai toan cuc (demo, thay bang DB o production) -------------------------
_labeled_descriptors: Dict[str, LabeledDescriptor] = {}       # student_id -> LabeledDescriptor
_active_trips: Dict[str, TripRoster] = {}                     # bus_id -> TripRoster
_driver_pipelines: Dict[str, DriverSafetyPipeline] = {}       # driver_id -> pipeline
_pulse_monitor = PulseMonitor()                               # 1 instance dung chung, tach theo person_id
_bus_trackers: Dict[str, BusTracker] = {}                     # route_id -> BusTracker
_emergency_dispatcher = EmergencyDispatcher(channel_senders={
    # Demo: log ra console. Production: cam Twilio SMS / FCM push / email SMTP tai day.
    channel: (lambda ch, result: print(f"[DISPATCH -> {ch}] {result.message} (bus={result.bus_id})"))
    for channel in ["police", "ambulance", "fire_department", "school", "parents", "driver"]
})


# =====================================================================================
# 1) FACE CHECK-IN  (face-api.js chay o browser -> gui descriptor 128-D ve day)
# =====================================================================================
@app.post("/api/checkin/register_student")
def register_student(req: RegisterStudentRequest):
    """Dang ky khuon mat 1 hoc sinh (thuc hien 1 lan, luu nhieu anh mau)."""
    entry = _labeled_descriptors.get(req.student_id)
    if entry is None:
        entry = LabeledDescriptor(student_id=req.student_id, full_name=req.full_name, descriptors=[])
        _labeled_descriptors[req.student_id] = entry
    entry.descriptors.extend(req.descriptors)
    return {"status": "ok", "student_id": req.student_id, "num_samples": len(entry.descriptors)}


@app.post("/api/checkin/start_trip")
def start_trip(req: StartTripRequest):
    """Tai xe bam 'bat dau chuyen' - nap danh sach hoc sinh du kien len xe nay."""
    _active_trips[req.bus_id] = TripRoster(bus_id=req.bus_id, expected_students=req.expected_students)
    return {"status": "ok", "bus_id": req.bus_id, "roster": _active_trips[req.bus_id].summary()}


@app.post("/api/checkin/scan")
async def face_scan(req: FaceScanRequest):
    """
    Ham goi chinh khi camera cua xe bat duoc 1 khuon mat luc len/xuong xe.
    1. FaceMatcher doi chieu descriptor -> tim hoc sinh
    2. TripRoster cap nhat trang thai + phat canh bao neu len nham xe / bat thuong
    3. Day canh bao ngay qua WebSocket toi dashboard tai xe (room = bus_id)
    """
    trip = _active_trips.get(req.bus_id)
    if trip is None:
        raise HTTPException(404, f"Chua co chuyen dang chay cho xe {req.bus_id}. Goi /start_trip truoc.")

    matcher = FaceMatcher(list(_labeled_descriptors.values()))
    match = matcher.find_best_match(req.student_descriptor)

    if not match.is_match:
        payload = {
            "type": "face_scan_alert",
            "severity": "critical",
            "message": "Khong nhan dien duoc khuon mat nay trong he thong - kiem tra thu cong",
            "distance": match.distance,
        }
        await manager.broadcast(req.bus_id, payload)
        return payload

    event = BoardingEvent.BOARDED if req.event == "boarded" else BoardingEvent.ALIGHTED
    alerts = trip.register_scan(match.student_id, event, match.full_name)

    payload = {
        "type": "face_scan_result",
        "student_id": match.student_id,
        "full_name": match.full_name,
        "event": req.event,
        "distance": match.distance,
        "alerts": [a.__dict__ for a in alerts],
        "roster_summary": trip.summary(),
    }
    await manager.broadcast(req.bus_id, payload)

    # Neu co canh bao critical (vd len nham xe) -> tu dong kich hoat emergency toi nha truong/phu huynh
    for a in alerts:
        if a.severity == "critical":
            _emergency_dispatcher.trigger(
                IncidentType.INTRUDER if a.code == "wrong_bus" else IncidentType.CHILD_LEFT_ON_BUS,
                bus_id=req.bus_id,
                custom_message=a.message,
            )
    return payload


@app.post("/api/checkin/end_trip/{bus_id}")
async def end_trip(bus_id: str):
    """
    Ham goi khi tai xe bam 'ket thuc chuyen': quet lai khoang xe (cabin sweep).
    Neu con hoc sinh 'on_bus' -> canh bao KHAN CAP toi tai xe + nha truong + phu huynh.
    """
    trip = _active_trips.get(bus_id)
    if trip is None:
        raise HTTPException(404, "Khong tim thay chuyen dang chay")

    sweep_alerts = trip.cabin_sweep_check()
    payload = {"type": "cabin_sweep_result", "alerts": [a.__dict__ for a in sweep_alerts],
               "roster_summary": trip.summary()}
    await manager.broadcast(bus_id, payload)

    for a in sweep_alerts:
        _emergency_dispatcher.trigger(IncidentType.CHILD_LEFT_ON_BUS, bus_id=bus_id, custom_message=a.message)

    return payload


# =====================================================================================
# 2) DRIVER DROWSINESS  (anh gui dang base64 JPEG tu camera huong vao tai xe)
# =====================================================================================
def _decode_base64_image(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str.split(",")[-1])
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Khong doc duoc anh base64")
    return frame


@app.post("/api/driver/{driver_id}/frame")
async def driver_frame(driver_id: str, payload: DriverFramePayload):
    """Nhan 1 frame webcam tai xe, tra ve risk_score + alerts (drowsy/yawn/distracted)."""
    if driver_id not in _driver_pipelines:
        _driver_pipelines[driver_id] = DriverSafetyPipeline()
    pipeline = _driver_pipelines[driver_id]

    frame = _decode_base64_image(payload.image_base64)
    packet = FramePacket(frame=frame, timestamp=payload.timestamp, frame_index=payload.frame_index, source_id=driver_id)
    processed = pipeline.process_frame(packet)
    alerts = pipeline.alerts_for(processed)

    payload = {
        "type": "driver_status",
        "driver_id": driver_id,
        "state": processed.state.value,
        "risk_score": processed.risk_score,
        "signals": processed.signals,
        "alerts": [a.__dict__ for a in alerts],
    }
    await manager.broadcast(f"driver_{driver_id}", payload)

    for a in alerts:
        if a.severity == "critical":
            _emergency_dispatcher.trigger(
                IncidentType.MEDICAL, bus_id=driver_id,
                custom_message=f"Tai xe {driver_id}: {a.message}",
            )
    return payload


# =====================================================================================
# 3) PULSE MONITOR  (dung chung cho ca tai xe va hoc sinh, phan biet bang person_id)
# =====================================================================================
@app.post("/api/pulse/{person_id}/frame")
async def pulse_frame(person_id: str, payload: PulseFramePayload):
    frame = _decode_base64_image(payload.image_base64)
    result = _pulse_monitor.process_frame(person_id, frame)
    await manager.broadcast(f"pulse_{person_id}", {"type": "pulse_status", **result})

    if result.get("alert"):
        _emergency_dispatcher.trigger(
            IncidentType.MEDICAL, bus_id=person_id,
            custom_message=result.get("message", "Nhip tim bat thuong"),
        )
    return result


# =====================================================================================
# 4) EMERGENCY BUTTON
# =====================================================================================
@app.post("/api/emergency/trigger")
async def emergency_trigger(req: EmergencyTriggerRequest):
    try:
        incident = IncidentType(req.incident_type)
    except ValueError:
        raise HTTPException(400, f"incident_type khong hop le: {req.incident_type}")

    location = {"lat": req.lat, "lng": req.lng} if req.lat is not None else None
    result = _emergency_dispatcher.trigger(incident, req.bus_id, location, req.custom_message)
    await manager.broadcast(req.bus_id, {"type": "emergency_dispatch", **result.__dict__})
    return result


# =====================================================================================
# 5) BUS TRACKING - ETA + canh bao chech huong
# =====================================================================================
@app.post("/api/tracking/init_route")
def init_route(req: InitRouteRequest):
    """req.stops: [{'id':1,'name':...,'lat':...,'lng':...}, ...] theo dung thu tu tuyen."""
    _bus_trackers[req.route_id] = BusTracker(route_stops=req.stops)
    return {"status": "ok", "route_id": req.route_id, "num_stops": len(req.stops)}


@app.post("/api/tracking/update_location")
async def update_location(update: BusLocationUpdate):
    tracker = _bus_trackers.get(update.route_id)
    if tracker is None:
        raise HTTPException(404, f"Tuyen {update.route_id} chua duoc khoi tao. Goi /init_route truoc.")

    result = tracker.update_location(update.bus_id, update.lat, update.lng, update.traffic_level)
    await manager.broadcast(update.route_id, {"type": "bus_location_update", **result})

    if result["route_deviation"]["deviated"]:
        _emergency_dispatcher.trigger(
            IncidentType.TRAFFIC_ACCIDENT, bus_id=update.bus_id,
            location={"lat": update.lat, "lng": update.lng},
            custom_message=result["route_deviation"]["message"],
        )
    return result


# =====================================================================================
# WEBSOCKET - dashboard tai xe / nha truong / phu huynh lang nghe theo room
# =====================================================================================
@app.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    await manager.connect(websocket, room)
    try:
        while True:
            await websocket.receive_text()  # giu ket noi song, khong can xu ly input
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)


@app.get("/")
def root():
    return {
        "service": "He thong dam bao an toan xe dua don hoc sinh bang AI",
        "features": [
            "checkin - diem danh khuon mat len/xuong xe",
            "driver - canh bao tai xe ngu gat",
            "pulse - do nhip tim tu xa",
            "emergency - nut bam khan cap",
            "tracking - ETA va canh bao chech huong",
        ],
    }
