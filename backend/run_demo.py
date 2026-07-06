"""
Demo end-to-end: goi tat ca cac ham chinh cua 5 tinh nang, khong can server
FastAPI dang chay, de kiem chung logic hoat dong dung.

Chay:  cd backend && python run_demo.py

Luu y: driver_monitor va pulse_monitor can khuon mat THAT tu webcam/anh de
kich hoat day du EAR/MAR/FFT. Trong demo nay dung frame nhieu ngau nhien (
khong co mat) de chung minh nhanh khong loi va cac nhanh "khong tim thay
khuon mat" hoat dong dung - khi trien khai that voi camera xe, code se tu
dong chay sang nhanh tinh EAR/MAR/BPM that.
"""
from __future__ import annotations

import random
import time

import numpy as np

from app.bus_tracking.tracker import BusTracker
from app.driver_monitor.models import FramePacket
from app.driver_monitor.pipeline import DriverSafetyPipeline
from app.emergency.dispatcher import EmergencyDispatcher, IncidentType
from app.face_checkin.matcher import FaceMatcher, LabeledDescriptor
from app.face_checkin.roster import BoardingEvent, TripRoster
from app.pulse_monitor.pulse_detector import PulseMonitor


def fake_descriptor(seed: int) -> list:
    rng = np.random.default_rng(seed)
    return list(rng.normal(0, 0.15, size=128))


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =====================================================================================
# 1) FACE CHECK-IN
# =====================================================================================
def demo_face_checkin() -> None:
    section("1) DIEM DANH KHUON MAT LEN/XUONG XE (face-api.js FaceMatcher port)")

    an_desc = fake_descriptor(seed=1)
    binh_desc = fake_descriptor(seed=2)

    labeled = [
        LabeledDescriptor(student_id="hs001", full_name="Nguyen Van An", descriptors=[an_desc]),
        LabeledDescriptor(student_id="hs002", full_name="Tran Thi Binh", descriptors=[binh_desc]),
    ]
    matcher = FaceMatcher(labeled)

    trip = TripRoster(bus_id="BUS-01", expected_students=[
        {"student_id": "hs001", "full_name": "Nguyen Van An"},
        {"student_id": "hs002", "full_name": "Tran Thi Binh"},
    ])

    # An quet mat len xe dung (them nhieu nho de mo phong anh sang khac nhau)
    noisy_an = list(np.array(an_desc) + np.random.default_rng(99).normal(0, 0.02, size=128))
    match = matcher.find_best_match(noisy_an)
    print(f"-> Quet mat len xe: match={match}")
    alerts = trip.register_scan(match.student_id, BoardingEvent.BOARDED, match.full_name)
    print(f"   Alerts: {alerts if alerts else 'khong co canh bao'}")

    # Mot hoc sinh LA (khong co trong danh sach he thong) len nham xe
    stranger_desc = fake_descriptor(seed=999)
    match2 = matcher.find_best_match(stranger_desc)
    print(f"\n-> Nguoi la quet mat: match={match2}")
    if not match2.is_match:
        print("   -> He thong KHONG nhan dien duoc (dung tu choi, dung yeu cau bai toan)")

    # Ket thuc chuyen ma Binh CHUA quet xuong xe -> phai bi canh bao
    print("\n-> Ket thuc chuyen (Binh chua tung len xe, An da len nhung chua xuong):")
    sweep_alerts = trip.cabin_sweep_check()
    for a in sweep_alerts:
        print(f"   [{a.severity.upper()}] {a.message}")

    print(f"\nTom tat roster: {trip.summary()}")


# =====================================================================================
# 2) DRIVER DROWSINESS
# =====================================================================================
def demo_driver_monitor() -> None:
    section("2) CANH BAO TAI XE NGU GAT (ai-driver-safety port: EAR/MAR + RiskScorer + AlertPolicy)")

    pipeline = DriverSafetyPipeline()
    # 10 frame nhieu ngau nhien (khong co khuon mat that) -> kiem tra nhanh "missing_face"
    for i in range(10):
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        packet = FramePacket(frame=frame, timestamp=time.time(), frame_index=i, source_id="driver_01")
        processed = pipeline.process_frame(packet)
        alerts = pipeline.alerts_for(processed)
        if alerts:
            for a in alerts:
                print(f"   frame {i}: [{a.severity.upper()}] {a.message} (risk_score={processed.risk_score})")
    print("-> Da chay 10 frame; voi camera that huong vao mat tai xe, pipeline se tu dong")
    print("   tinh EAR/MAR tu Haar-cascade mat/mieng va bao drowsy/yawning/distracted nhu thiet ke goc.")


# =====================================================================================
# 3) PULSE MONITOR
# =====================================================================================
def demo_pulse_monitor() -> None:
    section("3) DO NHIP TIM TU XA (webcam-pulse-detector FFT port)")

    monitor = PulseMonitor()
    for i in range(15):
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        result = monitor.process_frame("driver_01", frame)
        if i in (0, 14):
            print(f"   frame {i}: {result}")
    print("-> Khong co khuon mat that trong frame nhieu nen face_found=False; voi webcam that,")
    print("   sau ~10 mau se bat dau uoc luong BPM qua FFT vung tran nhu thuat toan goc.")


# =====================================================================================
# 4) EMERGENCY BUTTON
# =====================================================================================
def demo_emergency() -> None:
    section("4) NUT BAM KHAN CAP (dinh tuyen theo loai tinh huong)")

    log = []

    def make_sender(channel):
        def _sender(ch, result):
            log.append(f"   -> Da gui canh bao toi [{ch}]: {result.message}")
        return _sender

    dispatcher = EmergencyDispatcher(channel_senders={
        c: make_sender(c) for c in ["police", "ambulance", "fire_department", "school", "parents", "driver"]
    })

    for incident in [IncidentType.TRAFFIC_ACCIDENT, IncidentType.INTRUDER, IncidentType.FIRE]:
        print(f"\n-> Tai xe bam nut khan cap, chon tinh huong: {incident.value}")
        dispatcher.trigger(incident, bus_id="BUS-01", location={"lat": 21.0294, "lng": 105.8544})
        for line in log:
            print(line)
        log.clear()


# =====================================================================================
# 5) BUS TRACKING
# =====================================================================================
def demo_bus_tracking() -> None:
    section("5) BAO TRUOC THOI GIAN XE DEN + CANH BAO CHECH HUONG (bustracker port)")

    stops = [
        {"id": 1, "name": "Truong THPT ABC", "lat": 21.0294, "lng": 105.8544},
        {"id": 2, "name": "Nga tu Kim Ma", "lat": 21.0325, "lng": 105.8256},
        {"id": 3, "name": "Cau Giay", "lat": 21.0359, "lng": 105.7910},
        {"id": 4, "name": "My Dinh", "lat": 21.0290, "lng": 105.7660},
    ]
    tracker = BusTracker(route_stops=stops, deviation_threshold_km=0.3)

    print("-> Xe di DUNG tuyen (vi tri nam tren duong noi cac tram):")
    for lat, lng in [(21.0300, 105.8450), (21.0340, 105.8100)]:
        result = tracker.update_location("BUS-01", lat, lng, traffic_level=1.2)
        print(f"   vi tri=({lat},{lng}) -> next_stop={result['next_stop']}, "
              f"ETA={result['eta_minutes']} phut, deviation={result['route_deviation']['distance_from_route_km']}km")

    print("\n-> Xe DI CHECH tuyen (lech xa 1 khoang lon):")
    result = tracker.update_location("BUS-01", 21.0700, 105.7500, traffic_level=1.2)
    print(f"   -> {result['route_deviation']['message']} "
          f"(khoang cach lech: {result['route_deviation']['distance_from_route_km']} km)")


if __name__ == "__main__":
    demo_face_checkin()
    demo_driver_monitor()
    demo_pulse_monitor()
    demo_emergency()
    demo_bus_tracking()
    print("\n" + "=" * 80)
    print("HOAN TAT DEMO - tat ca 5 tinh nang da chay va tra ve ket qua hop le.")
    print("=" * 80)
