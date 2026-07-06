"""
BusTracker - ham goi hoan chinh cho tinh nang "bao truoc thoi gian xe den va
canh bao chech huong".

Cac ham duoi day duoc port/adapt tu app.py (repo: Terrificdatabytes/bustracker):
  - detect_bus_direction        (huong di forward/backward dua vao lich su vi tri)
  - find_next_stop_bidirectional (tim tram ke tiep theo huong di)
  - predict_eta                  (uoc luong ETA, fallback khi khong co model ML)
  - calculate_speed_from_history (toc do tu lich su 5 vi tri gan nhat)

`detect_route_deviation` (module geo.py) duoc goi kem de bao sung tinh nang
canh bao chech huong ma bustracker goc CHUA co san.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import List, Optional

from app.bus_tracking.geo import (
    calculate_distance_with_waypoints,
    detect_route_deviation,
    haversine_distance,
)


class BusTracker:
    """
    Trang thai theo doi 1 tuyen xe (nhieu xe buyt / route_id), giu lich su
    vi tri de tinh toc do + huong di + ETA + phat hien chech tuyen.
    """

    def __init__(self, route_stops: List[dict], deviation_threshold_km: float = 0.3):
        """
        route_stops: danh sach cac tram theo dung thu tu tuyen,
                     vd [{'id':1,'name':'Truong THPT A','lat':...,'lng':...}, ...]
                     (tuong duong ORIGINAL_STOPS / STOP_COORDS trong bustracker/app.py)
        """
        self.route_stops = route_stops
        self.deviation_threshold_km = deviation_threshold_km

        self._position_history: dict = defaultdict(list)   # bus_id -> [{'lat','lng','time'}]
        self._speed_history: dict = defaultdict(list)       # bus_id -> [{'lat','lng','time'}]
        self._direction: dict = defaultdict(lambda: "forward")
        self._last_passed_stop: dict = {}

    # ---------- port tu bustracker/app.py::calculate_speed_from_history ----------
    def _calculate_speed(self, bus_id: str, lat: float, lng: float, now: datetime) -> float:
        history = self._speed_history[bus_id]
        history.append({"lat": lat, "lng": lng, "time": now})
        if len(history) > 5:
            history.pop(0)
        if len(history) < 2:
            return 0.0

        oldest, newest = history[0], history[-1]
        distance_km = calculate_distance_with_waypoints(
            self.route_stops, oldest["lat"], oldest["lng"], newest["lat"], newest["lng"]
        )
        time_diff_s = (newest["time"] - oldest["time"]).total_seconds()
        if time_diff_s > 0.1:
            speed_kmh = (distance_km / time_diff_s) * 3600
            return min(max(speed_kmh, 0), 100)
        return 0.0

    # ---------- port tu bustracker/app.py::detect_bus_direction ----------
    def _detect_direction(self, bus_id: str, lat: float, lng: float) -> str:
        if len(self.route_stops) < 2:
            return "forward"

        history = self._position_history[bus_id]
        history.append({"lat": lat, "lng": lng})
        if len(history) > 5:
            history.pop(0)

        if len(history) < 3:
            first, last = self.route_stops[0], self.route_stops[-1]
            dist_first = haversine_distance(lat, lng, first["lat"], first["lng"])
            dist_last = haversine_distance(lat, lng, last["lat"], last["lng"])
            return "forward" if dist_first < dist_last else "backward"

        indices = []
        for pos in history:
            best_i, best_d = 0, float("inf")
            for i, stop in enumerate(self.route_stops):
                d = haversine_distance(pos["lat"], pos["lng"], stop["lat"], stop["lng"])
                if d < best_d:
                    best_d, best_i = d, i
            indices.append(best_i)

        if indices[-1] > indices[0]:
            return "forward"
        if indices[-1] < indices[0]:
            return "backward"
        return self._direction[bus_id]

    # ---------- port tu bustracker/app.py::find_next_stop_bidirectional (rut gon) ----------
    def _find_next_stop(self, bus_id: str, lat: float, lng: float, direction: str):
        min_d, nearest, nearest_idx = float("inf"), None, 0
        for idx, stop in enumerate(self.route_stops):
            d = haversine_distance(lat, lng, stop["lat"], stop["lng"])
            if d < min_d:
                min_d, nearest, nearest_idx = d, stop, idx

        if min_d < 0.1:  # dang o tai tram (trong ban kinh 100m)
            if direction == "forward" and nearest_idx < len(self.route_stops) - 1:
                nxt = self.route_stops[nearest_idx + 1]
            elif direction == "backward" and nearest_idx > 0:
                nxt = self.route_stops[nearest_idx - 1]
            else:
                nxt = nearest
            return nxt, haversine_distance(lat, lng, nxt["lat"], nxt["lng"])

        return nearest, min_d

    # ---------- port tu bustracker/app.py::predict_eta (fallback path, khong dung model.pkl) ----------
    @staticmethod
    def predict_eta(distance_km: float, traffic_level: float) -> float:
        base_speed = 30.0  # km/h toc do trung binh gia dinh
        speed = base_speed / traffic_level if traffic_level > 0 else base_speed
        return (distance_km / speed) * 60.0  # phut

    def update_location(self, bus_id: str, lat: float, lng: float, traffic_level: float = 1.0) -> dict:
        """
        Ham goi chinh: nhan vi tri GPS moi nhat cua 1 xe buyt, tra ve
        {toc do, huong di, tram ke tiep, ETA (phut), canh bao chech tuyen}.
        """
        now = datetime.now()
        speed_kmh = self._calculate_speed(bus_id, lat, lng, now)
        direction = self._detect_direction(bus_id, lat, lng)
        self._direction[bus_id] = direction

        next_stop, distance_km = self._find_next_stop(bus_id, lat, lng, direction)
        eta_minutes = self.predict_eta(distance_km, traffic_level)

        deviation = detect_route_deviation((lat, lng), self.route_stops, self.deviation_threshold_km)

        return {
            "bus_id": bus_id,
            "lat": lat,
            "lng": lng,
            "speed_kmh": round(speed_kmh, 1),
            "direction": direction,
            "next_stop": next_stop["name"] if next_stop else None,
            "distance_to_next_stop_km": round(distance_km, 3),
            "eta_minutes": round(eta_minutes, 1),
            "route_deviation": deviation,
            "timestamp": now.isoformat(),
        }
