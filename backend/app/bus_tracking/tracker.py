"""
tracker.py – BusTracker hoàn chỉnh: ETA + cảnh báo chệch hướng.

Port & mở rộng từ Terrificdatabytes/bustracker/app.py:
  - detect_bus_direction         → _detect_direction()
  - find_next_stop_bidirectional → _find_next_stop()
  - calculate_speed_from_history → _calculate_speed()
  - predict_eta                  → dùng tốc độ thực + traffic_level
  - precalculate_stop_distances  → build_stop_distance_cache() lúc __init__

Tính năng bổ sung (không có trong repo gốc):
  - detect_route_deviation       → cảnh báo chệch tuyến
  - Tính khoảng cách còn lại đến trạm từ cache O(1)
  - ETA dùng tốc độ thực tế đo được (không hardcode 30 km/h)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.bus_tracking.geo import (
    build_stop_distance_cache,
    calculate_bus_distance_from_start,
    detect_route_deviation,
    find_nearest_stop_with_cache,
    get_stop_distance,
    get_total_distance,
    haversine_distance,
)
from app.bus_tracking.route_data import get_segment_distances


class BusTracker:
    """
    Theo dõi một tuyến xe (route) gồm nhiều xe buýt (bus_id).

    Mỗi instance giữ:
      - route_stops      : danh sách trạm dừng theo đúng thứ tự tuyến
      - stop_dist_cache  : khoảng cách tích luỹ forward/backward cho từng trạm
      - Lịch sử vị trí / tốc độ / hướng đi của từng xe
      - ML model (tuỳ chọn) để tính ETA chính xác hơn
    """

    def __init__(
        self,
        route_stops: List[dict],
        route_id: str = "unknown",
        deviation_threshold_km: float = 0.3,
    ) -> None:
        """
        Tham số:
          route_stops          : [{'id':1,'name':...,'lat':...,'lng':...}, ...]
          route_id             : ID tuyến (dùng để tra segment distances)
          deviation_threshold_km: ngưỡng cảnh báo chệch tuyến (km), mặc định 300m
        """
        self.route_stops = route_stops
        self.route_id = route_id
        self.deviation_threshold_km = deviation_threshold_km

        # Pre-calculate stop distance cache lúc khởi tạo
        segment_distances = get_segment_distances(route_id)
        self.stop_dist_cache = build_stop_distance_cache(
            route_id=route_id,
            route_stops=route_stops,
            segment_distances=segment_distances,
        )
        self.total_route_distance_km = get_total_distance(self.stop_dist_cache)

        # State per bus_id
        self._speed_history: Dict[str, List[dict]] = defaultdict(list)
        self._position_history: Dict[str, List[dict]] = defaultdict(list)
        self._direction: Dict[str, str] = defaultdict(lambda: "forward")
        self._last_known_speed: Dict[str, float] = defaultdict(lambda: 0.0)
        self._bus_distance_from_start: Dict[str, float] = defaultdict(lambda: 0.0)

    # =======================================================================
    # Private helpers – port từ bustracker/app.py
    # =======================================================================

    def _calculate_speed(
        self, bus_id: str, lat: float, lng: float, now: datetime
    ) -> float:
        """
        Port từ calculate_speed_from_history (bustracker/app.py).
        Dùng 5 vị trí gần nhất để tính tốc độ trung bình (km/h).
        """
        history = self._speed_history[bus_id]
        history.append({"lat": lat, "lng": lng, "time": now})
        if len(history) > 5:
            history.pop(0)
        if len(history) < 2:
            return self._last_known_speed[bus_id]

        oldest, newest = history[0], history[-1]
        time_diff_s = (newest["time"] - oldest["time"]).total_seconds()
        if time_diff_s <= 0.1:
            return self._last_known_speed[bus_id]

        # Dùng cache trạm để tính khoảng cách "theo đường" chính xác hơn
        dist_km = abs(
            calculate_bus_distance_from_start(
                newest["lat"], newest["lng"],
                self.route_stops, self.stop_dist_cache, self._direction[bus_id],
            )
            - calculate_bus_distance_from_start(
                oldest["lat"], oldest["lng"],
                self.route_stops, self.stop_dist_cache, self._direction[bus_id],
            )
        )

        speed = (dist_km / time_diff_s) * 3600
        speed = min(max(speed, 0.0), 120.0)   # clamp 0–120 km/h
        self._last_known_speed[bus_id] = speed
        return round(speed, 1)

    def _detect_direction(self, bus_id: str, lat: float, lng: float) -> str:
        """
        Port từ detect_bus_direction (bustracker/app.py).
        Dùng 5 vị trí lịch sử để xác định chiều đi forward/backward.
        """
        if len(self.route_stops) < 2:
            return "forward"

        history = self._position_history[bus_id]
        history.append({"lat": lat, "lng": lng})
        if len(history) > 5:
            history.pop(0)

        # Chưa đủ lịch sử → ước lượng từ khoảng cách đến 2 đầu tuyến
        if len(history) < 3:
            first, last = self.route_stops[0], self.route_stops[-1]
            d_first = haversine_distance(lat, lng, first["lat"], first["lng"])
            d_last = haversine_distance(lat, lng, last["lat"], last["lng"])
            return "forward" if d_first < d_last else "backward"

        # Ánh xạ mỗi vị trí lịch sử → index trạm gần nhất
        indices = []
        for pos in history:
            _, idx, _ = find_nearest_stop_with_cache(pos["lat"], pos["lng"], self.route_stops)
            indices.append(idx)

        if indices[-1] > indices[0]:
            return "forward"
        if indices[-1] < indices[0]:
            return "backward"
        return self._direction[bus_id]  # giữ nguyên nếu không thay đổi

    def _find_next_stop(
        self, bus_id: str, lat: float, lng: float, direction: str
    ) -> Tuple[Optional[dict], float]:
        """
        Port từ find_next_stop_bidirectional (bustracker/app.py).

        Trả về (next_stop_dict, distance_remaining_km).
        distance_remaining_km = stop_cumulative - bus_cumulative (từ cache).
        """
        nearest, nearest_idx, dist_to_nearest = find_nearest_stop_with_cache(
            lat, lng, self.route_stops
        )

        if nearest is None:
            return None, 0.0

        ARRIVE_THRESHOLD_KM = 0.1  # trong bán kính 100m = đang ở tại trạm

        if dist_to_nearest < ARRIVE_THRESHOLD_KM:
            # Xe đang đứng tại trạm → trạm tiếp theo
            if direction == "forward" and nearest_idx < len(self.route_stops) - 1:
                next_stop = self.route_stops[nearest_idx + 1]
            elif direction == "backward" and nearest_idx > 0:
                next_stop = self.route_stops[nearest_idx - 1]
            else:
                next_stop = nearest
        else:
            next_stop = nearest

        # Tính distance_remaining từ cache (chính xác hơn haversine thẳng)
        bus_dist = calculate_bus_distance_from_start(
            lat, lng, self.route_stops, self.stop_dist_cache, direction
        )
        stop_dist = get_stop_distance(self.stop_dist_cache, next_stop["id"], direction)

        if stop_dist is not None:
            distance_remaining = max(0.0, stop_dist - bus_dist)
        else:
            # Fallback haversine nếu không có cache
            distance_remaining = haversine_distance(lat, lng, next_stop["lat"], next_stop["lng"])

        return next_stop, round(distance_remaining, 3)

    # =======================================================================
    # Internal – tính ETA từ tốc độ thực
    # =======================================================================
    @staticmethod
    def _predict_eta(distance_km: float, speed_kmh: float, traffic_level: float) -> float:
        """Tính ETA (phút) từ khoảng cách, tốc độ thực và hệ số kẹt xe."""
        effective_speed = max(speed_kmh, 5.0) / max(traffic_level, 0.1)
        effective_speed = max(effective_speed, 5.0)  # tối thiểu 5 km/h
        return round((distance_km / effective_speed) * 60.0, 1)

    # =======================================================================
    # Public API – update_location
    # =======================================================================
    def update_location(
        self,
        bus_id: str,
        lat: float,
        lng: float,
        traffic_level: float = 1.0,
    ) -> dict:
        """
        Hàm gọi chính: nhận vị trí GPS mới nhất của 1 xe buýt.

        Trả về dict gồm:
          bus_id, lat, lng
          speed_kmh                    : tốc độ thực (km/h)
          direction                    : "forward" | "backward"
          next_stop                    : tên trạm tiếp theo
          next_stop_id                 : id trạm tiếp theo
          distance_to_next_stop_km     : khoảng cách còn lại (km)
          eta_minutes                  : thời gian đến trạm tiếp theo (phút)
          distance_from_route_start_km : xe đã đi được bao nhiêu km từ đầu tuyến
          total_route_distance_km      : tổng chiều dài tuyến
          route_deviation              : dict cảnh báo chệch tuyến
          timestamp                    : ISO string
        """
        now = datetime.now()

        speed_kmh = self._calculate_speed(bus_id, lat, lng, now)
        direction = self._detect_direction(bus_id, lat, lng)
        self._direction[bus_id] = direction

        next_stop, distance_remaining_km = self._find_next_stop(bus_id, lat, lng, direction)

        # ETA dựa trên tốc độ thực đo được
        effective_speed = speed_kmh if speed_kmh > 2.0 else 30.0
        eta_minutes = self._predict_eta(distance_remaining_km, effective_speed, traffic_level)

        # Vị trí tích luỹ xe từ đầu tuyến
        bus_dist_from_start = calculate_bus_distance_from_start(
            lat, lng, self.route_stops, self.stop_dist_cache, direction
        )
        self._bus_distance_from_start[bus_id] = bus_dist_from_start

        # Cảnh báo chệch tuyến
        deviation = detect_route_deviation(
            current_pos=(lat, lng),
            route_points=self.route_stops,
            deviation_threshold_km=self.deviation_threshold_km,
        )

        return {
            "bus_id": bus_id,
            "lat": lat,
            "lng": lng,
            "speed_kmh": round(speed_kmh, 1),
            "direction": direction,
            "next_stop": next_stop["name"] if next_stop else None,
            "next_stop_id": next_stop["id"] if next_stop else None,
            "distance_to_next_stop_km": round(distance_remaining_km, 3),
            "eta_minutes": eta_minutes,
            "distance_from_route_start_km": round(bus_dist_from_start, 3),
            "total_route_distance_km": self.total_route_distance_km,
            "route_deviation": deviation,
            "timestamp": now.isoformat(),
        }

    def get_passenger_eta(
        self,
        bus_id: str,
        passenger_stop_id: int,
        traffic_level: float = 1.0,
    ) -> dict:
        """
        Tính ETA từ vị trí xe hiện tại đến trạm mà hành khách muốn xuống.

        Công thức (port từ bustracker):
          remaining = stop_cache[passenger_stop_id] - bus_distance_from_start

        Trả về:
          {
            bus_id, passenger_stop_id, stop_name,
            distance_remaining_km,
            eta_minutes,
            bus_distance_from_start_km,
            stop_cumulative_distance_km,
            direction,
          }
        """
        direction = self._direction[bus_id]
        bus_dist = self._bus_distance_from_start.get(bus_id, 0.0)

        stop_entry = self.stop_dist_cache.get(passenger_stop_id)
        if stop_entry is None:
            return {
                "error": f"Không tìm thấy trạm id={passenger_stop_id} trong cache",
                "bus_id": bus_id,
                "passenger_stop_id": passenger_stop_id,
            }

        stop_cum_dist = stop_entry.get(direction, 0.0)
        stop_name = stop_entry.get("name", f"Trạm {passenger_stop_id}")

        remaining_km = max(0.0, stop_cum_dist - bus_dist)

        effective_speed = self._last_known_speed.get(bus_id, 0.0)
        if effective_speed < 2.0:
            effective_speed = 30.0

        eta_minutes = self._predict_eta(remaining_km, effective_speed, traffic_level)

        return {
            "bus_id": bus_id,
            "passenger_stop_id": passenger_stop_id,
            "stop_name": stop_name,
            "distance_remaining_km": round(remaining_km, 3),
            "eta_minutes": eta_minutes,
            "bus_distance_from_start_km": round(bus_dist, 3),
            "stop_cumulative_distance_km": round(stop_cum_dist, 3),
            "direction": direction,
        }

    def get_all_buses_status(self) -> List[dict]:
        """Trả về trạng thái ngắn gọn của tất cả xe đang hoạt động trên tuyến."""
        result = []
        for bus_id, dist in self._bus_distance_from_start.items():
            result.append({
                "bus_id": bus_id,
                "distance_from_start_km": round(dist, 3),
                "direction": self._direction[bus_id],
                "speed_kmh": round(self._last_known_speed.get(bus_id, 0.0), 1),
            })
        return result
