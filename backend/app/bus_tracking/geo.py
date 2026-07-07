"""
geo.py – Các hàm hình học / khoảng cách địa lý.

Port từ: Terrificdatabytes/bustracker/app.py
  - haversine_distance
  - calculate_distance_with_waypoints

Bổ sung (không có trong repo gốc):
  - build_stop_distance_cache      : pre-calculate khoảng cách tích luỹ lúc startup
  - get_stop_distance              : tra cứu O(1) từ cache
  - distance_point_to_segment_km   : khoảng cách điểm → đoạn thẳng (xấp xỉ phẳng)
  - detect_route_deviation         : cảnh báo chệch tuyến
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Coord = Tuple[float, float]  # (lat, lng)

# Kiểu cache: route_id → stop_id → direction → km tích luỹ
StopDistanceCache = Dict[str, Dict[int, Dict[str, float]]]


# ===========================================================================
# 1. Haversine – port nguyên từ bustracker/app.py
# ===========================================================================
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Trả về khoảng cách Haversine (km) giữa 2 điểm GPS."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ===========================================================================
# 2. Khoảng cách theo waypoints – port từ bustracker/app.py
# ===========================================================================
def calculate_distance_with_waypoints(
    route_points: List[dict],
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Ước lượng khoảng cách "theo đường" giữa 2 điểm GPS dọc theo tuyến.
    Tìm waypoint gần nhất với mỗi điểm rồi cộng dồn haversine giữa các waypoints.
    """
    if not route_points:
        return haversine_distance(lat1, lon1, lat2, lon2)

    def _closest_idx(lat: float, lng: float) -> Tuple[int, float]:
        best_i, best_d = 0, float("inf")
        for i, p in enumerate(route_points):
            d = haversine_distance(lat, lng, p["lat"], p["lng"])
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    start_idx, dist_start = _closest_idx(lat1, lon1)
    end_idx, dist_end = _closest_idx(lat2, lon2)

    total = dist_start
    step = 1 if start_idx < end_idx else -1
    for i in range(start_idx, end_idx, step):
        p1, p2 = route_points[i], route_points[i + step]
        total += haversine_distance(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
    if start_idx != end_idx:
        total += dist_end
    return total


# ===========================================================================
# 3. Stop distance cache – port & mở rộng từ bustracker/app.py
#    (precalculate_stop_distances_manual + stop_distance_cache)
# ===========================================================================
def build_stop_distance_cache(
    route_id: str,
    route_stops: List[dict],
    segment_distances: Optional[List[float]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Pre-calculate khoảng cách tích luỹ forward + backward cho mỗi trạm.

    Tham số:
      route_id         : ID tuyến (dùng để log)
      route_stops      : list dict {'id','name','lat','lng'} theo đúng thứ tự tuyến
      segment_distances: list khoảng cách (km) giữa các trạm kế tiếp
                         (len = num_stops - 1). Nếu None → dùng haversine fallback.

    Trả về:
      dict: stop_id (int) → {'forward': float, 'backward': float}
      Thêm key đặc biệt:  0 → {'total_distance': float}
    """
    if not route_stops:
        return {}

    n = len(route_stops)

    # Xác định khoảng cách từng đoạn
    use_ai_distances = (
        segment_distances is not None
        and len(segment_distances) == n - 1
    )

    if use_ai_distances:
        print(f"[RULER] Route {route_id}: dùng AI-calculated segment distances ({n - 1} đoạn)")
        segs = segment_distances
    else:
        if segment_distances is not None:
            print(
                f"[WARN] Route {route_id}: segment_distances có {len(segment_distances)} đoạn, "
                f"cần {n - 1} → dùng haversine fallback"
            )
        else:
            print(f"[RULER] Route {route_id}: không có segment distances → dùng haversine fallback")
        segs = [
            haversine_distance(
                route_stops[i]["lat"], route_stops[i]["lng"],
                route_stops[i + 1]["lat"], route_stops[i + 1]["lng"],
            )
            for i in range(n - 1)
        ]

    # Tính cumulative forward
    cum = 0.0
    forward: Dict[int, float] = {}
    for i, stop in enumerate(route_stops):
        if i > 0:
            cum += segs[i - 1]
        forward[stop["id"]] = round(cum, 4)

    total_distance = cum

    # Tính backward
    backward: Dict[int, float] = {}
    for stop in route_stops:
        backward[stop["id"]] = round(total_distance - forward[stop["id"]], 4)

    # Gom thành cache per stop_id
    cache: Dict[int, Dict[str, float]] = {}
    for stop in route_stops:
        sid = stop["id"]
        cache[sid] = {
            "forward": forward[sid],
            "backward": backward[sid],
            "name": stop.get("name", f"Stop {sid}"),
        }

    # Key 0 để lưu tổng khoảng cách tuyến
    cache[0] = {"total_distance": round(total_distance, 4)}

    print(
        f"[OK] Route {route_id}: cache {n} trạm, "
        f"tổng {total_distance:.3f} km "
        f"({'AI segments' if use_ai_distances else 'haversine'})"
    )
    return cache


def get_stop_distance(
    cache: Dict[int, Dict[str, float]],
    stop_id: int,
    direction: str = "forward",
) -> Optional[float]:
    """
    Tra cứu O(1) khoảng cách tích luỹ của trạm từ cache.
    Trả về None nếu không tìm thấy.
    """
    entry = cache.get(stop_id)
    if entry is None:
        return None
    return entry.get(direction)


def get_total_distance(cache: Dict[int, Dict[str, float]]) -> float:
    """Tổng khoảng cách toàn tuyến (km), lấy từ key 0 trong cache."""
    return cache.get(0, {}).get("total_distance", 0.0)


def find_nearest_stop_with_cache(
    lat: float,
    lng: float,
    route_stops: List[dict],
) -> Tuple[Optional[dict], int, float]:
    """
    Tìm trạm gần nhất và index của nó.
    Trả về (stop_dict, index, haversine_distance_km).
    """
    min_d, nearest, nearest_idx = float("inf"), None, 0
    for idx, stop in enumerate(route_stops):
        d = haversine_distance(lat, lng, stop["lat"], stop["lng"])
        if d < min_d:
            min_d, nearest, nearest_idx = d, stop, idx
    return nearest, nearest_idx, min_d


def calculate_bus_distance_from_start(
    lat: float,
    lng: float,
    route_stops: List[dict],
    cache: Dict[int, Dict[str, float]],
    direction: str = "forward",
) -> float:
    """
    Tính khoảng cách xe buýt đã đi từ điểm đầu tuyến (km).
    Dùng cache trạm gần nhất + offset haversine.

    Công thức:
      bus_distance = stop_cache[nearest_stop][direction] ± offset_to_nearest_stop
    """
    if not route_stops or not cache:
        return 0.0

    nearest, nearest_idx, dist_to_nearest = find_nearest_stop_with_cache(lat, lng, route_stops)
    if nearest is None:
        return 0.0

    stop_cum = get_stop_distance(cache, nearest["id"], direction)
    if stop_cum is None:
        return 0.0

    # Ước lượng xe đang ở giữa 2 trạm → cộng thêm offset
    if direction == "forward":
        return max(0.0, stop_cum - dist_to_nearest)
    else:
        return max(0.0, stop_cum - dist_to_nearest)


# ===========================================================================
# 4. Khoảng cách điểm → đoạn thẳng (xấp xỉ phẳng)
# ===========================================================================
def distance_point_to_segment_km(point: Coord, seg_a: Coord, seg_b: Coord) -> float:
    """
    Khoảng cách xấp xỉ (km) từ 1 điểm GPS đến đoạn [seg_a, seg_b].
    Dùng xấp xỉ phẳng cục bộ – đủ chính xác trong phạm vi vài km.
    """
    lat0 = math.radians((seg_a[0] + seg_b[0]) / 2.0)
    kx = 111.320 * math.cos(lat0)  # km / độ kinh
    ky = 110.574                    # km / độ vĩ

    px, py = point[1] * kx, point[0] * ky
    ax, ay = seg_a[1] * kx, seg_a[0] * ky
    bx, by = seg_b[1] * kx, seg_b[0] * ky

    abx, aby = bx - ax, by - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq == 0:
        return haversine_distance(point[0], point[1], seg_a[0], seg_a[1])

    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


# ===========================================================================
# 5. Cảnh báo chệch tuyến – mở rộng từ phiên bản cũ
# ===========================================================================
def detect_route_deviation(
    current_pos: Coord,
    route_points: List[dict],
    deviation_threshold_km: float = 0.3,
) -> dict:
    """
    Phát hiện chệch tuyến: tìm đoạn đường gần nhất trong tuyến và so với ngưỡng.

    Tham số:
      current_pos          : (lat, lng) vị trí hiện tại xe
      route_points         : list dict {'id','name','lat','lng'} theo thứ tự tuyến
      deviation_threshold_km: ngưỡng cảnh báo (km), mặc định 300m

    Trả về:
      {
        deviated             : bool
        distance_from_route_km: float
        threshold_km         : float
        nearest_segment_idx  : int   (index đoạn gần nhất)
        nearest_stop_before  : str   (tên trạm trước đoạn gần nhất)
        nearest_stop_after   : str   (tên trạm sau đoạn gần nhất)
        message              : str   (tiếng Việt)
        severity             : str   ("ok" | "warning" | "critical")
      }
    """
    if len(route_points) < 2:
        return {
            "deviated": False,
            "distance_from_route_km": 0.0,
            "threshold_km": deviation_threshold_km,
            "nearest_segment_idx": 0,
            "nearest_stop_before": "",
            "nearest_stop_after": "",
            "message": "Không đủ dữ liệu tuyến để kiểm tra",
            "severity": "ok",
        }

    best_dist = float("inf")
    best_seg_idx = 0

    for i in range(len(route_points) - 1):
        a = (route_points[i]["lat"], route_points[i]["lng"])
        b = (route_points[i + 1]["lat"], route_points[i + 1]["lng"])
        d = distance_point_to_segment_km(current_pos, a, b)
        if d < best_dist:
            best_dist = d
            best_seg_idx = i

    deviated = best_dist > deviation_threshold_km

    # Xác định severity
    if not deviated:
        severity = "ok"
    elif best_dist < deviation_threshold_km * 2:
        severity = "warning"   # 1x–2x ngưỡng
    else:
        severity = "critical"  # > 2x ngưỡng

    stop_before = route_points[best_seg_idx].get("name", f"Trạm {best_seg_idx + 1}")
    stop_after = route_points[best_seg_idx + 1].get("name", f"Trạm {best_seg_idx + 2}")

    if deviated:
        message = (
            f"CẢNH BÁO: Xe đã đi chệch tuyến khoảng {best_dist * 1000:.0f} m "
            f"(giữa '{stop_before}' và '{stop_after}')"
        )
    else:
        message = f"Xe đang đi đúng tuyến (giữa '{stop_before}' và '{stop_after}')"

    return {
        "deviated": deviated,
        "distance_from_route_km": round(best_dist, 4),
        "threshold_km": deviation_threshold_km,
        "nearest_segment_idx": best_seg_idx,
        "nearest_stop_before": stop_before,
        "nearest_stop_after": stop_after,
        "message": message,
        "severity": severity,
    }
