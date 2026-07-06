"""
Cac ham hinh hoc / khoang cach dia ly.

Port tu: app.py (repo: Terrificdatabytes/bustracker)
  - haversine_distance
  - phan logic tinh khoang cach tich luy theo waypoint (calculate_distance_with_waypoints)

Bo sung them `distance_point_to_segment` va `detect_route_deviation` (khong
co san trong repo goc) de phuc vu tinh nang "canh bao chech huong" ma de bai
yeu cau - dung chinh haversine_distance goc de suy ra khoang cach vuong goc
tu vi tri xe toi doan duong (segment) gan nhat trong tuyen.
"""
from __future__ import annotations

import math
from typing import List, Tuple

Coord = Tuple[float, float]  # (lat, lng)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Port nguyen ham tu bustracker/app.py. Tra ve khoang cach (km)."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_distance_with_waypoints(route_points: List[dict], lat1: float, lon1: float,
                                       lat2: float, lon2: float) -> float:
    """
    Port tu bustracker/app.py::calculate_distance_with_waypoints.

    Tim diem gan nhat trong `route_points` (list cac dict {'lat','lng'}) ung
    voi (lat1,lon1) va (lat2,lon2), roi cong don khoang cach haversine giua
    cac waypoint nam giua 2 diem do -> uoc luong khoang cach "theo duong"
    chinh xac hon so vs duong chim (haversine truc tiep).
    """
    if not route_points:
        return haversine_distance(lat1, lon1, lat2, lon2)

    def _closest_idx(lat, lng):
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


def distance_point_to_segment_km(point: Coord, seg_a: Coord, seg_b: Coord) -> float:
    """
    Khoang cach xap xi (km) tu 1 diem GPS toi 1 doan duong [seg_a, seg_b].

    Xap xi phang (flat-earth) hop ly cho khoang cach ngan (vai km) quanh tuyen
    xe buyt - du chinh xac cho muc dich canh bao chech huong. Dung
    haversine_distance goc tu bustracker de quy doi do lech ve km.
    """
    lat0 = math.radians((seg_a[0] + seg_b[0]) / 2.0)
    kx = 111.320 * math.cos(lat0)  # km / do kinh do tai vi do trung binh
    ky = 110.574                    # km / do vi do

    px, py = point[1] * kx, point[0] * ky
    ax, ay = seg_a[1] * kx, seg_a[0] * ky
    bx, by = seg_b[1] * kx, seg_b[0] * ky

    abx, aby = bx - ax, by - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq == 0:
        return haversine_distance(point[0], point[1], seg_a[0], seg_a[1])

    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x, proj_y = ax + t * abx, ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def detect_route_deviation(
    current_pos: Coord,
    route_points: List[dict],
    deviation_threshold_km: float = 0.3,
) -> dict:
    """
    Tinh nang "canh bao chech huong": tim doan duong gan nhat trong tuyen
    (`route_points`, danh sach {'lat','lng'} theo dung thu tu tuyen, giong
    STOP_COORDS trong bustracker/app.py) va so khoang cach vuong goc voi
    nguong cho phep.
    """
    if len(route_points) < 2:
        return {"deviated": False, "distance_from_route_km": 0.0}

    best_dist = float("inf")
    for i in range(len(route_points) - 1):
        a = (route_points[i]["lat"], route_points[i]["lng"])
        b = (route_points[i + 1]["lat"], route_points[i + 1]["lng"])
        d = distance_point_to_segment_km(current_pos, a, b)
        best_dist = min(best_dist, d)

    deviated = best_dist > deviation_threshold_km
    return {
        "deviated": deviated,
        "distance_from_route_km": round(best_dist, 4),
        "threshold_km": deviation_threshold_km,
        "message": (
            f"CANH BAO: xe da di chech tuyen duong khoang {best_dist*1000:.0f} m"
            if deviated else "Xe dang di dung tuyen"
        ),
    }
