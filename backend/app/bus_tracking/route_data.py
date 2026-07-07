"""
route_data.py – Lưu khoảng cách từng đoạn tuyến đường (km) được tính trước.

Cách dùng:
  - Mỗi route_id ánh xạ tới list các khoảng cách (km) giữa 2 trạm kế tiếp.
  - Số phần tử = số trạm - 1.
  - Dữ liệu này thay thế việc gọi OSRM API mỗi lần, đạt độ chính xác ~98-99%.

Cách tính khoảng cách từng đoạn:
  Option A: Đo trực tiếp trên Google Maps (công cụ "Đo khoảng cách")
  Option B: Nhờ AI (Copilot/ChatGPT) tính từ tọa độ GPS
  Option C: GraphHopper API (miễn phí 500 req/ngày)

Port từ: Terrificdatabytes/bustracker/manual_distances.py
  - Cấu trúc giữ nguyên, thêm helper functions
  - Dữ liệu mặc định để trống (điền route thực của trường vào đây)
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Dữ liệu khoảng cách từng đoạn (km) – ĐIỀN VÀO ĐÂY
# ---------------------------------------------------------------------------
# Ví dụ từ repo tham khảo (tuyến 48AC, Madurai - India):
# ROUTE_SEGMENT_DISTANCES = {
#     "48AC": [
#         0.640,  # Trạm 1 → Trạm 2
#         0.730,  # Trạm 2 → Trạm 3
#         ...
#     ]
# }
#
# Hướng dẫn: thêm route của trường vào đây với định dạng:
# "route_id": [kc_1_2, kc_2_3, kc_3_4, ...]
# Số phần tử phải bằng (số trạm - 1).

ROUTE_SEGMENT_DISTANCES: Dict[str, List[float]] = {
    # Ví dụ tuyến tham khảo từ bustracker (Madurai, India) – giữ để test
    "48AC": [
        0.640, 0.730, 1.280, 1.430, 0.570, 0.480, 0.327,
        0.270, 0.640, 0.080, 0.210, 0.620, 0.550, 0.810,
        0.120, 0.100, 0.370, 0.270, 1.190, 0.570, 0.920,
        0.830, 0.220, 0.830, 0.960, 0.110, 0.450,
    ],
    "23": [
        0.640, 0.730, 1.280, 1.430, 0.570, 0.480, 0.327,
        0.270, 0.640, 0.080, 0.210, 0.620, 0.550, 0.810,
        0.120, 0.100,
    ],
    # ---------- THÊM TUYẾN THỰC CỦA TRƯỜNG VÀO ĐÂY ----------
    # "route_truong_01": [
    #     1.2, 0.8, 1.5, 0.9, ...
    # ],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_segment_distances(route_id: str) -> Optional[List[float]]:
    """
    Trả về list khoảng cách từng đoạn (km) của route.
    Trả về None nếu route không có trong database.
    """
    return ROUTE_SEGMENT_DISTANCES.get(route_id)


def validate_segment_distances(route_id: str, num_stops: int) -> bool:
    """
    Kiểm tra số đoạn có khớp với số trạm không.
    Số đoạn đúng = num_stops - 1.
    """
    segments = get_segment_distances(route_id)
    if segments is None:
        return False
    expected = num_stops - 1
    if len(segments) != expected:
        print(
            f"[WARN] Route {route_id}: cần {expected} đoạn, "
            f"có {len(segments)} đoạn trong ROUTE_SEGMENT_DISTANCES"
        )
        return False
    return True


def get_total_route_distance(route_id: str) -> Optional[float]:
    """Tổng khoảng cách toàn tuyến (km)."""
    segments = get_segment_distances(route_id)
    if segments is None:
        return None
    return sum(segments)
