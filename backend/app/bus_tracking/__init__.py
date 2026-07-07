"""bus_tracking – Export các symbol chính."""

from app.bus_tracking.geo import (
    build_stop_distance_cache,
    calculate_bus_distance_from_start,
    detect_route_deviation,
    get_stop_distance,
    get_total_distance,
    haversine_distance,
)
from app.bus_tracking.route_data import (
    ROUTE_SEGMENT_DISTANCES,
    get_segment_distances,
    validate_segment_distances,
)
from app.bus_tracking.tracker import BusTracker

__all__ = [
    # tracker
    "BusTracker",
    # geo
    "build_stop_distance_cache",
    "calculate_bus_distance_from_start",
    "detect_route_deviation",
    "get_stop_distance",
    "get_total_distance",
    "haversine_distance",
    # route_data
    "ROUTE_SEGMENT_DISTANCES",
    "get_segment_distances",
    "validate_segment_distances",
]
