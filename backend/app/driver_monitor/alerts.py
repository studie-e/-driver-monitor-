"""
AlertPolicy - chống spam cảnh báo bằng cooldown theo từng loại tín hiệu.

Port từ: driver_safety/core/alerts.py (repo: Inferensys/ai-driver-safety)
"""
from __future__ import annotations

from dataclasses import dataclass

from app.driver_monitor.models import DetectionEvent


@dataclass(slots=True)
class Alert:
    timestamp: float
    signal: str
    message: str
    severity: str


class AlertPolicy:
    def __init__(self, cooldown_seconds: float = 2.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_by_signal: dict = {}

    def evaluate(self, events: list) -> list:
        alerts = []
        for event in events:
            last_alert = self._last_alert_by_signal.get(event.signal)
            if last_alert is not None and event.timestamp - last_alert < self.cooldown_seconds:
                continue
            self._last_alert_by_signal[event.signal] = event.timestamp
            alerts.append(
                Alert(
                    timestamp=event.timestamp,
                    signal=event.signal,
                    message=event.message,
                    severity=event.severity.value,
                )
            )
        return alerts
