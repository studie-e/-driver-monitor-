"""
EmergencyDispatcher - nut bam khan cap tren app di dong cua giao vien di kem/tai xe.

Khong co repo mau cho tinh nang nay (de bai chi mo ta nghiep vu, khong dua
link) nen day la module tu xay dung theo dung yeu cau: tuy loai tinh huong
(tai nan giao thong / ke gian xam nhap / hoa hoan) se phat luong canh bao
KHAC NHAU toi TUNG doi tuong cu the (canh sat, cuu hoa, phu huynh, nha
truong), nham giam thoi gian phan ung trong kich ban hoang loan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional


class IncidentType(str, Enum):
    TRAFFIC_ACCIDENT = "traffic_accident"   # tai nan giao thong
    INTRUDER = "intruder"                   # ke gian xam nhap
    FIRE = "fire"                            # hoa hoan
    MEDICAL = "medical"                      # cap cuu y te (vd. tu pulse_monitor)
    CHILD_LEFT_ON_BUS = "child_left_on_bus"  # tu dong sinh ra tu cabin_sweep_check


# Ma tran dinh tuyen: loai tinh huong -> danh sach kenh nhan canh bao
ROUTING_MATRIX: Dict[IncidentType, List[str]] = {
    IncidentType.TRAFFIC_ACCIDENT: ["police", "ambulance", "school", "parents"],
    IncidentType.INTRUDER: ["police", "school", "parents"],
    IncidentType.FIRE: ["fire_department", "police", "school", "parents"],
    IncidentType.MEDICAL: ["ambulance", "school", "parents"],
    IncidentType.CHILD_LEFT_ON_BUS: ["school", "parents", "driver"],
}

DEFAULT_MESSAGES: Dict[IncidentType, str] = {
    IncidentType.TRAFFIC_ACCIDENT: "KHAN CAP: Xe buyt gap tai nan giao thong. Can ho tro ngay lap tuc.",
    IncidentType.INTRUDER: "KHAN CAP: Phat hien nguoi la xam nhap xe buyt cho hoc sinh.",
    IncidentType.FIRE: "KHAN CAP: Co hoa hoan/khoi tren xe buyt cho hoc sinh.",
    IncidentType.MEDICAL: "KHAN CAP: Can cap cuu y te tren xe buyt.",
    IncidentType.CHILD_LEFT_ON_BUS: "KHAN CAP: Hoc sinh con o lai tren xe sau khi ket thuc chuyen.",
}


@dataclass
class DispatchResult:
    incident_type: str
    channels_notified: List[str]
    message: str
    bus_id: str
    triggered_at: str
    location: Optional[dict] = None
    extra: dict = field(default_factory=dict)


ChannelSender = Callable[[str, DispatchResult], None]


class EmergencyDispatcher:
    """
    Ham goi hoan chinh cho tinh nang "nut bam khan cap".

    `channel_senders`: dict tu ten kenh ("police", "fire_department",
    "ambulance", "school", "parents", "driver") -> ham thuc thi gui thong bao
    that (SMS/Email/Websocket/push notification...). Kien truc cho phep cam
    bat ky provider nao (Twilio, Firebase Cloud Messaging, SMTP...) ma khong
    doi logic dinh tuyen ben tren.
    """

    def __init__(self, channel_senders: Optional[Dict[str, ChannelSender]] = None) -> None:
        self.channel_senders = channel_senders or {}
        self.dispatch_log: List[DispatchResult] = []

    def trigger(
        self,
        incident_type: IncidentType,
        bus_id: str,
        location: Optional[dict] = None,
        custom_message: Optional[str] = None,
    ) -> DispatchResult:
        channels = ROUTING_MATRIX.get(incident_type, ["school", "parents"])
        message = custom_message or DEFAULT_MESSAGES.get(incident_type, "KHAN CAP tren xe buyt")

        result = DispatchResult(
            incident_type=incident_type.value,
            channels_notified=channels,
            message=message,
            bus_id=bus_id,
            triggered_at=datetime.now().isoformat(),
            location=location,
        )

        for channel in channels:
            sender = self.channel_senders.get(channel)
            if sender is not None:
                sender(channel, result)
        self.dispatch_log.append(result)
        return result
