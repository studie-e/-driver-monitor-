"""
TripRoster - logic nghiep vu "diem danh da lop": quet mat khi len/xuong xe,
doi chieu danh sach, kiem tra khoang xe truoc khi ket thuc chuyen.

Day la phan logic quy trinh (khong co san trong repo face-api.js - repo do
chi cung cap detection/recognition o tang thap hon), duoc xay dung rieng de
noi ket qua tu FaceMatcher thanh nghiep vu an toan day du theo dung mo ta
bai toan: "doi chieu danh sach... neu hoc sinh len nham xe hoac chua quet
khuon mat khi xuong tram, giao dien se hien canh bao ngay lap tuc cho tai xe"
va "kiem tra khoang xe truoc khi ket thuc chuyen, neu con hoc sinh tren xe...
he thong se canh bao ngay cho tai xe, nha truong va phu huynh".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class BoardingEvent(str, Enum):
    BOARDED = "boarded"
    ALIGHTED = "alighted"


@dataclass
class RosterEntry:
    student_id: str
    full_name: str
    status: str = "not_boarded"        # not_boarded | on_bus | alighted
    boarded_at: Optional[str] = None
    alighted_at: Optional[str] = None


@dataclass
class ScanAlert:
    severity: str    # "warning" | "critical"
    code: str        # "wrong_bus" | "duplicate_scan" | "left_on_bus" | "mismatch"
    message: str
    student_id: Optional[str] = None


class TripRoster:
    """
    1 instance = 1 chuyen xe (bus_id + ngay + tuyen). Quan ly danh sach hoc
    sinh DUOC PHAN CONG cho chuyen nay va trang thai len/xuong cua tung em.
    """

    def __init__(self, bus_id: str, expected_students: List[dict]):
        """
        expected_students: [{'student_id':..., 'full_name':...}, ...]
        - danh sach hoc sinh DUNG duoc phan cong len xe nay trong chuyen nay.
        """
        self.bus_id = bus_id
        self.roster: Dict[str, RosterEntry] = {
            s["student_id"]: RosterEntry(student_id=s["student_id"], full_name=s["full_name"])
            for s in expected_students
        }
        self.history: List[dict] = []

    def register_scan(self, student_id: str, event: BoardingEvent,
                       matched_full_name: Optional[str] = None) -> List[ScanAlert]:
        """
        Ham goi chinh khi co 1 luot quet khuon mat (len hoac xuong xe).
        Tra ve danh sach canh bao (rong neu moi thu binh thuong) de day
        ngay len giao dien tai xe / websocket phu huynh-nha truong.
        """
        alerts: List[ScanAlert] = []
        now = datetime.now().isoformat()

        entry = self.roster.get(student_id)
        if entry is None:
            # Hoc sinh khong co trong danh sach duoc phan cong cho xe nay
            alerts.append(ScanAlert(
                severity="critical",
                code="wrong_bus",
                message=f"CANH BAO: hoc sinh {matched_full_name or student_id} "
                        f"khong thuoc danh sach xe {self.bus_id} - co the LEN NHAM XE",
                student_id=student_id,
            ))
            self._log(student_id, event.value, alerts)
            return alerts

        if event == BoardingEvent.BOARDED:
            if entry.status == "on_bus":
                alerts.append(ScanAlert(
                    severity="warning", code="duplicate_scan",
                    message=f"{entry.full_name} da quet len xe truoc do (quet trung lap)",
                    student_id=student_id,
                ))
            else:
                entry.status = "on_bus"
                entry.boarded_at = now

        elif event == BoardingEvent.ALIGHTED:
            if entry.status != "on_bus":
                alerts.append(ScanAlert(
                    severity="warning", code="mismatch",
                    message=f"{entry.full_name} quet xuong xe nhung he thong chua ghi nhan da len xe",
                    student_id=student_id,
                ))
            entry.status = "alighted"
            entry.alighted_at = now

        self._log(student_id, event.value, alerts)
        return alerts

    def cabin_sweep_check(self) -> List[ScanAlert]:
        """
        Ham goi truoc khi KET THUC CHUYEN: quet lai toan bo danh sach, neu
        con hoc sinh nao trang thai "on_bus" (da len nhung chua quet xuong)
        thi phat canh bao khan cap - dung cho tinh nang "kiem tra khoang xe
        truoc khi ket thuc chuyen" trong de bai.
        """
        alerts: List[ScanAlert] = []
        for entry in self.roster.values():
            if entry.status == "on_bus":
                alerts.append(ScanAlert(
                    severity="critical",
                    code="left_on_bus",
                    message=f"KHAN CAP: hoc sinh {entry.full_name} VAN CON TREN XE "
                            f"{self.bus_id} khi ket thuc chuyen - kiem tra khoang xe NGAY",
                    student_id=entry.student_id,
                ))
        return alerts

    def summary(self) -> dict:
        counts = {"not_boarded": 0, "on_bus": 0, "alighted": 0}
        for entry in self.roster.values():
            counts[entry.status] += 1
        return {
            "bus_id": self.bus_id,
            "total_students": len(self.roster),
            "counts": counts,
            "still_on_bus": [e.full_name for e in self.roster.values() if e.status == "on_bus"],
        }

    def _log(self, student_id: str, event: str, alerts: List[ScanAlert]) -> None:
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "student_id": student_id,
            "event": event,
            "alerts": [a.__dict__ for a in alerts],
        })
