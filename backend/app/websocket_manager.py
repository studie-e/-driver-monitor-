"""
ConnectionManager - quan ly cac ket noi WebSocket (dashboard tai xe, nha
truong, phu huynh) de day canh bao real-time. Dung chung cho ca 5 tinh nang.
"""
from __future__ import annotations

import json
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        # room = vd "bus_42" -> danh sach cac websocket dang lang nghe
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room: str) -> None:
        await websocket.accept()
        self.rooms.setdefault(room, []).append(websocket)

    def disconnect(self, websocket: WebSocket, room: str) -> None:
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)

    async def broadcast(self, room: str, payload: dict) -> None:
        dead = []
        for ws in self.rooms.get(room, []):
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, room)


manager = ConnectionManager()
