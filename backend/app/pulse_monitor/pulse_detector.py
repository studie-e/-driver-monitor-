"""
pulse_detector.py - do nhip tim tu xa (rPPG) qua webcam, dung thuat toan POS.

THAY THE hoan toan cach tiep can cu (port tu thearn/webcam-pulse-detector: trung
binh 3 kenh mau + FFT tho). Phan thuat toan loi (pos_algorithm, bandpass_filter,
estimate_hr_over_time trong algorithms.py; detect_face_box, get_forehead_roi_
from_face_box trong face_roi.py) la thuat toan POS (Wang et al., IEEE TBME 2016)
do nguoi dung tu xay va tu kiem chung bang du lieu tong hop (xem
backend/tests/test_pulse_synthetic.py) truoc khi tich hop vao day.

File nay CHI la phan orchestration: adapt thuat toan (von viet cho xu ly 1 file
video lien tuc) sang kien truc stateful theo TUNG NGUOI (person_id), nhan tung
frame roi rac qua API (POST /api/pulse/{person_id}/frame), giu dung API contract
cu (`PulseMonitor.process_frame(person_id, frame_bgr) -> dict`) de KHONG phai
sua main.py.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.pulse_monitor.algorithms import bandpass_filter, estimate_hr_over_time, pos_algorithm
from app.pulse_monitor.face_roi import detect_face_box, get_forehead_roi_from_face_box

# Cua so tinh HR (giay) - phai du dai de POS + bandpass co du chu ky nhip tim,
# dung gia tri da kiem chung trong test_synthetic.py (win_sec=10.0 cho ket qua on dinh)
WINDOW_SEC = 10.0
# Buffer giu du lieu dai hon 1 chut so voi window de co bien do khi trim theo thoi gian
BUFFER_SEC = 14.0
# Khong tinh lai POS/FFT moi frame (ton CPU vo ich) - chi tinh lai moi khi da qua >=1s
MIN_RECOMPUTE_INTERVAL_SEC = 1.0
# So mau toi thieu truoc khi thu tinh HR (tranh cua so qua thua du lieu luc moi bat dau)
MIN_SAMPLES = 30


class RPPGPulseSignal:
    """
    Buffer tin hieu RGB rPPG cho 1 NGUOI, chay POS + bandpass + FFT tren cua so
    truot moi khi co du du lieu. Thay the FaceMeanPulseSignal cu (RGB-mean + FFT tho).
    """

    def __init__(self, window_sec: float = WINDOW_SEC, buffer_sec: float = BUFFER_SEC) -> None:
        self.window_sec = window_sec
        self.buffer_sec = buffer_sec
        self.rgb_buffer: list = []   # list[[R,G,B]]
        self.timestamps: list = []
        self.bpm: Optional[float] = None
        self._last_recompute_time: float = 0.0

    def push_sample(self, rgb_mean: Tuple[float, float, float], timestamp: float) -> Optional[float]:
        """Nap 1 mau RGB trung binh vung tran. Tra ve BPM moi nhat (None neu chua du du lieu)."""
        self.rgb_buffer.append(rgb_mean)
        self.timestamps.append(timestamp)

        # Trim buffer theo thoi gian (giu toi da buffer_sec giay gan nhat)
        cutoff = timestamp - self.buffer_sec
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.rgb_buffer.pop(0)

        if len(self.timestamps) < MIN_SAMPLES:
            return None
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration < self.window_sec:
            return None  # chua du 1 cua so hoan chinh (win_sec=10s) de POS+FFT on dinh

        # Throttle: chi tinh lai HR moi >=1s, tranh chay POS/FFT tren tung frame lien tuc
        if timestamp - self._last_recompute_time < MIN_RECOMPUTE_INTERVAL_SEC and self.bpm is not None:
            return self.bpm
        self._last_recompute_time = timestamp

        fps = len(self.timestamps) / duration
        rgb = np.array(self.rgb_buffer)

        pulse_raw = pos_algorithm(rgb, fps)
        pulse_filtered = bandpass_filter(pulse_raw, fps)
        _, hr_values = estimate_hr_over_time(
            pulse_filtered, fps, win_sec=self.window_sec, step_sec=self.window_sec
        )
        if len(hr_values) == 0:
            return self.bpm  # giu gia tri cu neu FFT khong tim thay dinh pho hop le

        self.bpm = float(hr_values[-1])
        return self.bpm

    def reset(self) -> None:
        self.rgb_buffer, self.timestamps = [], []
        self.bpm = None
        self._last_recompute_time = 0.0


class PulseMonitor:
    """
    Ham goi hoan chinh cho tinh nang "do nhip tim tu xa theo doi trang thai
    cang thang / dot quy cua tai xe va hoc sinh tren xe" - ban nang cap dung POS.
    """

    BPM_LOW = 50    # nhip tim qua thap
    BPM_HIGH = 130  # nhip tim qua cao / cang thang cap tinh

    def __init__(self) -> None:
        self._signals: Dict[str, RPPGPulseSignal] = {}
        self._last_face_box: Dict[str, tuple] = {}  # giu bbox gan nhat, khong "mat" ROI khi detect hut vai frame

    def process_frame(self, person_id: str, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> dict:
        """
        Ham goi chinh: dua 1 frame webcam cua 1 nguoi (tai xe hoac hoc sinh),
        tra ve dict trang thai gom bpm hien tai va co canh bao bat thuong hay
        khong (dung cho tich hop vao dashboard / websocket qua main.py).
        """
        if person_id not in self._signals:
            self._signals[person_id] = RPPGPulseSignal()
        signal = self._signals[person_id]
        if timestamp is None:
            timestamp = time.time()

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box = detect_face_box(frame_bgr, gray)
        if box is not None:
            self._last_face_box[person_id] = box
        box = self._last_face_box.get(person_id)

        if box is None:
            return {
                "person_id": person_id,
                "face_found": False,
                "bpm": round(signal.bpm, 1) if signal.bpm else None,
                "status": "no_face",
                "alert": False,
            }

        x, y, w, h = box
        x1, y1, x2, y2 = get_forehead_roi_from_face_box(x, y, w, h)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "person_id": person_id,
                "face_found": True,
                "bpm": round(signal.bpm, 1) if signal.bpm else None,
                "status": "roi_invalid",
                "alert": False,
            }

        mean_bgr = roi.reshape(-1, 3).mean(axis=0)
        rgb_mean = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
        bpm = signal.push_sample(rgb_mean, timestamp)

        if bpm is None:
            duration = (signal.timestamps[-1] - signal.timestamps[0]) if len(signal.timestamps) > 1 else 0.0
            return {
                "person_id": person_id,
                "face_found": True,
                "bpm": None,
                "status": "calibrating",
                "alert": False,
                "progress_pct": round(100.0 * duration / WINDOW_SEC, 1),
            }

        alert = bpm <= self.BPM_LOW or bpm >= self.BPM_HIGH
        status = "abnormal_high" if bpm >= self.BPM_HIGH else ("abnormal_low" if bpm <= self.BPM_LOW else "normal")

        return {
            "person_id": person_id,
            "face_found": True,
            "bpm": round(bpm, 1),
            "status": status,
            "alert": alert,
            "message": (
                f"BAT THUONG: nhip tim {bpm:.0f} bpm - nghi ngo cang thang/dot quy, can kiem tra ngay"
                if alert else "Nhip tim binh thuong"
            ),
        }

    def reset(self, person_id: str) -> None:
        if person_id in self._signals:
            self._signals[person_id].reset()
        self._last_face_box.pop(person_id, None)
