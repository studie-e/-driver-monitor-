"""
PulseDetector - do nhip tim tu xa (remote photoplethysmography - rPPG) qua webcam.

Port & refactor tu:
  lib/processors.py -> class findFaceGetPulse  (repo: thearn/webcam-pulse-detector)

Thuat toan goc dung OpenCV de chay vong lap UI (imshow/waitKey) va luon giu
1 khuon mat "khoa" toan cuc. Ban port nay giu NGUYEN loi thuat toan xu ly tin
hieu (lay gia tri mau trung binh vung tran -> noi suy deu theo thoi gian ->
cua so Hamming -> khu DC -> FFT thuc -> loc dai tan 50-180 BPM -> chon dinh
pho lam BPM) nhung boc lai thanh 1 class stateful theo TUNG NGUOI (student_id /
driver_id), khong phu thuoc GUI, de co the goi qua REST/WebSocket cho nhieu
nguoi dung dong thoi (tai xe + hoc sinh) tren xe.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class FaceMeanPulseSignal:
    """Theo doi 1 luong tin hieu mach (1 nguoi) - buffer + FFT.

    Day la phan loi thuat toan duoc giu sat nhat voi `findFaceGetPulse.run()`
    goc, chi bo toan bo phan ve/UI (cv2.rectangle, cv2.putText, phim tat...).
    """

    def __init__(self, buffer_size: int = 250):
        self.buffer_size = buffer_size
        self.data_buffer: list = []
        self.times: list = []
        self.t0 = time.time()
        self.bpm: float = 0.0
        self.fps: float = 0.0
        self.freqs = np.array([])
        self.fft = np.array([])

    def push_forehead_sample(self, mean_intensity: float) -> Optional[float]:
        """Nap 1 mau = gia tri trung binh pixel vung tran tai thoi diem hien tai.

        Tra ve BPM uoc luong hien tai (None neu chua du du lieu, giong ban goc
        yeu cau L > 10 mau truoc khi uoc luong FFT).
        """
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(mean_intensity)

        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        if L <= 10:
            return None

        processed = np.array(self.data_buffer)
        self.fps = float(L) / (self.times[-1] - self.times[0])
        even_times = np.linspace(self.times[0], self.times[-1], L)
        interpolated = np.interp(even_times, self.times, processed)
        interpolated = np.hamming(L) * interpolated
        interpolated = interpolated - np.mean(interpolated)

        raw = np.fft.rfft(interpolated)
        self.fft = np.abs(raw)
        self.freqs = float(self.fps) / L * np.arange(L // 2 + 1)

        freqs_bpm = 60.0 * self.freqs
        idx = np.where((freqs_bpm > 50) & (freqs_bpm < 180))

        pruned = self.fft[idx]
        pfreq = freqs_bpm[idx]
        self.freqs = pfreq
        self.fft = pruned

        if pruned.size > 0:
            idx2 = int(np.argmax(pruned))
            self.bpm = float(pfreq[idx2])
        # neu khong tim thay dinh pho hop le trong dai 50-180 bpm thi giu bpm cu

        return self.bpm

    def reset(self) -> None:
        self.data_buffer, self.times = [], []
        self.t0 = time.time()
        self.bpm = 0.0


class PulseMonitor:
    """
    Ham goi hoan chinh cho tinh nang "do nhip tim tu xa theo doi trang thai
    cang thang / dot quy cua tai xe va hoc sinh tren xe".

    - Dung Haar cascade (co san trong OpenCV, giong `self.face_cascade` ban
      goc dung `haarcascade_frontalface_alt.xml`) de tim khuon mat.
    - Lay vung "tran" (forehead) bang ty le toa do tuong doi so voi bbox mat,
      dung ham `get_subface_coord`/`get_subface_means` port tu ban goc.
    - Duy tri 1 FaceMeanPulseSignal rieng cho MOI nguoi (theo person_id) de
      co the theo doi dong thoi ca tai xe lan nhieu hoc sinh.
    """

    # Nguong bat thuong (dung de canh bao cang thang / nghi ngo dot quy)
    BPM_LOW = 50   # nhip tim qua thap
    BPM_HIGH = 130  # nhip tim qua cao / cang thang cap tinh

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self._signals: dict = {}  # person_id -> FaceMeanPulseSignal

    @staticmethod
    def get_subface_coord(face_rect, fh_x: float, fh_y: float, fh_w: float, fh_h: float):
        """Port tu findFaceGetPulse.get_subface_coord - quy vung tran tu bbox mat."""
        x, y, w, h = face_rect
        return (
            int(x + w * fh_x - (w * fh_w / 2.0)),
            int(y + h * fh_y - (h * fh_h / 2.0)),
            int(w * fh_w),
            int(h * fh_h),
        )

    @staticmethod
    def get_subface_means(frame_bgr: np.ndarray, coord) -> float:
        """Port tu findFaceGetPulse.get_subface_means - trung binh 3 kenh mau vung con."""
        x, y, w, h = coord
        subframe = frame_bgr[y:y + h, x:x + w, :]
        if subframe.size == 0:
            return 0.0
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        return float((v1 + v2 + v3) / 3.0)

    def _detect_face(self, frame_bgr: np.ndarray):
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        detected = list(
            self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        )
        if not detected:
            return None
        detected.sort(key=lambda a: a[-1] * a[-2])
        return tuple(detected[-1])  # bbox lon nhat (gan camera nhat)

    def process_frame(self, person_id: str, frame_bgr: np.ndarray) -> dict:
        """
        Ham goi chinh: dua 1 frame webcam cua 1 nguoi (tai xe hoac hoc sinh),
        tra ve dict trang thai gom bpm hien tai va co canh bao bat thuong hay
        khong (dung cho tich hop vao dashboard / websocket).
        """
        if person_id not in self._signals:
            self._signals[person_id] = FaceMeanPulseSignal()
        signal = self._signals[person_id]

        face_rect = self._detect_face(frame_bgr)
        if face_rect is None:
            return {
                "person_id": person_id,
                "face_found": False,
                "bpm": round(signal.bpm, 1),
                "status": "no_face",
                "alert": False,
            }

        forehead = self.get_subface_coord(face_rect, 0.5, 0.18, 0.25, 0.15)
        mean_intensity = self.get_subface_means(frame_bgr, forehead)
        bpm = signal.push_forehead_sample(mean_intensity)

        if bpm is None:
            return {
                "person_id": person_id,
                "face_found": True,
                "bpm": None,
                "status": "calibrating",
                "alert": False,
                "progress_pct": round(100.0 * len(signal.data_buffer) / 10, 1),
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
