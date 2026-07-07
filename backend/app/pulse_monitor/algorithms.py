"""
algorithms.py - loi thuat toan rPPG (POS + bandpass + FFT sliding-window HR).

Port NGUYEN XI tu file rppg_core.py (nguoi dung tu viet), thay the hoan toan
cach tiep can "trung binh 3 kenh mau" cua webcam-pulse-detector cu. Day la
thuat toan POS (Plane-Orthogonal-to-Skin) - Wang, W. et al., "Algorithmic
Principles of Remote PPG", IEEE TBME 2016 - dung ty le giua cac kenh mau
(khong dung gia tri tuyet doi) de tu dong triet tieu phan lon nhieu do cuong
do anh sang thay doi dong deu tren ca 3 kenh, manh hon han so voi FFT tren
tin hieu tho (ban cu).

KHONG sua cong thuc toan hoc o day - chi tach rieng thanh module de dung lai
duoc o ca 2 boi canh: xu ly nguyen video offline va xu ly tung frame qua API
(pulse_detector.py, dung trong production cua he thong nay).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

# Ma tran chieu POS chuan (Wang et al. 2016): Xs = G-B ; Ys = -2R+G+B
_POS_PROJECTION = np.array([[0, 1, -1], [-2, 1, 1]])

# Dai tan hop le mac dinh (0.7-4Hz = 42-240 bpm), dung nhu ban goc cua nguoi dung
DEFAULT_LOW_HZ = 0.7
DEFAULT_HIGH_HZ = 4.0


def pos_algorithm(rgb_signal: np.ndarray, fps: float, win_sec: float = 1.6) -> np.ndarray:
    """
    Thuat toan POS (Plane Orthogonal to Skin).

    Input: rgb_signal (N,3) theo thu tu R,G,B
    Output: chuoi tin hieu mach (pulse signal) do dai N, da overlap-add qua cac cua so.
    """
    rgb = rgb_signal.T  # shape (3, N)
    N = rgb.shape[1]
    H = np.zeros(N)
    win_len = int(round(win_sec * fps))
    if win_len < 2:
        win_len = 2

    for n in range(N - win_len + 1):
        m = n + win_len
        segment = rgb[:, n:m]
        mean_seg = np.mean(segment, axis=1, keepdims=True)
        mean_seg[mean_seg == 0] = 1e-8
        Cn = segment / mean_seg  # chuan hoa moi kenh ve quanh gia tri 1

        S = _POS_PROJECTION @ Cn  # S[0] = Xs, S[1] = Ys
        std_ratio = np.std(S[0]) / (np.std(S[1]) + 1e-8)
        h = S[0] + std_ratio * S[1]
        H[n:m] += (h - np.mean(h))

    return H


def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float = DEFAULT_LOW_HZ,
                     high_hz: float = DEFAULT_HIGH_HZ, order: int = 3) -> np.ndarray:
    """
    Loc thong dai quanh khoang tan so nhip tim nguoi (0.7-4Hz = 42-240 bpm),
    loai bo nhieu tan so thap (thay doi anh sang cham) va tan so cao (rung/nhieu camera).
    """
    nyq = fps / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def estimate_hr_over_time(pulse_signal: np.ndarray, fps: float, win_sec: float = 10.0,
                           step_sec: float = 1.0, low_hz: float = DEFAULT_LOW_HZ,
                           high_hz: float = DEFAULT_HIGH_HZ):
    """
    Uoc luong HR theo thoi gian bang FFT tren cua so truot.
    Tra ve (times, hr_values) - moi diem la HR (bpm) uoc luong tai thoi diem do.
    """
    win_len = int(win_sec * fps)
    step_len = int(step_sec * fps)
    N = len(pulse_signal)

    n_fft = win_len * 8  # zero-padding de tang do phan giai tan so hien thi

    times, hr_values = [], []
    for start in range(0, N - win_len + 1, step_len):
        segment = pulse_signal[start:start + win_len]
        segment = segment * np.hanning(len(segment))  # giam ro ri pho (spectral leakage)

        freqs = rfftfreq(n_fft, d=1.0 / fps)
        mag = np.abs(rfft(segment, n=n_fft))

        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            continue

        peak_freq = freqs[mask][np.argmax(mag[mask])]
        hr_bpm = peak_freq * 60.0

        times.append((start + win_len / 2) / fps)
        hr_values.append(hr_bpm)

    return np.array(times), np.array(hr_values)
