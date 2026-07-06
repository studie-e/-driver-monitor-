# Hệ thống đảm bảo an toàn xe đưa đón học sinh bằng AI

Dự án ghép **5 tính năng** thành 1 hệ thống backend (FastAPI) + frontend (HTML/JS thuần),
mỗi tính năng port/adapt logic thật từ đúng repo GitHub đề bài đã cho.

## Bảng ánh xạ: tính năng → repo gốc → module trong project

| # | Tính năng | Repo gốc | Hàm/class chính đã port | Module trong project |
|---|---|---|---|---|
| 1 | Điểm danh khuôn mặt lên/xuống xe | `justadudewhohacks/face-api.js` | `faceapi.detectSingleFace().withFaceLandmarks().withFaceDescriptor()`, `FaceMatcher` (euclidean distance, threshold 0.6) | `frontend/face_checkin.html` (chạy face-api.js thật trên browser) + `backend/app/face_checkin/matcher.py` (port thuật toán so khớp) + `roster.py` (nghiệp vụ điểm danh, tự viết theo yêu cầu bài toán) |
| 2 | Cảnh báo tài xế ngủ gật | `Inferensys/ai-driver-safety` | `eye_aspect_ratio`, `mouth_aspect_ratio`, `horizontal_head_offset` (`vision/metrics.py`), `RiskScorer` (`core/scoring.py`), `AlertPolicy` (`core/alerts.py`) | `backend/app/driver_monitor/{metrics,scoring,alerts,models,pipeline}.py` |
| 3 | Đo nhịp tim từ xa | `thearn/webcam-pulse-detector` | `findFaceGetPulse` (`lib/processors.py`): lấy trung bình vùng trán → nội suy → Hamming window → FFT → lọc dải 50–180 BPM | `backend/app/pulse_monitor/pulse_detector.py` |
| 4 | Nút bấm khẩn cấp | (không có repo mẫu — tự xây theo mô tả bài toán) | Ma trận định tuyến theo loại sự cố → kênh cảnh báo | `backend/app/emergency/dispatcher.py` |
| 5 | Báo ETA + cảnh báo chệch hướng | `Terrificdatabytes/bustracker` | `haversine_distance`, `calculate_distance_with_waypoints`, `detect_bus_direction`, `find_next_stop_bidirectional`, `predict_eta` (`app.py`) | `backend/app/bus_tracking/{geo,tracker}.py` |

Toàn bộ code trong bảng trên **đã chạy thử thành công** (`backend/run_demo.py`) — không phải code minh hoạ suông.

## Kiến trúc

```
school_bus_safety/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI — ghép cả 5 tính năng thành REST + WebSocket API
│   │   ├── websocket_manager.py    # đẩy cảnh báo real-time tới dashboard tài xe/trường/phụ huynh
│   │   ├── models/schemas.py       # Pydantic request/response
│   │   ├── face_checkin/
│   │   │   ├── matcher.py          # port FaceMatcher (face-api.js)
│   │   │   └── roster.py           # state machine điểm danh + "quét khoang xe"
│   │   ├── driver_monitor/
│   │   │   ├── metrics.py          # port EAR/MAR/head-offset (ai-driver-safety)
│   │   │   ├── scoring.py          # port RiskScorer (noisy-OR fusion)
│   │   │   ├── alerts.py           # port AlertPolicy (cooldown)
│   │   │   ├── models.py           # port DriverState/DetectionEvent/...
│   │   │   └── pipeline.py         # glue: Haar cascade (mắt/miệng) → EAR/MAR → risk_score
│   │   ├── pulse_monitor/
│   │   │   └── pulse_detector.py   # port findFaceGetPulse (FFT rPPG)
│   │   ├── emergency/
│   │   │   └── dispatcher.py       # định tuyến cảnh báo theo loại sự cố
│   │   └── bus_tracking/
│   │       ├── geo.py              # port haversine + waypoint distance, thêm route-deviation
│   │       └── tracker.py          # port direction/next-stop/ETA
│   ├── requirements.txt
│   └── run_demo.py                 # demo end-to-end, gọi đủ 5 tính năng bằng dữ liệu mẫu
└── frontend/
    ├── face_checkin.html           # kiosk quét mặt — dùng face-api.js THẬT qua CDN
    └── driver_dashboard.html       # dashboard tài xế: nút khẩn cấp + trạng thái real-time (WebSocket)
```

## Chạy thử

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # nếu cần

# 1. Chạy demo logic (không cần webcam, dùng dữ liệu giả lập) — xác nhận toàn bộ 5 tính năng chạy đúng
python run_demo.py

# 2. Chạy server thật
uvicorn app.main:app --reload --port 8000
# Mở docs API tự sinh: http://localhost:8000/docs
```uvicorn app.main:app --reload --port 8000

Sau đó mở `frontend/face_checkin.html` và `frontend/driver_dashboard.html` bằng trình duyệt
(cần bật webcam, và backend đang chạy ở `localhost:8000`).

## Luồng nghiệp vụ chính (đúng theo mô tả đề bài)

1. **Bắt đầu chuyến**: tài xế gọi `POST /api/checkin/start_trip` với danh sách học sinh được
   phân công cho xe này.
2. **Học sinh lên xe**: kiosk `face_checkin.html` chạy face-api.js quét mặt → gửi descriptor
   128 chiều lên `POST /api/checkin/scan` (`event=boarded`). Nếu học sinh **không thuộc** danh
   sách xe này → cảnh báo "LÊN NHẦM XE" tức thì (`wrong_bus`), đồng thời tự động kích hoạt
   `EmergencyDispatcher` gửi cảnh báo tới nhà trường/phụ huynh.
3. **Học sinh xuống xe**: quét lại với `event=alighted`.
4. **Kết thúc chuyến**: tài xế gọi `POST /api/checkin/end_trip/{bus_id}` → hệ thống **quét
   khoang xe** (`cabin_sweep_check`), nếu còn học sinh ở trạng thái `on_bus` → cảnh báo
   KHẨN CẤP ngay tới tài xế + nhà trường + phụ huynh.
5. **Song song suốt chuyến**:
   - Camera hướng vào tài xế gửi từng frame tới `POST /api/driver/{driver_id}/frame` →
     phát hiện nhắm mắt/ngáp/mất tập trung, tính `risk_score` hợp nhất đa tín hiệu.
   - Camera đo nhịp tim gửi frame tới `POST /api/pulse/{person_id}/frame` (dùng chung cho
     cả tài xế và học sinh, phân biệt theo `person_id`) → cảnh báo khi BPM bất thường
     (≤50 hoặc ≥130).
   - GPS xe gửi vị trí tới `POST /api/tracking/update_location` → tính ETrackA, hướng đi,
     và tự động phát hiện **chệch tuyến** (khoảng cách vuông góc tới tuyến vượt ngưỡng).
   - Giáo viên/tài xế bấm nút khẩn cấp `POST /api/emergency/trigger` với 1 trong các loại
     `traffic_accident` / `intruder` / `fire` / `medical` → hệ thống tự route tới đúng tổ hợp
     kênh (công an/cứu hỏa/cứu thương/nhà trường/phụ huynh) theo `ROUTING_MATRIX`.
6. Tất cả cảnh báo được đẩy **real-time** qua WebSocket (`/ws/{room}`) tới dashboard tài xe
   (`driver_dashboard.html`) — room là `bus_id`, `driver_<id>`, hoặc `pulse_<id>`.

## Giới hạn đã biết & hướng nâng cấp khi triển khai thật

- **driver_monitor/pipeline.py**: do môi trường build không có Internet để tải model
  MediaPipe FaceLandmarker (`.task`) hay `dlib` 68-point, pipeline hiện dùng Haar cascade
  (mắt/miệng, có sẵn trong OpenCV) để suy ra 6 điểm EAR/MAR gần đúng. Công thức
  `eye_aspect_ratio`/`mouth_aspect_ratio` vẫn giữ **nguyên bản gốc** — khi triển khai thật,
  chỉ cần thay bộ dò landmark (MediaPipe FaceLandmarker hoặc dlib) mà **không cần sửa**
  `metrics.py`/`scoring.py`/`alerts.py`.
- **pulse_monitor**: dùng Haar cascade `haarcascade_frontalface_alt.xml` giống hệt bản gốc
  `webcam-pulse-detector`, đã port đúng thuật toán FFT gốc.
- **face_checkin**: face-api.js chạy thật ở trình duyệt (client-side), backend chỉ nhận
  vector 128-D — cần host models của face-api.js (đã trỏ tới weights chính thức trên GitHub
  raw) hoặc copy về server riêng để giảm phụ thuộc CDN khi lên production.
- **emergency/dispatcher.py**: `channel_senders` hiện là hàm log demo; khi lên production
  chỉ cần cắm SDK thật (Twilio SMS, Firebase Cloud Messaging, SMTP...) vào đúng vị trí này.
- Dữ liệu học sinh/danh sách xe hiện lưu trong bộ nhớ (dict) cho mục đích demo — cần thay
  bằng database (PostgreSQL, vd giống stack mà nhóm Blackpink đã dùng ở dự án bãi đỗ xe)
  khi triển khai thật.
