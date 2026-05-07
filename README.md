# 🚗 Hệ Thống Nhận Dạng Biển Số Xe Việt Nam
## License Plate Recognition System for Vietnam

> **Độ Chính Xác: 88%** - Hệ thống OCR tối ưu hóa cho biển số xe Việt Nam

### 📋 Mô Tả

Hệ thống nhận dạng biển số xe hoàn chỉnh tối ưu hóa cho chuẩn Việt Nam, kết hợp:
- **Xử lý ảnh nâng cao**: CLAHE, Bilateral Filter, Sharpening
- **Phát hiện bằng YOLOv8**: Độ chính xác cao cho bất kỳ điều kiện ánh sáng nào
- **OCR Chuẩn Việt Nam**: Công cụ OCR chuyên dụng với độ chính xác 88%
- **Xác thực tự động**: Kiểm tra định dạng biển số theo chuẩn Việt Nam

### ✨ Tính Năng Chính

| Tính Năng | Chi Tiết |
|-----------|---------|
| **Định dạng mới** | Hỗ trợ `30G - 493.44` (định dạng 2020) |
| **Định dạng cũ** | Hỗ trợ `51F.123.45` (định dạng cũ) |
| **Mã tỉnh/TP** | Nhận dạng tự động 63 tỉnh/thành phố Việt Nam |
| **Xử lý đa tỷ lệ** | Biển số ngang, dọc, tilt, mờ |
| **Hỗ trợ đa nền tảng** | Ảnh tĩnh, video, camera real-time |
| **GPU Acceleration** | Tùy chọn dùng CUDA (tự động fallback CPU) |

### 🏗️ Cấu Trúc Dự Án

```
├── src/
│   ├── vietnamese_ocr.py      ⭐ OCR engine (88% accuracy)
│   ├── plate_detector.py       Phát hiện biển số YOLOv8
│   ├── image_processing.py     Tiền xử lý hình ảnh
│   ├── plate_filter.py         Lọc kết quả false positive
│   └── utils.py                Hàm tiện ích
├── models/
│   └── best.pt                 YOLOv8 model
├── dataset_final/              Dataset huấn luyện
├── config.py                   Cấu hình chung
├── inference.py                Main API
├── train.py                    Huấn luyện model
├── video_recognition.py        Xử lý video/webcam
├── webcam.py                   Demo giao diện
├── web_app.py                  Web interface (Flask)
└── templates/, static/         Web UI assets
```

## 🚀 Cài Đặt & Khởi Động Nhanh

### 1. Cài Đặt Cơ Bản

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Nhận Dạng Ảnh Đơn Lẻ

```python
from inference import LicensePlateRecognizer
import cv2

# Khởi tạo với Vietnamese OCR (88% accuracy)
recognizer = LicensePlateRecognizer(use_vietnamese_ocr=True)

# Nhận dạng
result = recognizer.recognize_plate('car_image.jpg')

# In kết quả
for plate in result['plates']:
    print(f"Biển số: {plate['recognition']['plate_number']}")
    print(f"Độ tin cậy: {plate['recognition']['confidence']:.2%}")
    print(f"Hợp lệ: {plate['recognition'].get('is_valid', False)}")
```

### 3. Demo Webcam Real-Time

```bash
python webcam.py
```

### 4. Web Interface

```bash
python run_web.py
```
Truy cập: `http://localhost:5000`

### 5. Xử Lý Video

```python
from video_recognition import VideoPlateRecognizer

recognizer = VideoPlateRecognizer(use_vietnamese_ocr=True)
recognizer.process_video('input.mp4', output_path='output.mp4')
```

## ⚙️ Cấu Hình

Chỉnh sửa `config.py`:

```python
# OCR Engine
VIETNAM_OCR_ENABLED = True
VIETNAM_CONFIDENCE_THRESHOLD = 0.3

# Ký tự cho phép (loại bỏ I, O, Q)
ALLOWED_CHARS = '0123456789ABCDEFGHJKLMNPRSTUVWXYZ'

# Model
IMG_WIDTH = 128
IMG_HEIGHT = 32
MAX_PLATE_LENGTH = 12
```

## 📊 Hiệu Năng

| Tác Vụ | Độ Chính Xác | Thời Gian |
|--------|------------|----------|
| Phát hiện biển số | 95%+ | ~100ms |
| **OCR (định dạng mới)** | **88%** | ~200ms |
| OCR (định dạng cũ) | 85%+ | ~200ms |

## 🔬 Quy Trình Xử Lý

```
📸 Ảnh Đầu Vào
  ↓
🔍 Phát Hiện Biển Số (YOLOv8)
  ↓
✂️ Cắt Vùng Biển Số
  ↓
🖼️ Tiền Xử Lý (CLAHE, Bilateral Filter, Sharpening)
  ↓
👁️ OCR Chuẩn Việt Nam (EasyOCR + Shape Analysis)
  ↓
✅ Định Dạng & Xác Thực
  ↓
📤 Kết Quả: Biển Số + Độ Tin Cậy + Trạng Thái Hợp Lệ
```

## 🧠 Core Engine: `VietnameseOCREngine`

### Tính năng chính
- ✅ **Shape-based analysis**: Phân biệt G/6 với 95% accuracy
- ✅ **Context-aware replacement**: Sửa lỗi dựa trên ngữ cảnh
- ✅ **Advanced preprocessing**: CLAHE + Bilateral + Sharpening
- ✅ **Format flexibility**: Hỗ trợ tất cả định dạng biển số

### Sử dụng trực tiếp

```python
from src.vietnamese_ocr import VietnameseOCREngine
import cv2

ocr = VietnameseOCREngine(use_gpu=False)

# Từ ảnh
plate_img = cv2.imread('plate.jpg')
result = ocr.recognize_and_format(plate_img)
print(f"Kết quả: {result['formatted_text']}")
print(f"Độ tin cậy: {result['confidence']:.2%}")
```

## 🐛 Xử Lý Lỗi Thường Gặp

| Lỗi | Giải Pháp |
|-----|---------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| CUDA out of memory | `use_gpu=False` trong config |
| Độ chính xác thấp | Cải thiện chất lượng ảnh (sáng, tương phản) |
| Không nhận dạng biển số | Kiểm tra kích thước biển số trong ảnh |

## 📚 Tài Liệu & Tham Khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)

## 📦 Dependencies

| Thư Viện | Phiên Bản | Mục Đích |
|---------|---------|---------|
| `opencv-python` | 4.8.1+ | Xử lý ảnh |
| `easyocr` | 1.7.2+ | Nhận dạng ký tự |
| `ultralytics` | 8.0+ | YOLOv8 detection |
| `numpy` | 1.24+ | Tính toán số |
| `torch` | 2.0+ | Backend OCR/YOLO |
| `Flask` | 3.0+ | Web interface (tùy chọn) |

## 📈 Lịch Sử Phiên Bản

- **v2.1** (May 2026): Cải thiện OCR accuracy 88%, tối ưu A/4, 5/6 detection
- **v2.0** (April 2026): Thêm Vietnamese OCR Engine
- **v1.0**: Phiên bản gốc

## 💡 Mẹo & Best Practices

1. **Chất lượng ảnh**: Ảnh càng rõ ràng, độ chính xác càng cao
2. **Góc chụp**: Chụp biển số thẳng (tránh góc xiên quá)
3. **Ánh sáng**: Tránh chối sáng hoặc quá tối
4. **Kích thước**: Biển số nên chiếm ≥ 30% kích thước ảnh

---

**Cập nhật**: May 2026 | **License**: MIT
