#  CNN Flower Classification
**Phân loại 5 loài hoa bằng Mạng Nơ-ron Tích Chập (CNN)**  
TensorFlow · Keras · MobileNetV2 Transfer Learning

---

##  Mục Lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Kiến trúc mô hình](#kiến-trúc-mô-hình)
4. [Kỹ thuật sử dụng](#kỹ-thuật-sử-dụng)
5. [Cài đặt & Chạy](#cài-đặt--chạy)
6. [Kết quả](#kết-quả)
7. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
8. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Bài toán **Image Classification** là một trong những ứng dụng nền tảng của thị giác máy tính (Computer Vision). Dự án này xây dựng và so sánh hai phương pháp:

| Phương pháp | Mô tả |
|---|---|
| **Custom CNN** | CNN 4 khối tự thiết kế từ đầu |
| **Transfer Learning** | MobileNetV2 pretrained ImageNet + Fine-tuning |

---

## Dataset

**Flowers Recognition** — [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

| Thuộc tính | Giá trị |
|---|---|
| Số ảnh | ~4,317 ảnh |
| Số lớp | 5 (daisy, dandelion, rose, sunflower, tulip) |
| Kích thước ảnh | Đa dạng, resize về 150×150 |
| Định dạng | JPEG/PNG RGB |

### Tải Dataset

```bash
# Cách 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip -d flowers

# Cách 2: Tải thủ công từ Kaggle rồi giải nén vào thư mục flowers/
```

Cấu trúc thư mục sau khi giải nén:
```
flowers/
├── daisy/       (633 ảnh)
├── dandelion/   (898 ảnh)
├── rose/        (641 ảnh)
├── sunflower/   (699 ảnh)
└── tulip/       (799 ảnh)
```

---

## Kiến trúc Mô Hình

### 1. Custom CNN

```
Input (150×150×3)
  │
  ├── [Augmentation] RandomFlip, RandomRotation, RandomZoom
  ├── Rescaling (÷255)
  │
  ├── Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  ├── Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├── Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.30)
  ├── Block 4: Conv2D(256) → BN → MaxPool → Dropout(0.30)
  │
  ├── GlobalAveragePooling2D
  ├── Dense(512, relu) → BN → Dropout(0.5)
  ├── Dense(256, relu) → Dropout(0.4)
  └── Dense(5, softmax)
```

**Tổng tham số:** ~3.2M

### 2. Transfer Learning — MobileNetV2

```
Input (150×150×3)
  │
  ├── MobileNetV2 (pretrained ImageNet, frozen)   ← Feature Extractor
  ├── GlobalAveragePooling2D
  ├── Dropout(0.3)
  ├── Dense(256, relu)
  ├── Dropout(0.3)
  └── Dense(5, softmax)

[Fine-tuning] Mở khoá 30 layer cuối, LR = 1e-5
```

---

## Kỹ Thuật Sử Dụng

### Data Augmentation
Tăng cường dữ liệu để mô hình tổng quát hoá tốt hơn:

```python
RandomFlip("horizontal_and_vertical")
RandomRotation(0.2)        # xoay ±20°
RandomZoom(0.15)           # zoom ±15%
RandomContrast(0.1)        # thay đổi độ tương phản
RandomTranslation(0.1, 0.1) # dịch chuyển ±10%
```

### Regularization
- **L2 Regularization** (λ=1e-4) trên tất cả Conv & Dense layers
- **BatchNormalization** sau mỗi Conv layer
- **Dropout**: 0.25 → 0.30 → 0.50 (tăng dần)

### Callbacks
- **EarlyStopping**: dừng nếu val_accuracy không cải thiện sau 10 epoch
- **ReduceLROnPlateau**: giảm LR ×0.3 nếu val_loss plateau sau 5 epoch
- **ModelCheckpoint**: lưu model tốt nhất (theo val_accuracy)

### Optimizer & Loss
```python
optimizer = Adam(lr=1e-3)         # Custom CNN
optimizer = Adam(lr=1e-5)         # Fine-tuning
loss      = categorical_crossentropy
metrics   = [accuracy, top2_accuracy]
```

---

## Cài Đặt & Chạy

### Yêu cầu hệ thống
- Python ≥ 3.9
- TensorFlow ≥ 2.13 (hoặc tensorflow-gpu)
- RAM ≥ 8GB | GPU (tuỳ chọn, tăng tốc đáng kể)

### Cài đặt

```bash
# Clone hoặc copy thư mục dự án
cd cnn_flowers

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv 
source venv\Scripts\activate     # Linux/Mac
# hoặc: venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt
```

### Huấn luyện

```bash
# Đảm bảo dataset đã ở thư mục flowers/
python train.py
```

Kết quả sẽ được lưu vào `outputs/`:
- `best_cnn.keras` — mô hình CNN tốt nhất
- `best_mobilenet.keras` — mô hình TL tốt nhất
- `training_history_*.png` — biểu đồ accuracy/loss
- `confusion_matrix_*.png` — ma trận nhầm lẫn
- `predictions.png` — ảnh dự đoán mẫu

### Chạy App Demo

```bash
# Với Gradio UI
pip install gradio
python app.py

# Với CLI
python app.py path/to/your/flower.jpg
```

### Dự đoán ảnh đơn

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model("outputs/best_mobilenet.keras")
img = keras.utils.load_img("rose.jpg", target_size=(150, 150))
img_array = keras.utils.img_to_array(img)[np.newaxis, ...]
preds = model.predict(img_array)[0]
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
print(f"Dự đoán: {class_names[np.argmax(preds)]} ({np.max(preds)*100:.1f}%)")
```

---

## Kết Quả Kỳ Vọng

| Mô hình | Val Accuracy (kỳ vọng) |
|---|---|
| Custom CNN | 75–82% |
| MobileNetV2 (frozen) | 88–92% |
| MobileNetV2 (fine-tuned) | 90–95% |

> **Ghi chú:** Kết quả thực tế phụ thuộc vào random seed, phần cứng và số epoch.

---

## Cấu Trúc Thư Mục

```
cnn_flowers/
├── train.py          ← Script huấn luyện chính
├── app.py            ← App demo (Gradio / CLI)
├── requirements.txt
├── README.md
├── flowers/          ← Dataset (tải từ Kaggle)
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── outputs/          ← Tự động tạo khi chạy
    ├── best_cnn.keras
    ├── best_mobilenet.keras
    ├── training_history_cnn.png
    ├── confusion_matrix_cnn.png
    └── predictions.png
```

---

## Tài Liệu Tham Khảo

1. **TensorFlow Image Classification Tutorial**  
   https://www.tensorflow.org/tutorials/images/classification

2. **Flowers Recognition Dataset — Kaggle**  
   https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

3. **MobileNetV2 Paper** — Sandler et al., 2018  
   https://arxiv.org/abs/1801.04381

4. **Keras Documentation**  
   https://keras.io/api/

---


