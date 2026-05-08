#  CNN Flower Classification — TensorFlow/Keras

##  Tổng quan
Dự án thực hiện phân loại ảnh hoa bằng:

- CNN tự xây dựng (Custom CNN)
- Transfer Learning với MobileNetV2
- Fine-tuning MobileNetV2

Dataset gồm **5 loại hoa**:

| Class | Label |
|---|---|
|  Daisy | daisy |
|  Dandelion | dandelion |
|  Rose | rose |
|  Sunflower | sunflower |
|  Tulip | tulip |

---

#  Thông tin huấn luyện

## Dataset

| Thành phần | Số lượng |
|---|---|
| Tổng ảnh | 4317 |
| Train | 3454 |
| Validation | 863 |
| Số lớp | 5 |

---

#  1. Custom CNN

##  Kiến trúc mô hình

### Các thành phần chính

- Data Augmentation
- Batch Normalization
- Dropout
- GlobalAveragePooling2D
- Dense Layers

### Tổng số tham số

| Loại | Giá trị |
|---|---|
| Total Params | 851,237 |
| Trainable Params | 848,805 |
| Non-trainable Params | 2,432 |

---

##  Quá trình huấn luyện

| Epoch tốt nhất | Validation Accuracy |
|---|---|
| Epoch 38 | **77.87%** |

### Learning Rate Schedule

| Epoch | Learning Rate |
|---|---|
| 1 → 30 | 0.001 |
| 31 | 0.0003 |
| 43 | 0.00009 |
| 48 | 0.000027 |

### Early Stopping

- Dừng tại epoch 48
- Restore weight từ epoch tốt nhất: **38**

---

##  Kết quả đánh giá

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Daisy | 0.87 | 0.76 | 0.81 |
| Dandelion | 0.78 | 0.84 | 0.81 |
| Rose | 0.64 | 0.66 | 0.65 |
| Sunflower | 0.83 | 0.90 | 0.86 |
| Tulip | 0.77 | 0.71 | 0.74 |

---

###  Accuracy

```txt
Accuracy: 78%
```

---

#  2. Transfer Learning — MobileNetV2

##  Kiến trúc mô hình

### Backbone

- MobileNetV2 (pretrained ImageNet)

### Classifier Head

- GlobalAveragePooling2D
- Dense(256)
- Dropout
- Dense(5)

---

##  Thông số mô hình

| Loại | Giá trị |
|---|---|
| Total Params | 2,587,205 |
| Trainable Params | 329,221 |
| Non-trainable Params | 2,257,984 |

---

##  Huấn luyện

| Epoch tốt nhất | Validation Accuracy |
|---|---|
| Epoch 9 | **89.46%** |

### Learning Rate

| Epoch | Learning Rate |
|---|---|
| 1 → 8 | 0.001 |
| 9 → 14 | 0.0003 |
| 15 → 19 | 0.00009 |

### Early Stopping

- Dừng tại epoch 19
- Restore weight từ epoch tốt nhất: **9**

---

##  Kết quả đánh giá

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Daisy | 0.89 | 0.87 | 0.88 |
| Dandelion | 0.92 | 0.91 | 0.92 |
| Rose | 0.81 | 0.88 | 0.84 |
| Sunflower | 0.92 | 0.92 | 0.92 |
| Tulip | 0.92 | 0.88 | 0.90 |

---

###  Accuracy

```txt
Accuracy: 89%
```

---

#  3. Fine-tuning MobileNetV2

##  Fine-tuning

- Unfreeze một phần MobileNetV2
- Learning rate nhỏ để tránh phá hỏng pretrained weights

### Learning Rate

| Epoch | Learning Rate |
|---|---|
| 1 → 16 | 1e-5 |
| 17 → 18 | 3e-6 |

---

##  Kết quả tốt nhất

| Epoch tốt nhất | Validation Accuracy |
|---|---|
| Epoch 8 | **89.92%** |

---

##  Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Daisy | 0.89 | 0.88 | 0.88 |
| Dandelion | 0.90 | 0.94 | 0.92 |
| Rose | 0.86 | 0.87 | 0.87 |
| Sunflower | 0.92 | 0.91 | 0.91 |
| Tulip | 0.91 | 0.89 | 0.90 |

---

###  Accuracy

```txt
Accuracy: 90%
```

---

#  So sánh mô hình

| Model | Accuracy |
|---|---|
| Custom CNN | 78% |
| MobileNetV2 | 89% |
| MobileNetV2 Fine-tune | **90%** |

---

#  Các file output

```txt
outputs/
├── sample_images.png
├── best_cnn.keras
├── best_mobilenet.keras
├── best_mobilenet_ft.keras
├── training_history_cnn.png
├── training_history_mobilenet.png
├── training_history_mobilenet_finetune.png
├── confusion_matrix_cnn.png
├── confusion_matrix_mobilenet.png
├── confusion_matrix_mobilenet_finetune.png
└── predictions.png
```

---

#  Kết luận

- CNN tự xây dựng đạt accuracy khoảng **78%**
- Transfer Learning với MobileNetV2 cải thiện mạnh lên **89%**
- Fine-tuning tiếp tục nâng hiệu năng lên gần **90%**
- MobileNetV2 cho khả năng học đặc trưng ảnh tốt hơn đáng kể so với CNN custom

---

#  Công nghệ sử dụng

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

#  Kết quả cuối cùng

```txt
Hoàn tất! Kết quả lưu tại: ./outputs/
```
