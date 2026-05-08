"""
=============================================================
  App Demo: Phân loại hoa bằng CNN đã huấn luyện
  Chạy: python app.py
=============================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Cấu hình ────────────────────────────────────────────────
IMG_SIZE    = (150, 150)
CLASS_NAMES = ["🌼 Daisy", "🌻 Dandelion", "🌹 Rose", "🌻 Sunflower", "🌷 Tulip"]
MODEL_PATH  = "outputs/best_mobilenet.keras"    # hoặc best_cnn.keras

# ── Tải mô hình ─────────────────────────────────────────────
print(f" Tải mô hình từ {MODEL_PATH} ...")
model = keras.models.load_model(MODEL_PATH)
print(" Tải mô hình thành công!")


def predict_flower(image):
    """Nhận PIL Image, trả về dict xác suất từng lớp."""
    img_array = tf.image.resize(image, IMG_SIZE)
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array, verbose=0)[0]
    return {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}


# ── Chạy với Gradio nếu có ──────────────────────────────────
try:
    import gradio as gr

    demo = gr.Interface(
        fn           = predict_flower,
        inputs       = gr.Image(type="pil", label="📷 Tải ảnh hoa lên"),
        outputs      = gr.Label(num_top_classes=5, label="🌸 Dự Đoán"),
        title        = "🌺 Flower CNN Classifier",
        description  = "Phân loại 5 loài hoa: Daisy, Dandelion, Rose, Sunflower, Tulip",
        examples     = [],
        theme        = gr.themes.Soft(),
    )

    if __name__ == "__main__":
        demo.launch(share=False, server_port=7860)

except ImportError:
    # Nếu không có Gradio, chạy demo đơn giản qua CLI
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app.py <đường_dẫn_ảnh>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)

    result = predict_flower(img_array)
    print("\n🌸 Kết quả dự đoán:")
    for name, prob in sorted(result.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:20s} {bar:30s} {prob*100:5.1f}%")
