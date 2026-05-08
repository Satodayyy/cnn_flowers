"""
=============================================================
  CNN Image Classification - Flowers Recognition
  Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
  Reference: https://www.tensorflow.org/tutorials/images/classification
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  1. CẤU HÌNH TOÀN CỤC
# ─────────────────────────────────────────
CONFIG = {
    "data_dir"    : "flowers",          # thư mục chứa dataset Kaggle
    "img_size"    : (150, 150),         # kích thước ảnh đầu vào
    "batch_size"  : 32,
    "epochs"      : 50,
    "learning_rate": 1e-3,
    "seed"        : 42,
    "val_split"   : 0.2,
    "num_classes" : 5,                  # daisy, dandelion, rose, sunflower, tulip
    "class_names" : ["daisy", "dandelion", "rose", "sunflower", "tulip"],
    "save_dir"    : "outputs",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────────────────
#  2. TẢI & TIỀN XỬ LÝ DỮ LIỆU
# ─────────────────────────────────────────
def load_datasets():
    """
    Tải dữ liệu từ thư mục theo cấu trúc:
        flowers/
            daisy/      *.jpg
            dandelion/  *.jpg
            rose/       *.jpg
            sunflower/  *.jpg
            tulip/      *.jpg
    """
    print(" Đang tải dữ liệu...")

    train_ds = keras.utils.image_dataset_from_directory(
        CONFIG["data_dir"],
        validation_split = CONFIG["val_split"],
        subset           = "training",
        seed             = CONFIG["seed"],
        image_size       = CONFIG["img_size"],
        batch_size       = CONFIG["batch_size"],
        label_mode       = "categorical",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        CONFIG["data_dir"],
        validation_split = CONFIG["val_split"],
        subset           = "validation",
        seed             = CONFIG["seed"],
        image_size       = CONFIG["img_size"],
        batch_size       = CONFIG["batch_size"],
        label_mode       = "categorical",
    )

    # Tối ưu hoá pipeline đọc dữ liệu
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    print(f"    Train: {len(train_ds)} batch | Val: {len(val_ds)} batch")
    return train_ds, val_ds


def visualize_samples(train_ds):
    """Hiển thị 15 ảnh mẫu từ tập huấn luyện."""
    images, labels = next(iter(train_ds))
    class_names = CONFIG["class_names"]

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(" Mẫu Ảnh Tập Huấn Luyện", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype("uint8"))
        label_idx = np.argmax(labels[i].numpy())
        ax.set_title(class_names[label_idx], fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{CONFIG['save_dir']}/sample_images.png", dpi=150)
    plt.close()
    print(f"    Lưu ảnh mẫu → {CONFIG['save_dir']}/sample_images.png")


# ─────────────────────────────────────────
#  3. DATA AUGMENTATION
# ─────────────────────────────────────────
def build_augmentation():
    """
    Tạo lớp tăng cường dữ liệu để giảm overfitting.
    Các kỹ thuật: flip, rotation, zoom, contrast.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


# ─────────────────────────────────────────
#  4. XÂY DỰNG MÔ HÌNH CNN
# ─────────────────────────────────────────
def build_cnn_model(augmentation_layer):
    """
    Kiến trúc CNN tuỳ chỉnh:
      - 4 khối Conv2D + BatchNorm + MaxPooling
      - GlobalAveragePooling thay vì Flatten (ít tham số hơn)
      - Dense layers với Dropout
      - L2 Regularization chống overfitting

    Input shape: (150, 150, 3)
    Output: softmax 5 lớp hoa
    """
    inputs = keras.Input(shape=(*CONFIG["img_size"], 3), name="input_image")

    # ── Tiền xử lý ──────────────────────────────────
    x = augmentation_layer(inputs)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)   # chuẩn hoá [0,1]

    # ── Khối 1: 32 filters ───────────────────────────
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Khối 2: 64 filters ───────────────────────────
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Khối 3: 128 filters ──────────────────────────
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv6")(x)
    x = layers.BatchNormalization(name="bn6")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # ── Khối 4: 256 filters ──────────────────────────
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv7")(x)
    x = layers.BatchNormalization(name="bn7")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool4")(x)
    x = layers.Dropout(0.3, name="drop4")(x)

    # ── Classifier Head ──────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="fc1")(x)
    x = layers.BatchNormalization(name="bn8")(x)
    x = layers.Dropout(0.5, name="drop5")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="fc2")(x)
    x = layers.Dropout(0.4, name="drop6")(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="FlowerCNN")
    return model


# ─────────────────────────────────────────
#  5. TRANSFER LEARNING (MobileNetV2)
# ─────────────────────────────────────────
def build_transfer_model():
    """
    Mô hình Transfer Learning với MobileNetV2 (pretrained trên ImageNet).
    Phù hợp khi dataset nhỏ hoặc muốn đạt accuracy cao hơn nhanh hơn.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape = (*CONFIG["img_size"], 3),
        include_top = False,
        weights     = "imagenet",
    )
    base_model.trainable = False   # đóng băng trọng số ban đầu

    inputs = keras.Input(shape=(*CONFIG["img_size"], 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="FlowerMobileNetV2")
    return model, base_model


# ─────────────────────────────────────────
#  6. COMPILE & CALLBACKS
# ─────────────────────────────────────────
def compile_model(model):
    model.compile(
        optimizer = keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model


def get_callbacks(model_name="cnn"):
    """
    Callbacks:
    - EarlyStopping: dừng sớm nếu không cải thiện
    - ReduceLROnPlateau: giảm LR khi plateau
    - ModelCheckpoint: lưu model tốt nhất
    """
    return [
        EarlyStopping(
            monitor   = "val_accuracy",
            patience  = 10,
            restore_best_weights = True,
            verbose   = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.3,
            patience = 5,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath = f"{CONFIG['save_dir']}/best_{model_name}.keras",
            monitor  = "val_accuracy",
            save_best_only = True,
            verbose  = 1,
        ),
    ]


# ─────────────────────────────────────────
#  7. VISUALIZE KẾT QUẢ TRAINING
# ─────────────────────────────────────────
def plot_training_history(history, model_name="cnn"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f" Lịch Sử Huấn Luyện — {model_name.upper()}",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy",     "val_accuracy",     "Accuracy",     "#2196F3"),
        ("loss",         "val_loss",          "Loss",         "#F44336"),
        ("top2_acc",     "val_top2_acc",      "Top-2 Acc",    "#4CAF50"),
    ]
    for ax, (train_key, val_key, title, color) in zip(axes, metrics):
        if train_key in history.history:
            epochs = range(1, len(history.history[train_key]) + 1)
            ax.plot(epochs, history.history[train_key],
                    color=color, linewidth=2, label="Train")
            if val_key in history.history:
                ax.plot(epochs, history.history[val_key],
                        color=color, linewidth=2, linestyle="--",
                        alpha=0.7, label="Validation")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim(1, len(epochs))

    plt.tight_layout()
    path = f"{CONFIG['save_dir']}/training_history_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Lưu biểu đồ lịch sử → {path}")


# ─────────────────────────────────────────
#  8. ĐÁNH GIÁ MÔ HÌNH
# ─────────────────────────────────────────
def evaluate_model(model, val_ds, model_name="cnn"):
    print(f"\n Đánh giá mô hình {model_name}...")

    # Thu thập dự đoán
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_names = CONFIG["class_names"]

    # ── Classification Report ────────────────────────
    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Confusion Matrix ─────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Confusion Matrix — {model_name.upper()}", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Số lượng tuyệt đối", "Tỷ lệ phần trăm (%)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=30)

    plt.tight_layout()
    path = f"{CONFIG['save_dir']}/confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Lưu confusion matrix → {path}")

    return y_true, y_pred


# ─────────────────────────────────────────
#  9. DỰ ĐOÁN ẢNH MỚI
# ─────────────────────────────────────────
def predict_image(model, img_path):
    """
    Dự đoán loài hoa từ một ảnh đơn lẻ.
    Trả về tên lớp và xác suất dự đoán.
    """
    img = keras.utils.load_img(img_path, target_size=CONFIG["img_size"])
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)   # thêm batch dimension

    predictions = model.predict(img_array, verbose=0)
    proba = predictions[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    print(f"\n Dự đoán cho: {os.path.basename(img_path)}")
    for i, idx in enumerate(top3_idx, 1):
        print(f"   Top-{i}: {CONFIG['class_names'][idx]:12s}  {proba[idx]*100:.1f}%")

    return CONFIG["class_names"][np.argmax(proba)], float(np.max(proba))


def visualize_predictions(model, val_ds):
    """Hiển thị 12 ảnh với nhãn thật và dự đoán."""
    images, labels = next(iter(val_ds))
    preds   = model.predict(images[:12], verbose=0)
    class_names = CONFIG["class_names"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("🔮 Kết Quả Dự Đoán", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype("uint8"))
        true_lbl = class_names[np.argmax(labels[i])]
        pred_lbl = class_names[np.argmax(preds[i])]
        conf     = float(np.max(preds[i])) * 100
        color    = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f" {pred_lbl} ({conf:.0f}%)" if true_lbl == pred_lbl
                     else f" {pred_lbl} ({conf:.0f}%)\n[true: {true_lbl}]",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    path = f"{CONFIG['save_dir']}/predictions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Lưu kết quả dự đoán → {path}")


# ─────────────────────────────────────────
#  10. FINE-TUNING (Transfer Learning)
# ─────────────────────────────────────────
def fine_tune(model, base_model, train_ds, val_ds):
    """
    Bước 2 của Transfer Learning: mở khoá các layer cuối của base model
    và huấn luyện tiếp với learning rate nhỏ hơn.
    """
    print("\n Fine-tuning MobileNetV2...")
    base_model.trainable = True

    # Chỉ train 30 layer cuối
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = keras.optimizers.Adam(1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    history_ft = model.fit(
        train_ds,
        epochs          = 20,
        validation_data = val_ds,
        callbacks       = get_callbacks("mobilenet_ft"),
    )
    return history_ft


# ─────────────────────────────────────────
#  11. MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("   CNN Flower Classification — TensorFlow/Keras")
    print("=" * 60)

    # Kiểm tra GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f" GPU: {[g.name for g in gpus] if gpus else 'Không có GPU, dùng CPU'}\n")

    # ── Tải dữ liệu ─────────────────────────────────
    train_ds, val_ds = load_datasets()
    visualize_samples(train_ds)

    # ── Mô hình CNN tuỳ chỉnh ───────────────────────
    print("\n  Xây dựng CNN tuỳ chỉnh...")
    augmentation = build_augmentation()
    cnn_model    = build_cnn_model(augmentation)
    cnn_model    = compile_model(cnn_model)
    cnn_model.summary()

    print(f"\n Huấn luyện CNN ({CONFIG['epochs']} epochs max)...")
    history_cnn = cnn_model.fit(
        train_ds,
        epochs          = CONFIG["epochs"],
        validation_data = val_ds,
        callbacks       = get_callbacks("cnn"),
    )

    plot_training_history(history_cnn, "cnn")
    evaluate_model(cnn_model, val_ds, "cnn")
    visualize_predictions(cnn_model, val_ds)

    # ── Transfer Learning MobileNetV2 ───────────────
    print("\n  Xây dựng Transfer Learning (MobileNetV2)...")
    tl_model, base_model = build_transfer_model()
    tl_model = compile_model(tl_model)
    tl_model.summary()

    print("\n Huấn luyện Transfer Learning...")
    history_tl = tl_model.fit(
        train_ds,
        epochs          = 30,
        validation_data = val_ds,
        callbacks       = get_callbacks("mobilenet"),
    )

    plot_training_history(history_tl, "mobilenet")
    evaluate_model(tl_model, val_ds, "mobilenet")

    # Fine-tuning
    history_ft = fine_tune(tl_model, base_model, train_ds, val_ds)
    plot_training_history(history_ft, "mobilenet_finetune")
    evaluate_model(tl_model, val_ds, "mobilenet_finetune")

    print(f"\n Hoàn tất! Kết quả lưu tại: ./{CONFIG['save_dir']}/")


if __name__ == "__main__":
    main()
