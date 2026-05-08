# cnn_flowers
Kết quả chạy train.py:
============================================================
   CNN Flower Classification — TensorFlow/Keras
============================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
 GPU: Không có GPU, dùng CPU

 Đang tải dữ liệu...
Found 4317 files belonging to 5 classes.
Using 3454 files for training.
I0000 00:00:1778205576.697497   18460 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Found 4317 files belonging to 5 classes.
Using 863 files for validation.
    Train: 108 batch | Val: 27 batch
    Lưu ảnh mẫu → outputs/sample_images.png

  Xây dựng CNN tuỳ chỉnh...
Model: "FlowerCNN"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_image (InputLayer)             │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ data_augmentation (Sequential)       │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling (Rescaling)                │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1 (Conv2D)                       │ (None, 150, 150, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn1 (BatchNormalization)             │ (None, 150, 150, 32)        │             128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2 (Conv2D)                       │ (None, 150, 150, 32)        │           9,248 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn2 (BatchNormalization)             │ (None, 150, 150, 32)        │             128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool1 (MaxPooling2D)                 │ (None, 75, 75, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop1 (Dropout)                      │ (None, 75, 75, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv3 (Conv2D)                       │ (None, 75, 75, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn3 (BatchNormalization)             │ (None, 75, 75, 64)          │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv4 (Conv2D)                       │ (None, 75, 75, 64)          │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn4 (BatchNormalization)             │ (None, 75, 75, 64)          │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool2 (MaxPooling2D)                 │ (None, 37, 37, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop2 (Dropout)                      │ (None, 37, 37, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv5 (Conv2D)                       │ (None, 37, 37, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn5 (BatchNormalization)             │ (None, 37, 37, 128)         │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv6 (Conv2D)                       │ (None, 37, 37, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn6 (BatchNormalization)             │ (None, 37, 37, 128)         │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool3 (MaxPooling2D)                 │ (None, 18, 18, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop3 (Dropout)                      │ (None, 18, 18, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv7 (Conv2D)                       │ (None, 18, 18, 256)         │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn7 (BatchNormalization)             │ (None, 18, 18, 256)         │           1,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool4 (MaxPooling2D)                 │ (None, 9, 9, 256)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop4 (Dropout)                      │ (None, 9, 9, 256)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ gap (GlobalAveragePooling2D)         │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ fc1 (Dense)                          │ (None, 512)                 │         131,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn8 (BatchNormalization)             │ (None, 512)                 │           2,048 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop5 (Dropout)                      │ (None, 512)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ fc2 (Dense)                          │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ drop6 (Dropout)                      │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output (Dense)                       │ (None, 5)                   │           1,285 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 851,237 (3.25 MB)
 Trainable params: 848,805 (3.24 MB)
 Non-trainable params: 2,432 (9.50 KB)

 Huấn luyện CNN (50 epochs max)...
Epoch 1/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.3167 - loss: 2.2572 - top2_acc: 0.5978
Epoch 1: val_accuracy improved from None to 0.23870, saving model to outputs/best_cnn.keras

Epoch 1: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 364s 3s/step - accuracy: 0.3674 - loss: 2.0509 - top2_acc: 0.6514 - val_accuracy: 0.2387 - val_loss: 2.5750 - val_top2_acc: 0.4056 - learning_rate: 0.0010
Epoch 2/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.4581 - loss: 1.7242 - top2_acc: 0.7410
Epoch 2: val_accuracy did not improve from 0.23870
108/108 ━━━━━━━━━━━━━━━━━━━━ 131s 1s/step - accuracy: 0.4545 - loss: 1.6859 - top2_acc: 0.7339 - val_accuracy: 0.2387 - val_loss: 2.3821 - val_top2_acc: 0.4264 - learning_rate: 0.0010
Epoch 3/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.4743 - loss: 1.5373 - top2_acc: 0.7375
Epoch 3: val_accuracy improved from 0.23870 to 0.24797, saving model to outputs/best_cnn.keras

Epoch 3: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 130s 1s/step - accuracy: 0.4954 - loss: 1.4736 - top2_acc: 0.7664 - val_accuracy: 0.2480 - val_loss: 2.3582 - val_top2_acc: 0.5840 - learning_rate: 0.0010
Epoch 4/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.5300 - loss: 1.3593 - top2_acc: 0.7972
Epoch 4: val_accuracy improved from 0.24797 to 0.35921, saving model to outputs/best_cnn.keras

Epoch 4: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 134s 1s/step - accuracy: 0.5402 - loss: 1.3175 - top2_acc: 0.7988 - val_accuracy: 0.3592 - val_loss: 1.7832 - val_top2_acc: 0.6408 - learning_rate: 0.0010
Epoch 5/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.5707 - loss: 1.2604 - top2_acc: 0.8185
Epoch 5: val_accuracy improved from 0.35921 to 0.49363, saving model to outputs/best_cnn.keras

Epoch 5: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 129s 1s/step - accuracy: 0.5704 - loss: 1.2540 - top2_acc: 0.8191 - val_accuracy: 0.4936 - val_loss: 1.2821 - val_top2_acc: 0.8146 - learning_rate: 0.0010
Epoch 6/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.5895 - loss: 1.2110 - top2_acc: 0.8221
Epoch 6: val_accuracy improved from 0.49363 to 0.51912, saving model to outputs/best_cnn.keras

Epoch 6: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 130s 1s/step - accuracy: 0.5816 - loss: 1.2196 - top2_acc: 0.8277 - val_accuracy: 0.5191 - val_loss: 1.3154 - val_top2_acc: 0.8343 - learning_rate: 0.0010
Epoch 7/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6114 - loss: 1.1241 - top2_acc: 0.8491
Epoch 7: val_accuracy improved from 0.51912 to 0.62109, saving model to outputs/best_cnn.keras

Epoch 7: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 132s 1s/step - accuracy: 0.6094 - loss: 1.1242 - top2_acc: 0.8384 - val_accuracy: 0.6211 - val_loss: 1.0594 - val_top2_acc: 0.8378 - learning_rate: 0.0010
Epoch 8/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6171 - loss: 1.1369 - top2_acc: 0.8404
Epoch 8: val_accuracy did not improve from 0.62109
108/108 ━━━━━━━━━━━━━━━━━━━━ 129s 1s/step - accuracy: 0.6141 - loss: 1.1393 - top2_acc: 0.8474 - val_accuracy: 0.5805 - val_loss: 1.1760 - val_top2_acc: 0.8644 - learning_rate: 0.0010
Epoch 9/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6352 - loss: 1.1046 - top2_acc: 0.8538
Epoch 9: val_accuracy did not improve from 0.62109
108/108 ━━━━━━━━━━━━━━━━━━━━ 129s 1s/step - accuracy: 0.6309 - loss: 1.0939 - top2_acc: 0.8471 - val_accuracy: 0.5574 - val_loss: 1.3313 - val_top2_acc: 0.8204 - learning_rate: 0.0010
Epoch 10/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6351 - loss: 1.0897 - top2_acc: 0.8501
Epoch 10: val_accuracy did not improve from 0.62109
108/108 ━━━━━━━━━━━━━━━━━━━━ 129s 1s/step - accuracy: 0.6407 - loss: 1.0696 - top2_acc: 0.8547 - val_accuracy: 0.5991 - val_loss: 1.1146 - val_top2_acc: 0.8331 - learning_rate: 0.0010
Epoch 11/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6336 - loss: 1.0605 - top2_acc: 0.8588
Epoch 11: val_accuracy improved from 0.62109 to 0.64542, saving model to outputs/best_cnn.keras

Epoch 11: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 130s 1s/step - accuracy: 0.6422 - loss: 1.0519 - top2_acc: 0.8628 - val_accuracy: 0.6454 - val_loss: 1.0591 - val_top2_acc: 0.8552 - learning_rate: 0.0010
Epoch 12/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6498 - loss: 1.0205 - top2_acc: 0.8637
Epoch 12: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 129s 1s/step - accuracy: 0.6471 - loss: 1.0282 - top2_acc: 0.8625 - val_accuracy: 0.6292 - val_loss: 1.0264 - val_top2_acc: 0.8528 - learning_rate: 0.0010
Epoch 13/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6502 - loss: 1.0226 - top2_acc: 0.8577
Epoch 13: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 157s 1s/step - accuracy: 0.6543 - loss: 1.0032 - top2_acc: 0.8686 - val_accuracy: 0.6269 - val_loss: 1.0743 - val_top2_acc: 0.8691 - learning_rate: 0.0010
Epoch 14/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.6755 - loss: 0.9840 - top2_acc: 0.8642
Epoch 14: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 384s 4s/step - accuracy: 0.6757 - loss: 0.9732 - top2_acc: 0.8703 - val_accuracy: 0.6246 - val_loss: 1.2261 - val_top2_acc: 0.8250 - learning_rate: 0.0010
Epoch 15/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6724 - loss: 0.9795 - top2_acc: 0.8725
Epoch 15: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 414s 4s/step - accuracy: 0.6618 - loss: 0.9765 - top2_acc: 0.8738 - val_accuracy: 0.6292 - val_loss: 1.0693 - val_top2_acc: 0.8331 - learning_rate: 0.0010
Epoch 16/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6966 - loss: 0.9210 - top2_acc: 0.8875
Epoch 16: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 439s 4s/step - accuracy: 0.6821 - loss: 0.9584 - top2_acc: 0.8798 - val_accuracy: 0.6211 - val_loss: 1.1723 - val_top2_acc: 0.8273 - learning_rate: 0.0010
Epoch 17/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6715 - loss: 0.9490 - top2_acc: 0.8768
Epoch 17: val_accuracy did not improve from 0.64542
108/108 ━━━━━━━━━━━━━━━━━━━━ 425s 4s/step - accuracy: 0.6844 - loss: 0.9426 - top2_acc: 0.8798 - val_accuracy: 0.6246 - val_loss: 1.0021 - val_top2_acc: 0.8772 - learning_rate: 0.0010
Epoch 18/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7031 - loss: 0.9069 - top2_acc: 0.8804
Epoch 18: val_accuracy improved from 0.64542 to 0.65933, saving model to outputs/best_cnn.keras

Epoch 18: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 430s 4s/step - accuracy: 0.6925 - loss: 0.9244 - top2_acc: 0.8827 - val_accuracy: 0.6593 - val_loss: 0.9894 - val_top2_acc: 0.8656 - learning_rate: 0.0010
Epoch 19/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6939 - loss: 0.9142 - top2_acc: 0.8924
Epoch 19: val_accuracy did not improve from 0.65933
108/108 ━━━━━━━━━━━━━━━━━━━━ 413s 4s/step - accuracy: 0.6856 - loss: 0.9381 - top2_acc: 0.8880 - val_accuracy: 0.6269 - val_loss: 1.0226 - val_top2_acc: 0.8749 - learning_rate: 0.0010
Epoch 20/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6782 - loss: 0.9412 - top2_acc: 0.8776
Epoch 20: val_accuracy did not improve from 0.65933
108/108 ━━━━━━━━━━━━━━━━━━━━ 412s 4s/step - accuracy: 0.6888 - loss: 0.9357 - top2_acc: 0.8848 - val_accuracy: 0.6315 - val_loss: 1.0416 - val_top2_acc: 0.8470 - learning_rate: 0.0010
Epoch 21/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.6988 - loss: 0.9232 - top2_acc: 0.8822
Epoch 21: val_accuracy did not improve from 0.65933
108/108 ━━━━━━━━━━━━━━━━━━━━ 400s 4s/step - accuracy: 0.7009 - loss: 0.9164 - top2_acc: 0.8865 - val_accuracy: 0.6559 - val_loss: 0.9676 - val_top2_acc: 0.8714 - learning_rate: 0.0010
Epoch 22/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7186 - loss: 0.8744 - top2_acc: 0.8879
Epoch 22: val_accuracy did not improve from 0.65933
108/108 ━━━━━━━━━━━━━━━━━━━━ 413s 4s/step - accuracy: 0.7030 - loss: 0.8948 - top2_acc: 0.8807 - val_accuracy: 0.6118 - val_loss: 1.0765 - val_top2_acc: 0.8482 - learning_rate: 0.0010
Epoch 23/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7073 - loss: 0.8956 - top2_acc: 0.8909
Epoch 23: val_accuracy improved from 0.65933 to 0.68830, saving model to outputs/best_cnn.keras

Epoch 23: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 405s 4s/step - accuracy: 0.6975 - loss: 0.9079 - top2_acc: 0.8865 - val_accuracy: 0.6883 - val_loss: 0.8820 - val_top2_acc: 0.8876 - learning_rate: 0.0010
Epoch 24/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7198 - loss: 0.8684 - top2_acc: 0.8985
Epoch 24: val_accuracy did not improve from 0.68830
108/108 ━━━━━━━━━━━━━━━━━━━━ 404s 4s/step - accuracy: 0.7134 - loss: 0.8801 - top2_acc: 0.8940 - val_accuracy: 0.6802 - val_loss: 0.9328 - val_top2_acc: 0.8911 - learning_rate: 0.0010
Epoch 25/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7167 - loss: 0.8779 - top2_acc: 0.8915
Epoch 25: val_accuracy did not improve from 0.68830
108/108 ━━━━━━━━━━━━━━━━━━━━ 402s 4s/step - accuracy: 0.7160 - loss: 0.8717 - top2_acc: 0.8949 - val_accuracy: 0.6848 - val_loss: 1.0021 - val_top2_acc: 0.8911 - learning_rate: 0.0010
Epoch 26/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.7086 - loss: 0.8526 - top2_acc: 0.8993
Epoch 26: val_accuracy improved from 0.68830 to 0.71727, saving model to outputs/best_cnn.keras

Epoch 26: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 305s 3s/step - accuracy: 0.7061 - loss: 0.8597 - top2_acc: 0.8955 - val_accuracy: 0.7173 - val_loss: 0.8653 - val_top2_acc: 0.8922 - learning_rate: 0.0010
Epoch 27/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7222 - loss: 0.8242 - top2_acc: 0.9086
Epoch 27: val_accuracy did not improve from 0.71727
108/108 ━━━━━━━━━━━━━━━━━━━━ 414s 4s/step - accuracy: 0.7218 - loss: 0.8462 - top2_acc: 0.9021 - val_accuracy: 0.6593 - val_loss: 1.0035 - val_top2_acc: 0.8760 - learning_rate: 0.0010
Epoch 28/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.7203 - loss: 0.8441 - top2_acc: 0.9002
Epoch 28: val_accuracy did not improve from 0.71727
108/108 ━━━━━━━━━━━━━━━━━━━━ 398s 4s/step - accuracy: 0.7229 - loss: 0.8432 - top2_acc: 0.9021 - val_accuracy: 0.6454 - val_loss: 1.1270 - val_top2_acc: 0.8320 - learning_rate: 0.0010
Epoch 29/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7262 - loss: 0.8312 - top2_acc: 0.9089
Epoch 29: val_accuracy did not improve from 0.71727
108/108 ━━━━━━━━━━━━━━━━━━━━ 402s 4s/step - accuracy: 0.7351 - loss: 0.8264 - top2_acc: 0.9076 - val_accuracy: 0.6883 - val_loss: 0.9860 - val_top2_acc: 0.8737 - learning_rate: 0.0010
Epoch 30/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7395 - loss: 0.8080 - top2_acc: 0.9050
Epoch 30: val_accuracy did not improve from 0.71727
108/108 ━━━━━━━━━━━━━━━━━━━━ 404s 4s/step - accuracy: 0.7334 - loss: 0.8258 - top2_acc: 0.9021 - val_accuracy: 0.6373 - val_loss: 1.0244 - val_top2_acc: 0.8482 - learning_rate: 0.0010
Epoch 31/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7373 - loss: 0.8321 - top2_acc: 0.9051
Epoch 31: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.

Epoch 31: val_accuracy did not improve from 0.71727
108/108 ━━━━━━━━━━━━━━━━━━━━ 411s 4s/step - accuracy: 0.7331 - loss: 0.8292 - top2_acc: 0.9033 - val_accuracy: 0.6419 - val_loss: 1.1076 - val_top2_acc: 0.8470 - learning_rate: 0.0010
Epoch 32/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7616 - loss: 0.7853 - top2_acc: 0.9124
Epoch 32: val_accuracy improved from 0.71727 to 0.74623, saving model to outputs/best_cnn.keras

Epoch 32: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 436s 4s/step - accuracy: 0.7672 - loss: 0.7561 - top2_acc: 0.9221 - val_accuracy: 0.7462 - val_loss: 0.7777 - val_top2_acc: 0.9085 - learning_rate: 3.0000e-04
Epoch 33/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.7905 - loss: 0.6912 - top2_acc: 0.9305
Epoch 33: val_accuracy improved from 0.74623 to 0.76246, saving model to outputs/best_cnn.keras

Epoch 33: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 432s 4s/step - accuracy: 0.7875 - loss: 0.7031 - top2_acc: 0.9273 - val_accuracy: 0.7625 - val_loss: 0.7437 - val_top2_acc: 0.9154 - learning_rate: 3.0000e-04
Epoch 34/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.7886 - loss: 0.6884 - top2_acc: 0.9327
Epoch 34: val_accuracy did not improve from 0.76246
108/108 ━━━━━━━━━━━━━━━━━━━━ 398s 4s/step - accuracy: 0.7860 - loss: 0.6968 - top2_acc: 0.9291 - val_accuracy: 0.7335 - val_loss: 0.8128 - val_top2_acc: 0.9015 - learning_rate: 3.0000e-04
Epoch 35/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8053 - loss: 0.6481 - top2_acc: 0.9357
Epoch 35: val_accuracy did not improve from 0.76246
108/108 ━━━━━━━━━━━━━━━━━━━━ 398s 4s/step - accuracy: 0.7939 - loss: 0.6774 - top2_acc: 0.9302 - val_accuracy: 0.7346 - val_loss: 0.8499 - val_top2_acc: 0.8946 - learning_rate: 3.0000e-04
Epoch 36/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.7870 - loss: 0.6779 - top2_acc: 0.9357
Epoch 36: val_accuracy did not improve from 0.76246
108/108 ━━━━━━━━━━━━━━━━━━━━ 405s 4s/step - accuracy: 0.7927 - loss: 0.6683 - top2_acc: 0.9354 - val_accuracy: 0.7289 - val_loss: 0.8374 - val_top2_acc: 0.8922 - learning_rate: 3.0000e-04
Epoch 37/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.8038 - loss: 0.6409 - top2_acc: 0.9381
Epoch 37: val_accuracy did not improve from 0.76246
108/108 ━━━━━━━━━━━━━━━━━━━━ 402s 4s/step - accuracy: 0.7985 - loss: 0.6593 - top2_acc: 0.9325 - val_accuracy: 0.7555 - val_loss: 0.7536 - val_top2_acc: 0.9200 - learning_rate: 3.0000e-04
Epoch 38/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8002 - loss: 0.6437 - top2_acc: 0.9356
Epoch 38: val_accuracy improved from 0.76246 to 0.77868, saving model to outputs/best_cnn.keras

Epoch 38: finished saving model to outputs/best_cnn.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 435s 4s/step - accuracy: 0.8005 - loss: 0.6445 - top2_acc: 0.9378 - val_accuracy: 0.7787 - val_loss: 0.7109 - val_top2_acc: 0.9143 - learning_rate: 3.0000e-04
Epoch 39/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.8114 - loss: 0.6396 - top2_acc: 0.9376
Epoch 39: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 403s 4s/step - accuracy: 0.8098 - loss: 0.6426 - top2_acc: 0.9378 - val_accuracy: 0.7370 - val_loss: 0.8392 - val_top2_acc: 0.8841 - learning_rate: 3.0000e-04
Epoch 40/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8165 - loss: 0.6088 - top2_acc: 0.9421
Epoch 40: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 397s 4s/step - accuracy: 0.8043 - loss: 0.6259 - top2_acc: 0.9409 - val_accuracy: 0.7555 - val_loss: 0.7778 - val_top2_acc: 0.9108 - learning_rate: 3.0000e-04
Epoch 41/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8037 - loss: 0.6330 - top2_acc: 0.9412
Epoch 41: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 393s 4s/step - accuracy: 0.8121 - loss: 0.6203 - top2_acc: 0.9406 - val_accuracy: 0.7358 - val_loss: 0.8569 - val_top2_acc: 0.9027 - learning_rate: 3.0000e-04
Epoch 42/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8107 - loss: 0.6198 - top2_acc: 0.9465
Epoch 42: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 394s 4s/step - accuracy: 0.8153 - loss: 0.6246 - top2_acc: 0.9433 - val_accuracy: 0.7497 - val_loss: 0.8027 - val_top2_acc: 0.9154 - learning_rate: 3.0000e-04
Epoch 43/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8284 - loss: 0.5944 - top2_acc: 0.9444
Epoch 43: ReduceLROnPlateau reducing learning rate to 9.000000427477062e-05.

Epoch 43: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 397s 4s/step - accuracy: 0.8219 - loss: 0.6039 - top2_acc: 0.9470 - val_accuracy: 0.6941 - val_loss: 0.9864 - val_top2_acc: 0.8714 - learning_rate: 3.0000e-04
Epoch 44/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8219 - loss: 0.5831 - top2_acc: 0.9400
Epoch 44: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 396s 4s/step - accuracy: 0.8263 - loss: 0.5810 - top2_acc: 0.9441 - val_accuracy: 0.7555 - val_loss: 0.7283 - val_top2_acc: 0.9166 - learning_rate: 9.0000e-05
Epoch 45/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8559 - loss: 0.5355 - top2_acc: 0.9548
Epoch 45: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 393s 4s/step - accuracy: 0.8466 - loss: 0.5524 - top2_acc: 0.9508 - val_accuracy: 0.7694 - val_loss: 0.7426 - val_top2_acc: 0.9119 - learning_rate: 9.0000e-05
Epoch 46/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8363 - loss: 0.5586 - top2_acc: 0.9536
Epoch 46: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 392s 4s/step - accuracy: 0.8425 - loss: 0.5406 - top2_acc: 0.9592 - val_accuracy: 0.7590 - val_loss: 0.7582 - val_top2_acc: 0.9154 - learning_rate: 9.0000e-05
Epoch 47/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8536 - loss: 0.5110 - top2_acc: 0.9567
Epoch 47: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 396s 4s/step - accuracy: 0.8457 - loss: 0.5278 - top2_acc: 0.9543 - val_accuracy: 0.7601 - val_loss: 0.7609 - val_top2_acc: 0.9189 - learning_rate: 9.0000e-05
Epoch 48/50
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.8401 - loss: 0.5332 - top2_acc: 0.9523
Epoch 48: ReduceLROnPlateau reducing learning rate to 2.700000040931627e-05.

Epoch 48: val_accuracy did not improve from 0.77868
108/108 ━━━━━━━━━━━━━━━━━━━━ 417s 4s/step - accuracy: 0.8451 - loss: 0.5275 - top2_acc: 0.9525 - val_accuracy: 0.7636 - val_loss: 0.7924 - val_top2_acc: 0.9200 - learning_rate: 9.0000e-05
Epoch 48: early stopping
Restoring model weights from the end of the best epoch: 38.
    Lưu biểu đồ lịch sử → outputs/training_history_cnn.png

 Đánh giá mô hình cnn...

============================================================
 CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

       daisy       0.87      0.76      0.81       144
   dandelion       0.78      0.84      0.81       206
        rose       0.64      0.66      0.65       137
   sunflower       0.83      0.90      0.86       172
       tulip       0.77      0.71      0.74       204

    accuracy                           0.78       863
   macro avg       0.78      0.77      0.77       863
weighted avg       0.78      0.78      0.78       863

    Lưu confusion matrix → outputs/confusion_matrix_cnn.png
    Lưu kết quả dự đoán → outputs/predictions.png

  Xây dựng Transfer Learning (MobileNetV2)...
Model: "FlowerMobileNetV2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)           │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ true_divide (TrueDivide)             │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ subtract (Subtract)                  │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ mobilenetv2_1.00_224 (Functional)    │ (None, 5, 5, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1280)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │           1,285 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,587,205 (9.87 MB)
 Trainable params: 329,221 (1.26 MB)
 Non-trainable params: 2,257,984 (8.61 MB)

 Huấn luyện Transfer Learning...
Epoch 1/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 352ms/step - accuracy: 0.6003 - loss: 1.2083 - top2_acc: 0.7754
Epoch 1: val_accuracy improved from None to 0.84473, saving model to outputs/best_mobilenet.keras

Epoch 1: finished saving model to outputs/best_mobilenet.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 60s 475ms/step - accuracy: 0.7221 - loss: 0.7951 - top2_acc: 0.8720 - val_accuracy: 0.8447 - val_loss: 0.4346 - val_top2_acc: 0.9525 - learning_rate: 0.0010
Epoch 2/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 348ms/step - accuracy: 0.8228 - loss: 0.4544 - top2_acc: 0.9417
Epoch 2: val_accuracy improved from 0.84473 to 0.84705, saving model to outputs/best_mobilenet.keras

Epoch 2: finished saving model to outputs/best_mobilenet.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 48s 442ms/step - accuracy: 0.8208 - loss: 0.4615 - top2_acc: 0.9415 - val_accuracy: 0.8470 - val_loss: 0.4073 - val_top2_acc: 0.9421 - learning_rate: 0.0010
Epoch 3/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 345ms/step - accuracy: 0.8605 - loss: 0.3912 - top2_acc: 0.9583
Epoch 3: val_accuracy improved from 0.84705 to 0.87833, saving model to outputs/best_mobilenet.keras

Epoch 3: finished saving model to outputs/best_mobilenet.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 48s 445ms/step - accuracy: 0.8607 - loss: 0.3792 - top2_acc: 0.9598 - val_accuracy: 0.8783 - val_loss: 0.3617 - val_top2_acc: 0.9571 - learning_rate: 0.0010
Epoch 4/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - accuracy: 0.8865 - loss: 0.3167 - top2_acc: 0.9685
Epoch 4: val_accuracy did not improve from 0.87833
108/108 ━━━━━━━━━━━━━━━━━━━━ 48s 442ms/step - accuracy: 0.8830 - loss: 0.3209 - top2_acc: 0.9696 - val_accuracy: 0.8691 - val_loss: 0.3929 - val_top2_acc: 0.9560 - learning_rate: 0.0010
Epoch 5/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 345ms/step - accuracy: 0.9076 - loss: 0.2549 - top2_acc: 0.9785
Epoch 5: val_accuracy did not improve from 0.87833
108/108 ━━━━━━━━━━━━━━━━━━━━ 47s 432ms/step - accuracy: 0.8981 - loss: 0.2721 - top2_acc: 0.9751 - val_accuracy: 0.8760 - val_loss: 0.3692 - val_top2_acc: 0.9560 - learning_rate: 0.0010
Epoch 6/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 344ms/step - accuracy: 0.9121 - loss: 0.2500 - top2_acc: 0.9837
Epoch 6: val_accuracy did not improve from 0.87833
108/108 ━━━━━━━━━━━━━━━━━━━━ 48s 441ms/step - accuracy: 0.9097 - loss: 0.2415 - top2_acc: 0.9820 - val_accuracy: 0.8749 - val_loss: 0.3619 - val_top2_acc: 0.9606 - learning_rate: 0.0010
Epoch 7/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9160 - loss: 0.2227 - top2_acc: 0.9829
Epoch 7: val_accuracy did not improve from 0.87833
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 416ms/step - accuracy: 0.9140 - loss: 0.2255 - top2_acc: 0.9823 - val_accuracy: 0.8725 - val_loss: 0.3889 - val_top2_acc: 0.9571 - learning_rate: 0.0010
Epoch 8/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 334ms/step - accuracy: 0.9390 - loss: 0.1794 - top2_acc: 0.9865
Epoch 8: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.

Epoch 8: val_accuracy did not improve from 0.87833
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 417ms/step - accuracy: 0.9268 - loss: 0.2002 - top2_acc: 0.9852 - val_accuracy: 0.8714 - val_loss: 0.3898 - val_top2_acc: 0.9583 - learning_rate: 0.0010
Epoch 9/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 331ms/step - accuracy: 0.9480 - loss: 0.1609 - top2_acc: 0.9887
Epoch 9: val_accuracy improved from 0.87833 to 0.89455, saving model to outputs/best_mobilenet.keras

Epoch 9: finished saving model to outputs/best_mobilenet.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 422ms/step - accuracy: 0.9476 - loss: 0.1544 - top2_acc: 0.9907 - val_accuracy: 0.8946 - val_loss: 0.3525 - val_top2_acc: 0.9606 - learning_rate: 3.0000e-04
Epoch 10/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 338ms/step - accuracy: 0.9577 - loss: 0.1217 - top2_acc: 0.9945
Epoch 10: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 421ms/step - accuracy: 0.9557 - loss: 0.1241 - top2_acc: 0.9948 - val_accuracy: 0.8876 - val_loss: 0.3712 - val_top2_acc: 0.9629 - learning_rate: 3.0000e-04
Epoch 11/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9529 - loss: 0.1347 - top2_acc: 0.9929
Epoch 11: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 416ms/step - accuracy: 0.9577 - loss: 0.1251 - top2_acc: 0.9945 - val_accuracy: 0.8899 - val_loss: 0.3625 - val_top2_acc: 0.9676 - learning_rate: 3.0000e-04
Epoch 12/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9615 - loss: 0.1067 - top2_acc: 0.9975
Epoch 12: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 417ms/step - accuracy: 0.9574 - loss: 0.1151 - top2_acc: 0.9962 - val_accuracy: 0.8888 - val_loss: 0.3713 - val_top2_acc: 0.9641 - learning_rate: 3.0000e-04
Epoch 13/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9634 - loss: 0.1093 - top2_acc: 0.9929
Epoch 13: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 46s 429ms/step - accuracy: 0.9650 - loss: 0.1064 - top2_acc: 0.9933 - val_accuracy: 0.8911 - val_loss: 0.3711 - val_top2_acc: 0.9641 - learning_rate: 3.0000e-04
Epoch 14/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 332ms/step - accuracy: 0.9634 - loss: 0.1044 - top2_acc: 0.9958
Epoch 14: ReduceLROnPlateau reducing learning rate to 9.000000427477062e-05.

Epoch 14: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 415ms/step - accuracy: 0.9612 - loss: 0.1084 - top2_acc: 0.9957 - val_accuracy: 0.8922 - val_loss: 0.3698 - val_top2_acc: 0.9618 - learning_rate: 3.0000e-04
Epoch 15/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 331ms/step - accuracy: 0.9677 - loss: 0.0921 - top2_acc: 0.9958
Epoch 15: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 414ms/step - accuracy: 0.9693 - loss: 0.0898 - top2_acc: 0.9957 - val_accuracy: 0.8876 - val_loss: 0.3685 - val_top2_acc: 0.9618 - learning_rate: 9.0000e-05
Epoch 16/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 332ms/step - accuracy: 0.9671 - loss: 0.0971 - top2_acc: 0.9974
Epoch 16: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 414ms/step - accuracy: 0.9673 - loss: 0.0926 - top2_acc: 0.9977 - val_accuracy: 0.8922 - val_loss: 0.3673 - val_top2_acc: 0.9618 - learning_rate: 9.0000e-05
Epoch 17/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9611 - loss: 0.1037 - top2_acc: 0.9949
Epoch 17: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 415ms/step - accuracy: 0.9708 - loss: 0.0917 - top2_acc: 0.9954 - val_accuracy: 0.8934 - val_loss: 0.3645 - val_top2_acc: 0.9641 - learning_rate: 9.0000e-05
Epoch 18/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 333ms/step - accuracy: 0.9687 - loss: 0.0912 - top2_acc: 0.9971
Epoch 18: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 416ms/step - accuracy: 0.9710 - loss: 0.0865 - top2_acc: 0.9971 - val_accuracy: 0.8934 - val_loss: 0.3699 - val_top2_acc: 0.9629 - learning_rate: 9.0000e-05
Epoch 19/30
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 334ms/step - accuracy: 0.9722 - loss: 0.0839 - top2_acc: 0.9962
Epoch 19: ReduceLROnPlateau reducing learning rate to 2.700000040931627e-05.

Epoch 19: val_accuracy did not improve from 0.89455
108/108 ━━━━━━━━━━━━━━━━━━━━ 45s 417ms/step - accuracy: 0.9760 - loss: 0.0798 - top2_acc: 0.9971 - val_accuracy: 0.8922 - val_loss: 0.3769 - val_top2_acc: 0.9641 - learning_rate: 9.0000e-05
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
    Lưu biểu đồ lịch sử → outputs/training_history_mobilenet.png

 Đánh giá mô hình mobilenet...

============================================================
 CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

       daisy       0.89      0.87      0.88       144
   dandelion       0.92      0.91      0.92       206
        rose       0.81      0.88      0.84       137
   sunflower       0.92      0.92      0.92       172
       tulip       0.92      0.88      0.90       204

    accuracy                           0.89       863
   macro avg       0.89      0.89      0.89       863
weighted avg       0.90      0.89      0.89       863

    Lưu confusion matrix → outputs/confusion_matrix_mobilenet.png

 Fine-tuning MobileNetV2...
Epoch 1/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 592ms/step - accuracy: 0.8200 - loss: 0.4918
Epoch 1: val_accuracy improved from None to 0.89224, saving model to outputs/best_mobilenet_ft.keras

Epoch 1: finished saving model to outputs/best_mobilenet_ft.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 95s 720ms/step - accuracy: 0.8237 - loss: 0.4772 - val_accuracy: 0.8922 - val_loss: 0.3690 - learning_rate: 1.0000e-05
Epoch 2/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 533ms/step - accuracy: 0.8747 - loss: 0.3570
Epoch 2: val_accuracy did not improve from 0.89224
108/108 ━━━━━━━━━━━━━━━━━━━━ 67s 623ms/step - accuracy: 0.8741 - loss: 0.3436 - val_accuracy: 0.8922 - val_loss: 0.3710 - learning_rate: 1.0000e-05
Epoch 3/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 530ms/step - accuracy: 0.9006 - loss: 0.2791
Epoch 3: val_accuracy improved from 0.89224 to 0.89571, saving model to outputs/best_mobilenet_ft.keras

Epoch 3: finished saving model to outputs/best_mobilenet_ft.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 68s 629ms/step - accuracy: 0.9039 - loss: 0.2717 - val_accuracy: 0.8957 - val_loss: 0.3659 - learning_rate: 1.0000e-05
Epoch 4/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 533ms/step - accuracy: 0.9194 - loss: 0.2372
Epoch 4: val_accuracy did not improve from 0.89571
108/108 ━━━━━━━━━━━━━━━━━━━━ 66s 615ms/step - accuracy: 0.9201 - loss: 0.2336 - val_accuracy: 0.8922 - val_loss: 0.3657 - learning_rate: 1.0000e-05
Epoch 5/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 498ms/step - accuracy: 0.9300 - loss: 0.1993
Epoch 5: val_accuracy did not improve from 0.89571
108/108 ━━━━━━━━━━━━━━━━━━━━ 63s 582ms/step - accuracy: 0.9317 - loss: 0.1915 - val_accuracy: 0.8934 - val_loss: 0.3605 - learning_rate: 1.0000e-05
Epoch 6/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 500ms/step - accuracy: 0.9421 - loss: 0.1701
Epoch 6: val_accuracy improved from 0.89571 to 0.89687, saving model to outputs/best_mobilenet_ft.keras

Epoch 6: finished saving model to outputs/best_mobilenet_ft.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 64s 595ms/step - accuracy: 0.9441 - loss: 0.1675 - val_accuracy: 0.8969 - val_loss: 0.3565 - learning_rate: 1.0000e-05
Epoch 7/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 492ms/step - accuracy: 0.9521 - loss: 0.1365
Epoch 7: val_accuracy did not improve from 0.89687
108/108 ━━━━━━━━━━━━━━━━━━━━ 62s 576ms/step - accuracy: 0.9528 - loss: 0.1383 - val_accuracy: 0.8957 - val_loss: 0.3534 - learning_rate: 1.0000e-05
Epoch 8/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 488ms/step - accuracy: 0.9617 - loss: 0.1207
Epoch 8: val_accuracy improved from 0.89687 to 0.89919, saving model to outputs/best_mobilenet_ft.keras

Epoch 8: finished saving model to outputs/best_mobilenet_ft.keras
108/108 ━━━━━━━━━━━━━━━━━━━━ 62s 578ms/step - accuracy: 0.9647 - loss: 0.1181 - val_accuracy: 0.8992 - val_loss: 0.3518 - learning_rate: 1.0000e-05
Epoch 9/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9764 - loss: 0.1018
Epoch 9: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 569ms/step - accuracy: 0.9716 - loss: 0.1056 - val_accuracy: 0.8969 - val_loss: 0.3501 - learning_rate: 1.0000e-05
Epoch 10/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 488ms/step - accuracy: 0.9798 - loss: 0.0835
Epoch 10: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 62s 570ms/step - accuracy: 0.9789 - loss: 0.0884 - val_accuracy: 0.8934 - val_loss: 0.3509 - learning_rate: 1.0000e-05
Epoch 11/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 485ms/step - accuracy: 0.9834 - loss: 0.0728
Epoch 11: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 568ms/step - accuracy: 0.9820 - loss: 0.0771 - val_accuracy: 0.8946 - val_loss: 0.3498 - learning_rate: 1.0000e-05
Epoch 12/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 487ms/step - accuracy: 0.9855 - loss: 0.0664
Epoch 12: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 570ms/step - accuracy: 0.9820 - loss: 0.0686 - val_accuracy: 0.8934 - val_loss: 0.3519 - learning_rate: 1.0000e-05
Epoch 13/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9856 - loss: 0.0615
Epoch 13: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 569ms/step - accuracy: 0.9852 - loss: 0.0619 - val_accuracy: 0.8934 - val_loss: 0.3543 - learning_rate: 1.0000e-05
Epoch 14/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9883 - loss: 0.0540
Epoch 14: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 569ms/step - accuracy: 0.9899 - loss: 0.0512 - val_accuracy: 0.8969 - val_loss: 0.3546 - learning_rate: 1.0000e-05
Epoch 15/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9897 - loss: 0.0484
Epoch 15: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 569ms/step - accuracy: 0.9919 - loss: 0.0468 - val_accuracy: 0.8946 - val_loss: 0.3573 - learning_rate: 1.0000e-05
Epoch 16/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9955 - loss: 0.0352
Epoch 16: ReduceLROnPlateau reducing learning rate to 2.9999999242136253e-06.

Epoch 16: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 568ms/step - accuracy: 0.9942 - loss: 0.0370 - val_accuracy: 0.8957 - val_loss: 0.3584 - learning_rate: 1.0000e-05
Epoch 17/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - accuracy: 0.9934 - loss: 0.0332
Epoch 17: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 569ms/step - accuracy: 0.9928 - loss: 0.0344 - val_accuracy: 0.8946 - val_loss: 0.3585 - learning_rate: 3.0000e-06
Epoch 18/20
108/108 ━━━━━━━━━━━━━━━━━━━━ 0s 485ms/step - accuracy: 0.9963 - loss: 0.0298
Epoch 18: val_accuracy did not improve from 0.89919
108/108 ━━━━━━━━━━━━━━━━━━━━ 61s 567ms/step - accuracy: 0.9962 - loss: 0.0317 - val_accuracy: 0.8969 - val_loss: 0.3590 - learning_rate: 3.0000e-06
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
    Lưu biểu đồ lịch sử → outputs/training_history_mobilenet_finetune.png

 Đánh giá mô hình mobilenet_finetune...

============================================================
 CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

       daisy       0.89      0.88      0.88       144
   dandelion       0.90      0.94      0.92       206
        rose       0.86      0.87      0.87       137
   sunflower       0.92      0.91      0.91       172
       tulip       0.91      0.89      0.90       204

    accuracy                           0.90       863
   macro avg       0.90      0.90      0.90       863
weighted avg       0.90      0.90      0.90       863

    Lưu confusion matrix → outputs/confusion_matrix_mobilenet_finetune.png

 Hoàn tất! Kết quả lưu tại: ./outputs/
