# custom-cnn-binary-dog-cat-classifier

 - Dataset link - https://www.kaggle.com/datasets/princelv84/dogsvscats

# 🐶🐱 Binary CNN — Dog vs Cat Classifier

A **custom Convolutional Neural Network (CNN)** built from scratch using TensorFlow/Keras to perform **binary image classification** between dogs and cats.

---

## 📌 Project Overview

This project trains a deep CNN model on a **labeled dataset of dog and cat images**. The model learns to distinguish between the two classes using convolutional feature extraction followed by a fully connected ANN head with a sigmoid output for binary classification.

---

## 📁 Dataset Structure

The dataset is stored in Google Drive and unzipped into the Colab environment.

```
dataset_dogs_cats/
├── train/
│   ├── cats/       # Training cat images
│   └── dogs/       # Training dog images
└── val/
    ├── cats/       # Validation cat images
    └── dogs/       # Validation dog images
```

> Dataset source: Kaggle Dogs vs. Cats (`archive.zip`) stored in Google Drive.

---

## ⚙️ Setup & Installation

### 1. Mount Google Drive (Google Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Unzip the Dataset
```bash
!unzip '/content/drive/MyDrive/Deep Learning/Computer_Vision/Binary_CNN_DOG_VS_CAT/archive.zip' \
       -d '/content/dataset_dogs_cats'
```

### 3. Required Libraries
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
```

---

## 🔄 Data Preprocessing

**Images are preprocessed using `ImageDataGenerator`:**

- **Pixel scaling:** values normalized to `[0, 1]` (divided by 255)
- **Rotation range:** ±20%
- **Shear range:** 20%
- **Horizontal flip:** enabled
- **Target size:** 200×200 pixels
- **Batch size:** 32
- **Class mode:** `binary` (`cats = 0`, `dogs = 1`)

```python
training_data_preprocessing = ImageDataGenerator(
    1/255,
    rotation_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
```

---

## 🏗️ Model Architecture

The model is a **Sequential CNN** with **5 convolutional** blocks followed by **5 dense layers**.

### Convolutional Blocks (Feature Extraction)

| Layer | Filters | Kernel | Activation | Pooling |
|-------|---------|--------|------------|---------|
| Conv2D #1 | 500 | 3×3 | ReLU | MaxPool 2×2 |
| Conv2D #2 | 250 | 3×3 | ReLU | MaxPool 2×2 |
| Conv2D #3 | 100 | 3×3 | ReLU | MaxPool 2×2 |
| Conv2D #4 | 50  | 3×3 | ReLU | MaxPool 2×2 |
| Conv2D #5 | 20  | 3×3 | ReLU | MaxPool 2×2 |

> All Conv2D layers use `he_uniform` weight initialization and `valid` padding.

### Dense Layers (ANN Head)

| Layer | Units | Activation | Initializer |
|-------|-------|------------|-------------|
| Dense #1 | 300 | ReLU | he_uniform |
| Dense #2 | 200 | ReLU | he_uniform |
| Dense #3 | 100 | ReLU | he_uniform |
| Dense #4 | 50  | ReLU | he_uniform |
| Dense #5 | 20  | ReLU | he_uniform |
| Output   | 1   | Sigmoid | glorot_uniform |

---

## 🧪 Training

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['Accuracy']
)

model.fit(
    final_training_data,
    validation_data=final_valdiation_data,
    epochs=23
)
```

### Training Results (23 Epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|---------------|--------------|------------|----------|
| 1     | 51.4%         | 51.5%        | 5.13       | 0.693    |
| 6     | 51.3%         | 51.6%        | 0.692      | 0.690    |
| 11    | 50.0%         | 50.0%        | 0.694      | 0.693    |
| 23    | 49.9%         | 50.0%        | 0.693      | 0.693    |

> ⚠️ **Note:** The model converged to ~50% accuracy (random guessing), indicating the model did not learn effectively. This is likely due to the model being too deep/large for training without techniques like Batch Normalization, Dropout, or a lower learning rate. Consider using **Transfer Learning** (e.g., VGG16, ResNet) for better results.

---

## 🔮 Prediction

```python
import cv2

def predict_cnn(path_of_image):
    image = cv2.imread(path_of_image)
    resized_image = cv2.resize(image, (200, 200))
    scaled_pixel_value = resized_image / 255
    input_image = np.expand_dims(scaled_pixel_value, axis=0)
    result = model.predict(input_image)
    print(result)

    if result[0][0] > 0.5:
        print('cats')
    else:
        print('dogs')

    plt.imshow(image[:, :, ::-1])
    plt.show()

# Example usage
predict_cnn('/content/dog.jpeg')
```

> The model outputs a sigmoid probability. Values **> 0.5** → `cats`, values **≤ 0.5** → `dogs`.

---

## 🔧 Possible Improvements

- Add **Batch Normalization** after Conv layers to stabilize training
- Add **Dropout** layers to reduce overfitting
- Use **Transfer Learning** (VGG16, MobileNet, ResNet50) for higher accuracy
- Tune learning rate with a scheduler
- Increase dataset size or apply stronger augmentation
- Use `EarlyStopping` and `ModelCheckpoint` callbacks

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| ImageDataGenerator | Data augmentation & loading |
| OpenCV (`cv2`) | Image reading for prediction |
| Matplotlib | Visualization |
| Google Colab | Training environment |
| Google Drive | Dataset storage |

---

## 📄 License

This project is for educational purposes. Dataset sourced from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats).
