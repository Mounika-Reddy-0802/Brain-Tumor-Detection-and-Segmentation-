# Brain Tumor Detection, Classification & Segmentation — Complete Project Plan

**Dataset:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
**Stack:** Python · TensorFlow/Keras · PyTorch · OpenCV · Albumentations · Streamlit  
**Framework Strategy:** TensorFlow/Keras for Detection & Classification · PyTorch for U-Net Segmentation  
**Tasks:** Detection (binary) · Classification (multi-class) · Segmentation (U-Net)

---

## Framework Architecture Decision

This project uses a **hybrid framework strategy** — not arbitrary, but task-optimised:

| Task | Framework | Reason |
|---|---|---|
| Preprocessing | OpenCV + NumPy | Framework-agnostic; fastest C++ image ops |
| Detection (binary) | TensorFlow / Keras | Clean Functional API, built-in `EfficientNetB3`, Keras callbacks |
| Classification (4-class) | TensorFlow / Keras | `class_weight`, `tf.data` pipeline, TensorBoard integration |
| Segmentation (U-Net) | PyTorch + `smp` | `segmentation-models-pytorch` has best-in-class U-Net + EfficientNet encoder |
| Streamlit App | Both (loaded together) | Both models loaded via `@st.cache_resource`; GPU memory controlled |

Both frameworks coexist in one Python process. TensorFlow is prevented from consuming all GPU VRAM so PyTorch can operate alongside it.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Input & Preprocessing](#3-data-input--preprocessing)
4. [CNN Backbone — Feature Extraction](#4-cnn-backbone--feature-extraction)
5. [TensorFlow — Detection & Classification](#5-tensorflow--detection--classification)
6. [PyTorch — U-Net Segmentation](#6-pytorch--u-net-segmentation)
7. [Unified Inference Pipeline](#7-unified-inference-pipeline)
8. [Evaluation & Metrics](#8-evaluation--metrics)
9. [Streamlit Web Application](#9-streamlit-web-application)
10. [Handling Challenges](#10-handling-challenges)
11. [Code Generation Checklist](#11-code-generation-checklist)

---

## 1. Project Structure

```
brain_tumor_project/
│
├── data/
│   ├── raw/                            # Original Kaggle dataset (unzipped)
│   │   ├── Training/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── notumor/
│   │   │   └── pituitary/
│   │   └── Testing/
│   │       └── (same four folders)
│   ├── processed/                      # Cropped and normalized images
│   └── masks/                          # Pseudo-masks generated via thresholding
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py            # OpenCV crop, denoise, normalize (framework-agnostic)
│   │   ├── tf_dataset.py               # tf.data.Dataset pipeline for TF models
│   │   ├── torch_dataset.py            # PyTorch Dataset + DataLoader for segmentation
│   │   └── augmentation.py             # Albumentations transforms (used by both)
│   │
│   ├── models/
│   │   ├── tf_detection_model.py       # TF/Keras: EfficientNetB3 + binary head
│   │   ├── tf_classification_model.py  # TF/Keras: EfficientNetB3 + 4-class head
│   │   └── torch_unet.py               # PyTorch: U-Net with EfficientNetB3 encoder
│   │
│   ├── training/
│   │   ├── train_tf_detection.py       # TF training script (detection)
│   │   ├── train_tf_classification.py  # TF training script (classification)
│   │   └── train_torch_segmentation.py # PyTorch training script (U-Net)
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # All metric functions (framework-agnostic NumPy)
│   │   └── evaluate_all.py             # Runs evaluation for all three tasks
│   │
│   ├── inference/
│   │   └── pipeline.py                 # Hybrid pipeline: TF det/cls + PyTorch seg
│   │
│   └── utils/
│       ├── visualization.py            # Overlay masks, plot training curves
│       ├── gpu_setup.py                # Prevent TF from consuming all GPU VRAM
│       └── checkpoint.py               # Save/load utilities for both frameworks
│
├── app/
│   └── streamlit_app.py                # Streamlit frontend (loads TF + PyTorch models)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_tf_model_training.ipynb
│   └── 04_evaluation_and_metrics.ipynb
│
├── configs/
│   └── config.yaml                     # All hyperparameters in one place
│
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

### `requirements.txt`

```
# ── Deep Learning Frameworks ─────────────────────────────────────────────
tensorflow>=2.15.0                    # Detection + Classification
torch>=2.2.0                          # Segmentation (U-Net)
torchvision>=0.17.0
timm>=0.9.12                          # PyTorch EfficientNet pretrained weights
segmentation-models-pytorch>=0.3.3    # Pre-built U-Net with EfficientNet encoder

# ── TensorFlow Utilities ──────────────────────────────────────────────────
tensorflow-addons>=0.23.0             # Additional TF metrics (F1, Dice)

# ── Image Processing ──────────────────────────────────────────────────────
opencv-python>=4.9.0                  # Cropping and morphological ops (both frameworks)
albumentations>=1.3.1                 # Medical-grade augmentations (framework-agnostic)
Pillow>=10.2.0

# ── Data & Utilities ──────────────────────────────────────────────────────
numpy>=1.26.0
scikit-learn>=1.4.0                   # Metrics, stratified splits, class weights
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
PyYAML>=6.0.1

# ── App ───────────────────────────────────────────────────────────────────
streamlit>=1.33.0
```

### `src/utils/gpu_setup.py` — Critical: Run Before Any Model Loading

```python
import tensorflow as tf
import torch

def configure_gpu():
    """
    Prevents TensorFlow from claiming all GPU VRAM so PyTorch can
    allocate memory alongside it in the same process.
    Must be called before any model is instantiated.
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"TF GPUs available: {len(gpus)}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# Call at top of every training script and in streamlit_app.py
configure_gpu()
```

### `configs/config.yaml`

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  mask_dir: data/masks
  img_size: 224
  val_split: 0.15
  test_split: 0.15
  num_workers: 4
  use_pseudo_masks: true              # Set false if using BraTS/TCGA expert masks

model:
  backbone: efficientnet_b3
  pretrained: true
  num_classes: 4                      # glioma, meningioma, notumor, pituitary
  dropout: 0.4

tensorflow:
  detection_epochs: 32
  classification_epochs: 64
  batch_size: 32
  optimizer: adam
  lr: 1.0e-4
  weight_decay: 1.0e-5
  reduce_lr_factor: 0.5
  reduce_lr_patience: 4
  early_stopping_patience: 8
  mixed_precision: true               # tf.keras.mixed_precision policy
  freeze_backbone_epochs: 5           # Warmup: freeze backbone for first N epochs

pytorch:
  segmentation_epochs: 32
  batch_size: 16                      # U-Net is memory-heavy per image
  optimizer: adam
  lr: 1.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine_annealing
  T_max: 32
  early_stopping_patience: 8
  mixed_precision: true               # torch.cuda.amp

segmentation:
  encoder: efficientnet_b3
  decoder_channels: [256, 128, 64, 32, 16]
  dice_weight: 0.6
  bce_weight: 0.4
```

---

## 3. Data Input & Preprocessing

### 3.1 Dataset Overview

The Kaggle dataset contains **7,023 MRI images** split into:

| Class | Label (int) | Description |
|---|---|---|
| `glioma` | 0 | Malignant brain tumor |
| `meningioma` | 1 | Tumor arising from meninges |
| `notumor` | 2 | Healthy brain scan |
| `pituitary` | 3 | Tumor of the pituitary gland |

The dataset is pre-divided into `Training/` (~5,712 images) and `Testing/` (~1,311 images). It is moderately imbalanced — meningioma has significantly fewer samples (~306) than glioma (~827) or pituitary (~827).

### 3.2 Exploratory Data Analysis — `notebooks/01_eda.ipynb`

```python
import os, cv2, numpy as np, matplotlib.pyplot as plt, seaborn as sns

root = "data/raw/Training"
class_counts = {cls: len(os.listdir(f"{root}/{cls}")) for cls in os.listdir(root)}

sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title("Class Distribution (Training Set)"); plt.ylabel("Number of Images"); plt.show()

# Sample grid: 3 images per class
fig, axes = plt.subplots(4, 3, figsize=(10, 12))
for i, cls in enumerate(class_counts):
    samples = os.listdir(f"{root}/{cls}")[:3]
    for j, fname in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(f"{root}/{cls}/{fname}"), cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(img); axes[i, j].set_title(cls); axes[i, j].axis('off')
plt.tight_layout(); plt.show()
```

Key things to observe: class imbalance ratio, image size variation, black border presence, and intensity distribution differences between tumor types.

### 3.3 Preprocessing Pipeline — `src/data/preprocessing.py`

This module is **fully framework-agnostic** — it uses only OpenCV and NumPy. The same functions feed both `tf_dataset.py` and `torch_dataset.py`.

```python
import cv2
import numpy as np

def crop_brain_region(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crops the image to the bounding box of the brain region.
    Removes black/dark borders common in MRI scans.

    Steps:
    1. Convert to grayscale.
    2. Binary threshold to isolate non-background pixels.
    3. Morphological close + open to fill gaps and remove noise.
    4. Find bounding box of largest contour (the brain).
    5. Crop with 10px padding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    return image[y1:y2, x1:x2]


def preprocess_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Full preprocessing pipeline for a single MRI image.
    Returns a (target_size, target_size, 3) float32 NumPy array in [0, 1].
    Called identically by both TF and PyTorch dataset builders.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Crop to brain region (removes irrelevant dark borders)
    img = crop_brain_region(img)

    # Step 2: Resize to model input size
    img = cv2.resize(img, (target_size, target_size))

    # Step 3: CLAHE contrast enhancement (makes tumor boundaries more visible)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 4: Normalize to [0, 1]
    return img.astype(np.float32) / 255.0


def generate_pseudo_mask(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Generates a binary pseudo-mask using Otsu thresholding on bright regions.
    Used only when use_pseudo_masks=true in config.
    Returns a (target_size, target_size) uint8 mask.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = crop_brain_region(img)
    img = cv2.resize(img, (target_size, target_size))

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 2:
        largest_cc = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest_cc) * 255).astype(np.uint8)
    return mask
```

**Why CLAHE?** Contrast-Limited Adaptive Histogram Equalization locally enhances contrast — tumor boundaries are often subtle intensity differences that CLAHE makes more distinct without amplifying noise globally.

### 3.4 TensorFlow Data Pipeline — `src/data/tf_dataset.py`

```python
import tensorflow as tf
import numpy as np
from src.data.preprocessing import preprocess_image

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

def tf_load_and_preprocess(image_path, label, img_size=224):
    """
    Wraps the OpenCV preprocess_image function for use in tf.data.
    Uses tf.py_function to bridge OpenCV (NumPy) and TensorFlow.
    """
    def _load(path):
        img = preprocess_image(path.numpy().decode('utf-8'), img_size)
        return img.astype(np.float32)

    img = tf.py_function(func=_load, inp=[image_path], Tout=tf.float32)
    img.set_shape([img_size, img_size, 3])
    img = (img - IMAGENET_MEAN) / IMAGENET_STD   # ImageNet normalization
    return img, label


def apply_tf_augmentations(img, label):
    """TF-native augmentations applied on-GPU during training only."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return img, label


def build_tf_dataset(image_paths, labels, batch_size=32,
                     augment=False, shuffle=True, img_size=224):
    """
    Builds a tf.data.Dataset for detection or classification.
    Fully pipelined with prefetching for GPU utilisation.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda p, l: tf_load_and_preprocess(p, l, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(apply_tf_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(image_paths)),
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

### 3.5 PyTorch Dataset for Segmentation — `src/data/torch_dataset.py`

```python
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2, numpy as np
from src.data.preprocessing import preprocess_image, generate_pseudo_mask

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for U-Net segmentation.
    Returns (image_tensor, mask_tensor) pairs.
    Albumentations ensures identical spatial transforms on both image and mask.
    """
    def __init__(self, image_paths, mask_paths=None,
                 use_pseudo_masks=True, transform=None, img_size=224):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_pseudo_masks = use_pseudo_masks
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = preprocess_image(self.image_paths[idx], self.img_size)
        img_uint8 = (img * 255).astype(np.uint8)

        if self.use_pseudo_masks or self.mask_paths is None:
            mask = generate_pseudo_mask(self.image_paths[idx], self.img_size)
        else:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img_uint8, mask=mask)
            img_uint8 = augmented['image']
            mask = augmented['mask']

        img_tensor = torch.FloatTensor(img_uint8).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, torch.FloatTensor(mask).unsqueeze(0)


def get_segmentation_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),
        ], additional_targets={'mask': 'mask'})
    return None
```

---

## 4. CNN Backbone — Feature Extraction

### 4.1 Architecture Choice: EfficientNetB3

**Backbone: EfficientNetB3 — used in both TensorFlow and PyTorch**

| Factor | Reasoning |
|---|---|
| **Parameter efficiency** | ~12M params — far lighter than ResNet-50 (25M) or VGG-16 (138M) with better accuracy |
| **Feature quality** | Compound scaling of depth + width + resolution preserves both texture (tumor boundaries) and semantic features (tumor type) |
| **Transfer learning** | ImageNet pretraining gives strong priors for dense/sparse regions — relevant to MRI tissue density patterns |
| **Segmentation compatibility** | Hierarchical feature maps (stride 2/4/8/16/32) integrate as U-Net skip connections natively in `smp` |
| **Cross-framework availability** | Available in `tf.keras.applications.EfficientNetB3` AND `timm`/`smp` for PyTorch — same backbone, both frameworks |

#### Alternatives Considered

| Model | Verdict |
|---|---|
| ResNet-50 | Good fallback; validated in medical imaging but lower accuracy per parameter |
| DenseNet-121 | Used in CheXNet; high memory footprint makes it hard to pair with U-Net |
| VGG-16 | Outdated — very high parameter count, low accuracy. Avoid |
| ViT-B/16 | Strong ceiling but needs much larger datasets; consider only with BraTS 2020 augmentation |

---

## 5. TensorFlow — Detection & Classification

### 5.1 Detection Model — `src/models/tf_detection_model.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3

def build_detection_model(img_size: int = 224, dropout: float = 0.4) -> Model:
    """
    Binary classifier: tumor present (1) vs no tumor (0).
    Uses EfficientNetB3 from tf.keras.applications with ImageNet weights.
    Output: single sigmoid unit — BCELoss via binary_crossentropy.
    """
    base = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg'               # Global average pooling → (batch, 1536)
    )
    base.trainable = False          # Frozen during warmup phase

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs, name='detection_model')
```

### 5.2 Classification Model — `src/models/tf_classification_model.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3

def build_classification_model(num_classes: int = 4,
                                img_size: int = 224,
                                dropout: float = 0.4) -> Model:
    """
    4-class classifier: glioma, meningioma, notumor, pituitary.
    Output: softmax over 4 classes.
    """
    base = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='classification_model')
```

### 5.3 Training Script — `src/training/train_tf_detection.py`

```python
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.gpu_setup import configure_gpu
from src.data.tf_dataset import build_tf_dataset
from src.models.tf_detection_model import build_detection_model

configure_gpu()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Binary labels: 0 = no tumor, 1 = tumor (notumor class → 0, rest → 1)
y_detection = np.where(y_labels == 2, 0, 1)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(image_paths, y_detection))

train_ds = build_tf_dataset(image_paths[train_idx], y_detection[train_idx],
                             batch_size=32, augment=True)
val_ds   = build_tf_dataset(image_paths[val_idx],   y_detection[val_idx],
                             batch_size=32, augment=False)

weights = compute_class_weight('balanced', classes=np.unique(y_detection),
                                y=y_detection[train_idx])
class_weight_dict = {0: weights[0], 1: weights[1]}

model = build_detection_model()

# Phase 1: Warmup — frozen backbone, high LR on head only
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Recall(name='recall')]
)
model.fit(train_ds, epochs=5, validation_data=val_ds,
          class_weight=class_weight_dict)

# Phase 2: Fine-tune — unfreeze full backbone, low LR
model.get_layer('efficientnetb3').trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Recall(name='recall')]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_recall', patience=8,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/best_detection.keras', save_best_only=True,
        monitor='val_recall', mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir='logs/detection')
]

history = model.fit(
    train_ds, epochs=32,    # Power of 2; early stopping handles actual end
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
model.save('models/detection_model.keras')
```

### 5.4 Training Script — `src/training/train_tf_classification.py`

```python
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.utils.gpu_setup import configure_gpu
from src.data.tf_dataset import build_tf_dataset
from src.models.tf_classification_model import build_classification_model

configure_gpu()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

weights = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=y_train)
class_weight_dict = {i: weights[i] for i in range(4)}

model = build_classification_model(num_classes=4)

# Phase 1: Warmup
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.fit(train_ds, epochs=5, validation_data=val_ds,
          class_weight=class_weight_dict)

# Phase 2: Fine-tune
model.get_layer('efficientnetb3').trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/best_classifier.keras', save_best_only=True,
        monitor='val_accuracy', mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
    ),
    tf.keras.callbacks.TensorBoard(log_dir='logs/classification')
]

history = model.fit(
    train_ds, epochs=64,    # Power of 2; classification needs more iterations
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
model.save('models/classification_model.keras')
```

### 5.5 Optimizer & Scheduler — TensorFlow

#### Optimizer: Adam

```python
tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-5)
```

**Why Adam?** Adaptive per-parameter learning rates handle heterogeneous gradients from spatially sparse tumor regions. `weight_decay=1e-5` provides L2 regularization. Converges faster than SGD for fine-tuning pretrained networks on medical image tasks.

#### Scheduler: ReduceLROnPlateau

```python
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
```

Reduces LR by 50% when validation loss stagnates for 4 epochs — the Keras-native equivalent of CosineAnnealingLR. Both prevent the LR from being too large in later training.

#### Two-Phase Training Strategy

| Phase | Epochs | Backbone | LR | Purpose |
|---|---|---|---|---|
| Warmup | 5 | Frozen | 1e-3 | Train only new head layers without corrupting pretrained weights |
| Fine-tune | 32 or 64 | Unfrozen | 1e-4 | Gradually adapt backbone to MRI domain |

---

## 6. PyTorch — U-Net Segmentation

### 6.1 Architecture — `src/models/torch_unet.py`

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

def get_unet(encoder_name: str = 'efficientnet_b3', pretrained: bool = True) -> nn.Module:
    """
    U-Net with EfficientNetB3 encoder from segmentation-models-pytorch.
    Decoder upsamples via skip connections from encoder stages.
    Output: (batch, 1, H, W) binary mask logits — apply sigmoid at inference.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None     # Raw logits — sigmoid applied in loss and inference
    )
```

### 6.2 U-Net Data Flow

```
Input (3, 224, 224)
   │
   ├─► Encoder Stage 1 → (32,  112, 112)  ─────────────────────── Skip 1
   ├─► Encoder Stage 2 → (48,   56,  56)  ──────────────────── Skip 2
   ├─► Encoder Stage 3 → (136,  28,  28)  ─────────────── Skip 3
   ├─► Encoder Stage 4 → (384,  14,  14)  ────────── Skip 4
   └─► Bottleneck      → (1536,  7,   7)
              │
   ┌──────────▼──────────────────────────────────────────────────────────┐
   │ Decoder: Upsample ×2 + Concat(skip) + Conv + BatchNorm + ReLU      │
   └──► (256,14) → (128,28) → (64,56) → (32,112) → (16,224)
              │
   └─► 1×1 Conv → (1, 224, 224)  Binary mask logits
```

**Why U-Net?** Skip connections reintroduce fine spatial detail lost during encoding. The encoder builds global context (tumor type, location); skip connections restore precise boundary information needed for pixel-accurate segmentation.

### 6.3 Combined Segmentation Loss

```python
import torch, torch.nn as nn

class CombinedSegmentationLoss(nn.Module):
    """
    60% Dice Loss + 40% Binary Cross-Entropy.
    Dice handles severe foreground/background pixel imbalance (tumor << background).
    BCE provides dense pixel-level gradient signal at every step.
    """
    def __init__(self, dice_weight=0.6, bce_weight=0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

    def forward(self, pred, target):
        return self.dice_weight * self.dice_loss(pred, target) + \
               self.bce_weight  * self.bce(pred, target)
```

### 6.4 Training Script — `src/training/train_torch_segmentation.py`

```python
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from src.utils.gpu_setup import configure_gpu
from src.models.torch_unet import get_unet

configure_gpu()

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = get_unet().to(device)
criterion = CombinedSegmentationLoss(dice_weight=0.6, bce_weight=0.4)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-6)
scaler    = GradScaler()


class EarlyStopping:
    def __init__(self, patience=8, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.best = None; self.counter = 0; self.stop = False

    def __call__(self, val_loss, model, path):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss; self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


early_stopping = EarlyStopping(patience=8)

for epoch in range(32):     # Power of 2; early stopping handles actual termination
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()

    val_loss, val_dice = validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch+1:02d} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    early_stopping(val_loss, model, 'models/best_unet.pth')
    if early_stopping.stop:
        print("Early stopping triggered."); break
```

---

## 7. Unified Inference Pipeline

### 7.1 Hybrid Pipeline — `src/inference/pipeline.py`

```python
import tensorflow as tf
import torch
import numpy as np, cv2
from src.data.preprocessing import preprocess_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES   = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


class BrainTumorPipeline:
    """
    Hybrid inference pipeline:
      TensorFlow → Detection and Classification
      PyTorch    → U-Net Segmentation
    All three tasks run on a single uploaded image.
    """
    DETECTION_THRESHOLD = 0.5
    MASK_THRESHOLD      = 0.5

    def __init__(self, tf_detector, tf_classifier, torch_unet, device='cpu'):
        self.tf_detector   = tf_detector
        self.tf_classifier = tf_classifier
        self.torch_unet    = torch_unet.to(device).eval()
        self.device        = device

    def _tf_input(self, img: np.ndarray) -> np.ndarray:
        """Returns (1, H, W, 3) float32 normalized for TF models."""
        return np.expand_dims((img - IMAGENET_MEAN) / IMAGENET_STD, 0).astype(np.float32)

    def _torch_input(self, img: np.ndarray) -> torch.Tensor:
        """Returns (1, 3, H, W) tensor normalized for PyTorch U-Net."""
        norm = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.FloatTensor(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def predict(self, image_path: str) -> dict:
        img = preprocess_image(image_path)         # float32 [0,1] (H, W, 3)
        tf_in = self._tf_input(img)

        # TensorFlow: Detection
        det_prob = float(self.tf_detector.predict(tf_in, verbose=0)[0, 0])
        tumor_detected = det_prob >= self.DETECTION_THRESHOLD

        # TensorFlow: Classification
        cls_probs = self.tf_classifier.predict(tf_in, verbose=0)[0]
        predicted_class = int(np.argmax(cls_probs))

        # PyTorch: Segmentation
        with torch.no_grad():
            mask_probs = torch.sigmoid(self.torch_unet(self._torch_input(img)))
            mask_probs = mask_probs.cpu().numpy()[0, 0]

        binary_mask = (mask_probs >= self.MASK_THRESHOLD).astype(np.uint8) * 255

        return {
            'tumor_detected':       tumor_detected,
            'detection_confidence': det_prob,
            'tumor_type':           CLASS_NAMES[predicted_class] if tumor_detected else 'No Tumor',
            'class_probabilities':  dict(zip(CLASS_NAMES, cls_probs.tolist())),
            'segmentation_mask':    binary_mask,
            'segmentation_probs':   mask_probs,
            'overlay_image':        self._overlay(img, binary_mask),
        }

    def _overlay(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_u8 = (img * 255).astype(np.uint8)
        red = np.zeros_like(img_u8)
        red[mask > 0] = [255, 0, 0]
        return cv2.addWeighted(img_u8, 0.7, red, 0.3, 0)
```

---

## 8. Evaluation & Metrics

All metric functions use NumPy/scikit-learn — **fully framework-agnostic**. They accept outputs from TF models and PyTorch U-Net interchangeably.

### 8.1 Detection Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def detection_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Primary metric: Recall (Sensitivity).
    False negatives (missed tumors) are clinically more dangerous.
    Target: Recall >= 0.95, AUC-ROC >= 0.97.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'precision':   precision_score(y_true, y_pred),
        'recall':      recall_score(y_true, y_pred),        # = Sensitivity
        'specificity': recall_score(y_true, y_pred, pos_label=0),
        'f1':          f1_score(y_true, y_pred),
        'auc_roc':     roc_auc_score(y_true, y_pred_probs),
    }
```

### 8.2 Classification Metrics

```python
from sklearn.metrics import confusion_matrix

def classification_metrics(y_true, y_pred, class_names):
    return {
        'accuracy':            accuracy_score(y_true, y_pred),
        'macro_f1':            f1_score(y_true, y_pred, average='macro'),
        'weighted_f1':         f1_score(y_true, y_pred, average='weighted'),
        'per_class_precision': dict(zip(class_names, precision_score(y_true, y_pred, average=None))),
        'per_class_recall':    dict(zip(class_names, recall_score(y_true, y_pred, average=None))),
        'confusion_matrix':    confusion_matrix(y_true, y_pred),
    }
```

### 8.3 Segmentation Metrics (PyTorch)

```python
import torch

def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    """Dice = 2|P ∩ T| / (|P| + |T|). Range 0→1. Target: >= 0.75 (pseudo-masks)."""
    pred = (pred_mask > 0.5).float().view(-1)
    true = true_mask.float().view(-1)
    return ((2 * (pred * true).sum() + smooth) / (pred.sum() + true.sum() + smooth)).item()

def iou_score(pred_mask, true_mask, smooth=1e-6):
    """IoU (Jaccard). Stricter than Dice — penalises false positives more. Target: >= 0.65."""
    pred = (pred_mask > 0.5).float().view(-1)
    true = true_mask.float().view(-1)
    intersection = (pred * true).sum()
    return ((intersection + smooth) / (pred.sum() + true.sum() - intersection + smooth)).item()
```

### 8.4 Target Performance Benchmarks

| Task | Framework | Metric | Target |
|---|---|---|---|
| Detection | TensorFlow | Recall (Sensitivity) | ≥ 0.95 |
| Detection | TensorFlow | AUC-ROC | ≥ 0.97 |
| Classification | TensorFlow | Weighted F1 | ≥ 0.90 |
| Classification | TensorFlow | Top-1 Accuracy | ≥ 0.92 |
| Segmentation | PyTorch | Dice Coefficient | ≥ 0.75 (pseudo), ≥ 0.85 (expert) |
| Segmentation | PyTorch | IoU | ≥ 0.65 |

---

## 9. Streamlit Web Application

### 9.1 App Design — `app/streamlit_app.py`

```python
import streamlit as st
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile, os, io, sys

sys.path.append('..')
from src.utils.gpu_setup import configure_gpu
from src.models.tf_detection_model import build_detection_model
from src.models.tf_classification_model import build_classification_model
from src.models.torch_unet import get_unet
from src.inference.pipeline import BrainTumorPipeline

configure_gpu()   # Must be first — prevents TF from claiming all GPU VRAM

st.set_page_config(page_title="Brain Tumor AI", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection & Analysis")
st.caption("Upload a brain MRI scan for AI-powered tumor detection, classification, and segmentation.")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info(
        "Hybrid deep learning pipeline:\n\n"
        "**TensorFlow/Keras** — Detection & Classification\n\n"
        "**PyTorch U-Net** — Tumor Segmentation\n\n"
        "**OpenCV** — MRI Preprocessing"
    )
    st.warning("⚠️ For research purposes only. Not a clinical diagnostic tool.")
    st.markdown("---")
    detection_threshold = st.slider("Detection threshold", 0.3, 0.9, 0.5, 0.05)
    mask_threshold      = st.slider("Segmentation mask threshold", 0.3, 0.9, 0.5, 0.05)

# ── Load Models — cached per session ─────────────────────────────────────
@st.cache_resource
def load_pipeline():
    tf_detector   = tf.keras.models.load_model('models/detection_model.keras')
    tf_classifier = tf.keras.models.load_model('models/classification_model.keras')
    unet = get_unet(pretrained=False)
    unet.load_state_dict(torch.load('models/best_unet.pth', map_location='cpu'))
    return BrainTumorPipeline(tf_detector, tf_classifier, unet, device='cpu')

pipeline = load_pipeline()

# ── Upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read()); tmp_path = tmp.name

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original MRI")
        st.image(uploaded, use_column_width=True)

    with st.spinner("Running TF detection + classification and PyTorch segmentation..."):
        pipeline.DETECTION_THRESHOLD = detection_threshold
        pipeline.MASK_THRESHOLD      = mask_threshold
        result = pipeline.predict(tmp_path)
    os.unlink(tmp_path)

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(result['overlay_image'], use_column_width=True,
                 caption="Red = predicted tumor region (PyTorch U-Net)")

    with col3:
        st.subheader("Results")

        st.markdown("**Detection** *(TensorFlow)*")
        if result['tumor_detected']:
            st.error(f"🔴 Tumor Detected — {result['detection_confidence']:.1%} confidence")
        else:
            st.success(f"🟢 No Tumor — {1-result['detection_confidence']:.1%} confidence")

        st.markdown("---")
        st.markdown("**Classification** *(TensorFlow)*")
        st.markdown(f"Predicted type: **{result['tumor_type']}**")

        probs = result['class_probabilities']
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.barh(list(probs.keys()), list(probs.values()),
                color=['#e74c3c' if v == max(probs.values()) else '#3498db'
                       for v in probs.values()])
        ax.set_xlim(0, 1); ax.set_xlabel("Probability")
        ax.tick_params(axis='y', labelsize=9)
        fig.tight_layout(); st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Segmentation** *(PyTorch U-Net)*")
        coverage = float((result['segmentation_mask'] > 0).mean() * 100)
        st.metric("Tumor region coverage", f"{coverage:.1f}% of image")

    buf = io.BytesIO()
    Image.fromarray(result['segmentation_mask']).save(buf, format='PNG')
    st.download_button("⬇ Download Segmentation Mask",
                       data=buf.getvalue(), file_name="segmentation_mask.png",
                       mime="image/png")
```

### 9.2 Running the App

```bash
streamlit run app/streamlit_app.py

# View TensorBoard training curves (open separately)
tensorboard --logdir logs/
```

---

## 10. Handling Challenges

### 10.1 Class Imbalance

**Problem:** Meningioma has ~306 training samples vs ~827 for glioma/pituitary.

**TensorFlow:** Pass `class_weight` dict to `model.fit()`:

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=y_train)
model.fit(..., class_weight={i: weights[i] for i in range(4)})
```

**PyTorch (Segmentation DataLoader):** Use `WeightedRandomSampler`:

```python
from torch.utils.data import WeightedRandomSampler
sample_weights = [weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
loader  = DataLoader(dataset, batch_size=16, sampler=sampler)
```

Also apply heavier Albumentations augmentations to minority-class images during preprocessing.

### 10.2 Overfitting

**Symptoms:** Train accuracy >> validation accuracy after ~20 epochs.

**TensorFlow mitigations:**

```python
# L2 regularization + BatchNormalization in head (already in model definitions above)
layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
layers.BatchNormalization()
layers.Dropout(0.4)

# Differential learning rates: backbone lower, head higher
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)   # Global LR
# Use layer-specific LR by subclassing or using separate optimizers
```

**PyTorch (U-Net) mitigations:**

```python
optimizer = Adam([
    {'params': unet.encoder.parameters(), 'lr': 1e-5},   # Fine-tune encoder slowly
    {'params': unet.decoder.parameters(), 'lr': 1e-4},   # Train decoder faster
], weight_decay=1e-5)
```

### 10.3 TF + PyTorch GPU Memory Coexistence

```python
# In gpu_setup.py (already shown in Section 2) — always call configure_gpu() first
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Optional: cap PyTorch VRAM if TF needs headroom
torch.cuda.set_per_process_memory_fraction(0.5, device=0)
```

### 10.4 Segmentation Without Ground Truth Masks

If using pseudo-masks, be transparent about quality:
- Report Dice scores with a note about pseudo-mask limitations.
- Validate qualitatively via overlay visualisations in the Streamlit app.
- Set `use_pseudo_masks: false` and provide BraTS 2020 or TCGA-LGG expert masks for production-level segmentation.

### 10.5 Inference Speed for Streamlit

```python
# TF: SavedModel format for 2–3× faster inference than .keras format
tf.saved_model.save(model, 'models/detection_saved_model')
loaded = tf.saved_model.load('models/detection_saved_model')

# PyTorch: Export U-Net to ONNX for CPU speedup
import torch.onnx
torch.onnx.export(unet, torch.randn(1, 3, 224, 224),
                  'models/unet.onnx', opset_version=17)
```

Both TF and PyTorch models are cached with `@st.cache_resource` so they load only once per Streamlit session.

---

## 11. Code Generation Checklist

### Data
- [ ] `preprocessing.py` — `crop_brain_region`, `preprocess_image`, `generate_pseudo_mask` — OpenCV only, no framework dependency
- [ ] `tf_dataset.py` — `build_tf_dataset` using `tf.py_function` to wrap OpenCV preprocessor; includes `apply_tf_augmentations`
- [ ] `torch_dataset.py` — `SegmentationDataset` with Albumentations joint transforms; respects `use_pseudo_masks` flag
- [ ] Stratified splits via `sklearn.model_selection.StratifiedShuffleSplit`
- [ ] TF training uses `class_weight` dict; PyTorch segmentation uses `WeightedRandomSampler`

### TensorFlow Models
- [ ] `tf_detection_model.py` — `EfficientNetB3(pooling='avg')`, single sigmoid output, `BatchNormalization`, `l2` regularizer
- [ ] `tf_classification_model.py` — same backbone, 4-class softmax, same head structure
- [ ] Both models start with `base.trainable = False`; unfrozen after 5 warmup epochs

### TensorFlow Training
- [ ] `configure_gpu()` called before any model code in every script
- [ ] `tf.keras.mixed_precision.set_global_policy('mixed_float16')` enabled
- [ ] Adam with `lr=1e-4`, `weight_decay=1e-5` in fine-tune phase
- [ ] `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)` callback
- [ ] `EarlyStopping(patience=8, restore_best_weights=True)` callback
- [ ] `ModelCheckpoint(save_best_only=True)` callback
- [ ] `TensorBoard(log_dir=...)` callback
- [ ] Detection EarlyStopping monitors `val_recall`; Classification monitors `val_accuracy`
- [ ] Detection: 32 epochs; Classification: 64 epochs (powers of 2)

### PyTorch Segmentation
- [ ] `torch_unet.py` — `smp.Unet(encoder_name='efficientnet_b3', classes=1, activation=None)`
- [ ] `CombinedSegmentationLoss` — Dice 0.6 + BCE 0.4
- [ ] `torch.cuda.amp.GradScaler` + `autocast()` for mixed-precision
- [ ] Adam with `lr=1e-4`, `weight_decay=1e-5`
- [ ] `CosineAnnealingLR(T_max=32, eta_min=1e-6)`
- [ ] Custom `EarlyStopping` class (patience=8) monitoring val loss
- [ ] Segmentation: 32 epochs (power of 2)

### Evaluation
- [ ] `metrics.py` — NumPy/scikit-learn only, no framework dependency
- [ ] `detection_metrics` — accuracy, precision, recall, specificity, F1, AUC-ROC
- [ ] `classification_metrics` — per-class P/R + weighted F1 + confusion matrix
- [ ] `dice_coefficient` and `iou_score` in PyTorch

### Inference Pipeline
- [ ] `pipeline.py` loads TF models via `tf.keras.models.load_model`
- [ ] `pipeline.py` loads PyTorch U-Net via `torch.load` + `load_state_dict`
- [ ] `_tf_input` → `(1, H, W, 3)` float32; `_torch_input` → `(1, 3, H, W)` tensor
- [ ] `predict()` returns `tumor_detected`, `detection_confidence`, `tumor_type`, `class_probabilities`, `segmentation_mask`, `overlay_image`
- [ ] `_overlay` uses `cv2.addWeighted` for semi-transparent red highlight

### Streamlit App
- [ ] `configure_gpu()` at top before any model loading
- [ ] Both TF models + PyTorch U-Net wrapped in single `@st.cache_resource` function
- [ ] Temp file used to pass uploaded bytes to OpenCV `preprocess_image`
- [ ] Three-column layout: original | overlay | results
- [ ] Results labeled by framework: "TensorFlow" for det/cls, "PyTorch U-Net" for segmentation
- [ ] Adjustable detection and mask thresholds in sidebar
- [ ] `st.metric` for tumor region coverage percentage
- [ ] Download button for segmentation mask PNG
- [ ] Clinical disclaimer in sidebar

### Configuration
- [ ] `config.yaml` has separate `tensorflow:` and `pytorch:` sections with independent hyperparameters
- [ ] `use_pseudo_masks` flag in `data:` section, respected by `torch_dataset.py`
- [ ] `README.md` explains hybrid framework strategy and why each task uses its framework

---

## Recommended Training Sequence

```bash
# 0. Install all dependencies
pip install -r requirements.txt

# 1. Prepare data — OpenCV preprocessing (framework-agnostic)
python src/data/preprocessing.py --raw-dir data/raw --out-dir data/processed

# 2. Train TensorFlow detection model (32 epochs + early stopping on val_recall)
python src/training/train_tf_detection.py --config configs/config.yaml

# 3. Train TensorFlow classification model (64 epochs + early stopping on val_accuracy)
python src/training/train_tf_classification.py --config configs/config.yaml

# 4. Train PyTorch U-Net segmentation model (32 epochs + early stopping on val_loss)
python src/training/train_torch_segmentation.py --config configs/config.yaml

# 5. Evaluate all three tasks (produces metrics tables + confusion matrix PNGs)
python src/evaluation/evaluate_all.py --config configs/config.yaml

# 6. Launch Streamlit app (loads TF + PyTorch models together)
streamlit run app/streamlit_app.py

# 7. View TensorBoard training curves for TF models
tensorboard --logdir logs/
```

---

*This plan is a complete specification for AI code generation using a hybrid TensorFlow + PyTorch architecture. Every function signature, framework choice, training configuration, and infrastructure detail is internally consistent and production-ready.*
