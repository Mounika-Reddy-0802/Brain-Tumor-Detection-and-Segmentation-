# Brain Tumor Detection, Classification, and Segmentation

Hybrid deep learning project for MRI analysis:
- Detection (binary): TensorFlow/Keras EfficientNetB3
- Classification (4-class): TensorFlow/Keras EfficientNetB3
- Segmentation (U-Net): PyTorch + segmentation-models-pytorch

## Project Layout

- app/: Streamlit application
- configs/: YAML configuration
- data/: Raw, processed, and mask folders
- src/data/: preprocessing and dataset pipelines
- src/models/: TensorFlow and PyTorch model definitions
- src/training/: training scripts for all tasks
- src/evaluation/: metrics and evaluation scripts
- src/inference/: unified inference pipeline
- src/utils/: utilities for GPU setup, checkpoints, and visualization
- notebooks/: exploratory and demo notebooks

## Dataset

Expected classes under:
- Training/glioma
- Training/meningioma
- Training/notumor
- Training/pituitary
- Testing/glioma
- Testing/meningioma
- Testing/notumor
- Testing/pituitary

The code first tries data/raw and automatically falls back to Dataset if data/raw is not present.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Training Sequence

1. Detection model:

   python -m src.training.train_tf_detection --config configs/config.yaml

2. Classification model:

   python -m src.training.train_tf_classification --config configs/config.yaml

3. Segmentation model:

   python -m src.training.train_torch_segmentation --config configs/config.yaml

4. Evaluate all tasks:

   python -m src.evaluation.evaluate_all --config configs/config.yaml

5. Launch app:

   streamlit run app/streamlit_app.py

## Notes

- Run GPU setup before loading models to allow TensorFlow and PyTorch to coexist.
- If you only have CPU, scripts still run but training will be slow.
- Segmentation supports pseudo-masks when no expert masks are available.
