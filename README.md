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

The bundled `Dataset/` folder holds 1400 training and 400 testing images per class.
`data/processed/` and `data/masks/` are derived from it and are not committed —
regenerate them with the commands below.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Configurations

| Config | Reads from | Use for |
|---|---|---|
| `configs/config.yaml` | `Dataset/` | Running straight from the original images; applies crop/CLAHE while loading, which is slow |
| `configs/config_processed.yaml` | `data/processed/` | GPU training; decodes prepared JPEGs and reads precomputed masks |

Two flags control this: `data.preprocessed_input` (skip the OpenCV pipeline at load
time because the files on disk already went through it) and `data.precomputed_masks`
(read `data/masks/` instead of regenerating pseudo-masks every epoch). Both matter
for throughput — with them off, the Python/OpenCV work single-threads the input
pipeline and leaves the GPU idle.

## Training Sequence

Training is done on Kaggle (see below), but the same commands run anywhere:

0. Prepare the derived data (once, ~3 min on local disk):

   python -m src.data.preprocessing --raw-dir Dataset --out-dir data/processed --img-size 224
   python -m scripts.generate_masks --config configs/config_processed.yaml

1. Detection model:

   python -m src.training.train_tf_detection --config configs/config_processed.yaml

2. Classification model:

   python -m src.training.train_tf_classification --config configs/config_processed.yaml

3. Segmentation model:

   python -m src.training.train_torch_segmentation --config configs/config_processed.yaml

4. Evaluate all tasks on the held-out Testing split:

   python -m src.evaluation.evaluate_all --config configs/config_processed.yaml

5. Launch app:

   streamlit run app/streamlit_app.py

## Training on Kaggle

`notebooks/kaggle_train.ipynb` runs the full sequence on a Kaggle GPU. Set the
accelerator to GPU and Internet to On, point `REPO_URL` at your repository, and run
all cells. Roughly two hours of GPU time end to end, against a 30 h weekly quota.

## Notes

- **Input range**: `keras.applications.EfficientNet*` carries its own
  `Rescaling(1/255)` and `Normalization` layers, so every TensorFlow entry point
  feeds it pixels in `[0, 255]` (`EFFICIENTNET_INPUT_SCALE`). The PyTorch U-Net
  encoder has no built-in preprocessing and uses explicit ImageNet mean/std instead.
- **Keras 3.11**: loading ImageNet weights into the V1 EfficientNets is broken in
  this version (an unadapted `Normalization` layer shifts the index-based `.h5`
  loader). `src/models/backbone.py` falls back to name-based loading, so both older
  and newer Keras work.
- Run GPU setup before loading models to allow TensorFlow and PyTorch to coexist.
- Set `tensorflow.require_gpu: true` to fail fast rather than train for hours on CPU.
- **Segmentation labels are weak, and their quality is the limiting factor.**
  This dataset ships no expert annotations. `generate_pseudo_mask` approximates a
  lesion as the brightest compact blob inside the brain, after eroding away the
  skull and rejecting elongated or non-outlier components; scans labelled
  `notumor` get an empty mask from their folder label. Measured on 40 images per
  class, it fires on 73% of tumour scans and 42% of `notumor` scans, and visual
  inspection shows roughly a third of the masks land on the actual lesion — the
  rest catch eye globes, skull-base structures or ventricles. It is a usable weak
  label and far better than the previous version (which outlined the whole brain
  at ~70% of brain area on every image, carrying no tumour information at all),
  but it is not ground truth.

  **For genuine segmentation quality, train against expert masks.** The Cheng
  figshare dataset (https://doi.org/10.6084/m9.figshare.1512427.v5, CC BY 4.0)
  covers exactly these three tumour types — 3064 T1-contrast slices from 233
  patients with manually traced tumour borders. Convert its `.mat` files to
  image/mask pairs, point `data.mask_dir` at them and set `use_pseudo_masks: false`.
  Report Dice against those, not against the heuristic.
- Segmentation Dice/IoU is averaged over images that carry a reference lesion.
  Empty-reference scans are reported separately as a true-negative rate, because
  an empty prediction against an empty target scores 1.0 and would inflate the mean.
- Working inside a OneDrive-synced folder makes data preparation very slow, because
  reading each dehydrated file triggers a download. Prefer Kaggle or a local disk.
