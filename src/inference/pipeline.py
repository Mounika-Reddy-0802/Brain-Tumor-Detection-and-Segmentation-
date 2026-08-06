from __future__ import annotations

import cv2
import numpy as np

try:
    import torch
except ImportError:  # segmentation is optional; detection/classification are TF-only
    torch = None

from src.data.preprocessing import preprocess_image
from src.utils.project import EFFICIENTNET_INPUT_SCALE

# Only the PyTorch U-Net needs these; the TF models normalize internally.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NO_TUMOR_INDEX = CLASS_NAMES.index("No Tumor")


class BrainTumorPipeline:
    """
    Hybrid inference pipeline:
      TensorFlow -> detection and classification
      PyTorch -> segmentation
    """

    DETECTION_THRESHOLD = 0.5
    MASK_THRESHOLD = 0.5

    def __init__(self, tf_detector, tf_classifier, torch_unet=None, device: str = "cpu") -> None:
        """torch_unet may be None, in which case predict() skips segmentation.

        Detection and classification are pure TensorFlow, so the pipeline stays
        usable on a host where PyTorch could not be installed.
        """
        self.tf_detector = tf_detector
        self.tf_classifier = tf_classifier
        self.torch_unet = torch_unet.to(device).eval() if torch_unet is not None else None
        self.device = device

    def _tf_input(self, img: np.ndarray) -> np.ndarray:
        """EfficientNet rescales and normalizes internally, so it wants [0, 255]."""
        return np.expand_dims(img * EFFICIENTNET_INPUT_SCALE, 0).astype(np.float32)

    def _torch_input(self, img: np.ndarray) -> torch.Tensor:
        """smp's EfficientNet encoder has no built-in preprocessing, so normalize here."""
        norm = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.FloatTensor(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def _overlay(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_u8 = (img * 255).astype(np.uint8)
        red = np.zeros_like(img_u8)
        red[mask > 0] = [255, 0, 0]
        return cv2.addWeighted(img_u8, 0.7, red, 0.3, 0)

    def predict(self, image_path: str) -> dict:
        img = preprocess_image(image_path)
        tf_in = self._tf_input(img)

        det_prob = float(self.tf_detector.predict(tf_in, verbose=0)[0, 0])
        tumor_detected = det_prob >= self.DETECTION_THRESHOLD

        cls_probs = self.tf_classifier.predict(tf_in, verbose=0)[0]
        predicted_class = int(np.argmax(cls_probs))

        if self.torch_unet is None:
            mask_probs = np.zeros(img.shape[:2], dtype=np.float32)
        else:
            with torch.no_grad():
                mask_probs = torch.sigmoid(self.torch_unet(self._torch_input(img))).cpu().numpy()[0, 0]

        binary_mask = (mask_probs >= self.MASK_THRESHOLD).astype(np.uint8) * 255

        # Report what the classifier actually predicted rather than overriding it
        # with the detector's verdict. Overriding produced displays that
        # contradicted the probability chart shown beside them - "No Tumor"
        # printed above a bar chart with meningioma at 0.85. The two models are
        # independent, so surface the disagreement instead of hiding it.
        classifier_says_tumor = predicted_class != NO_TUMOR_INDEX

        return {
            "tumor_detected": tumor_detected,
            "detection_confidence": det_prob,
            "tumor_type": CLASS_NAMES[predicted_class],
            "class_confidence": float(cls_probs[predicted_class]),
            "models_disagree": tumor_detected != classifier_says_tumor,
            "class_probabilities": dict(zip(CLASS_NAMES, cls_probs.tolist())),
            "segmentation_mask": binary_mask,
            "segmentation_probs": mask_probs,
            "overlay_image": self._overlay(img, binary_mask),
        }
