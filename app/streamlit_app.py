from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.pipeline import BrainTumorPipeline
from src.models.torch_unet import get_unet
from src.utils.gpu_setup import configure_gpu

configure_gpu()

st.set_page_config(page_title="Brain Tumor AI", page_icon="BT", layout="wide")
st.title("Brain Tumor Detection and Analysis")
st.caption("Upload a brain MRI image for detection, classification, and segmentation.")


@st.cache_resource
def load_pipeline() -> BrainTumorPipeline:
    tf_detector = tf.keras.models.load_model("models/detection_model.keras")
    tf_classifier = tf.keras.models.load_model("models/classification_model.keras")

    unet = get_unet(pretrained=False)
    unet_path = "models/best_unet.pth" if Path("models/best_unet.pth").exists() else "models/unet_last.pth"
    unet.load_state_dict(torch.load(unet_path, map_location="cpu"))

    return BrainTumorPipeline(tf_detector, tf_classifier, unet, device="cpu")


with st.sidebar:
    st.header("About")
    st.info(
        "Hybrid pipeline\n\n"
        "TensorFlow/Keras: Detection and Classification\n\n"
        "PyTorch U-Net: Segmentation\n\n"
        "OpenCV: MRI preprocessing"
    )
    st.warning("For research use only. Not a clinical diagnostic tool.")
    st.markdown("---")
    detection_threshold = st.slider("Detection threshold", 0.3, 0.9, 0.5, 0.05)
    mask_threshold = st.slider("Segmentation mask threshold", 0.3, 0.9, 0.5, 0.05)

pipeline = load_pipeline()
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(uploaded, use_container_width=True)

    with st.spinner("Running inference..."):
        pipeline.DETECTION_THRESHOLD = detection_threshold
        pipeline.MASK_THRESHOLD = mask_threshold
        result = pipeline.predict(tmp_path)

    os.unlink(tmp_path)

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(result["overlay_image"], use_container_width=True)

    with col3:
        st.subheader("Results")
        st.markdown("Detection (TensorFlow)")
        if result["tumor_detected"]:
            st.error(f"Tumor detected: {result['detection_confidence']:.1%} confidence")
        else:
            st.success(f"No tumor: {1 - result['detection_confidence']:.1%} confidence")

        st.markdown("---")
        st.markdown("Classification (TensorFlow)")
        st.markdown(
            f"Predicted type: **{result['tumor_type']}** "
            f"({result['class_confidence']:.1%} confidence)"
        )
        if result["models_disagree"]:
            st.warning(
                "The detector and the classifier disagree. They are separate "
                "models, so treat this scan as uncertain rather than trusting "
                "either verdict."
            )

        probs = result["class_probabilities"]
        fig, ax = plt.subplots(figsize=(4, 2.5))
        values = list(probs.values())
        max_val = max(values)
        colors = ["#d9534f" if v == max_val else "#428bca" for v in values]
        ax.barh(list(probs.keys()), values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        coverage = float((result["segmentation_mask"] > 0).mean() * 100)
        st.metric("Tumor region coverage", f"{coverage:.1f}%")

    buf = io.BytesIO()
    Image.fromarray(result["segmentation_mask"]).save(buf, format="PNG")
    st.download_button(
        "Download segmentation mask",
        data=buf.getvalue(),
        file_name="segmentation_mask.png",
        mime="image/png",
    )
