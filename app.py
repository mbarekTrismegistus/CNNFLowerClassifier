"""Streamlit interface for the bundled flower image classifier."""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


MODEL_PATH = Path(__file__).with_name("flower_cnn_model.h5")
IMAGE_SIZE = (150, 150)
DEFAULT_LABELS = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]


@st.cache_resource
def load_model():
    # Import lazily so the app can show a useful message when TensorFlow is missing.
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input},
    )


def get_labels(class_count: int) -> list[str]:
    """Read optional labels, falling back to the common five-flower class order."""
    labels_file = Path(__file__).with_name("class_names.txt")
    if labels_file.exists():
        labels = [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
        if len(labels) == class_count:
            return labels
        st.warning(
            f"class_names.txt has {len(labels)} labels, but the model returns "
            f"{class_count} classes. Showing class indices instead."
        )
    if class_count == len(DEFAULT_LABELS):
        return DEFAULT_LABELS
    return [f"Class {index}" for index in range(class_count)]


st.set_page_config(page_title="Flower Classifier", page_icon="🌼", layout="centered")
st.title("🌼 Flower classifier")
st.caption("Upload a photo to identify the flower.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read that image: {exc}")
        st.stop()

    preview_col, action_col = st.columns([1, 1.6], vertical_alignment="center")
    with preview_col:
        st.image(image, caption="Image preview", width=260)
    with action_col:
        st.write("Ready to identify this flower")
        classify = st.button("🌼  Identify flower", type="primary", use_container_width=True)

    if classify:
        try:
            with st.status("Loading model and analyzing image…", expanded=False) as status:
                model = load_model()
                resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
                predictions = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)
                labels = get_labels(len(predictions))
                best_index = int(np.argmax(predictions))
                status.update(label="Identification complete", state="complete")

            result_col, scores_col = st.columns([1, 1.5])
            with result_col:
                st.success("Most likely match")
                st.subheader(f"🌸 {labels[best_index]}")
                st.metric("Model score", f"{float(predictions[best_index]):.1%}")
            with scores_col:
                st.markdown("#### Class scores")
                st.caption("Scores may not be calibrated probabilities.")
                ranked = np.argsort(predictions)[::-1]
                for index in ranked:
                    st.write(f"**{labels[int(index)]}** · {float(predictions[index]):.1%}")
                    st.progress(float(np.clip(predictions[index], 0, 1)))
        except Exception as exc:
            st.error(f"Could not load or run the model: {exc}")
            st.info("Install the packages in requirements.txt, then restart the app.")

with st.expander("About class names"):
    st.write(
        "This app uses the common five-flower order: Daisy, Dandelion, Rose, Sunflower, "
        "Tulip. The model file does not include its label mapping, so verify this order "
        "against the training data. To override it, add one label per line to "
        "`class_names.txt` in the model's output order."
    )
