# Flower Classifier

A small Streamlit app for `flower_cnn_model.h5`.

## Run locally

Use Python 3.10 or 3.11, install the dependencies, and start Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app resizes uploaded RGB images to 150 × 150 pixels, matching the model input.
The model's original class names and index order are not included in the supplied
artifact, so it displays `Class 0`, `Class 1`, etc. by default. To show meaningful
names, create `class_names.txt` with one label per line in the exact training order.

The app currently assumes the common five-flower output order: Daisy, Dandelion,
Rose, Sunflower, Tulip. The H5 file does not include its label mapping, so confirm
that order against the training data. You can override it by creating
`class_names.txt` with one label per line in the model output order.

The model artifact is about 9 MB and is stored locally beside the app.
