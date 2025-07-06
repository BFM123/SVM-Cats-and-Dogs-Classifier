import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog

# Load models
svm_model = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca.joblib")

IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 12,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True
}

st.title("🐱🐶 Cats vs. Dogs Classifier (SVM)")

uploaded_file = st.file_uploader("Upload a cat or dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    hog_feature = hog(
        img_gray,
        **HOG_PARAMS
    ).reshape(1, -1)

    hog_scaled = scaler.transform(hog_feature)
    hog_pca = pca.transform(hog_scaled)
    prediction = svm_model.predict(hog_pca)[0]
    label = "Cat 🐱" if prediction == 0 else "Dog 🐶"

    st.markdown(f"### Prediction: **{label}**")