import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from skimage.feature import hog
import joblib

# CONFIG
IMAGE_SIZE = (128, 128)
DATA_DIR = "Data/train"
MAX_IMAGES_PER_CLASS = 3000  # Adjust for memory

HOG_PARAMS = {
    "orientations": 12,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True
}

# Load images
def load_images_hog(data_dir, label_name, max_images):
    features = []
    labels = []
    count = 0
    for fname in tqdm(os.listdir(data_dir)):
        if label_name in fname:
            img_path = os.path.join(data_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_feature = hog(
                img_gray,
                **HOG_PARAMS
            )
            features.append(hog_feature)
            labels.append(0 if label_name == "cat" else 1)
            count += 1
            if count >= max_images:
                break
    return np.array(features), np.array(labels)

# Load data
X_cats, y_cats = load_images_hog(DATA_DIR, "cat", MAX_IMAGES_PER_CLASS)
X_dogs, y_dogs = load_images_hog(DATA_DIR, "dog", MAX_IMAGES_PER_CLASS)
X = np.vstack((X_cats, X_dogs))
y = np.concatenate((y_cats, y_dogs))

print(f"Loaded {X.shape[0]} samples with shape {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Grid search
param_grid = {
    "C": [1, 10],
    "gamma": ["scale", 0.0001],
    "kernel": ["rbf"]
}
grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=2)
grid.fit(X_train_pca, y_train)

print("Best parameters:", grid.best_params_)

y_pred = grid.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_estimator_, "models/svm_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca.joblib")
print("Models saved in 'models/'")