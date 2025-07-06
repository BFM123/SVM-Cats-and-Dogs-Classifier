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
IMAGE_SIZE = (128, 128)   # Increase size for better HOG detail
DATA_DIR = "Data/train"
MAX_IMAGES_PER_CLASS = 3000  # Adjust to your RAM (4000+ recommended)

# Load images and extract optimized HOG
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
            # Optimized HOG parameters
            hog_feature = hog(
                img_gray,
                orientations=12,              # increased from 9 to 12
                pixels_per_cell=(8, 8),       # finer cells
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            features.append(hog_feature)
            labels.append(0 if label_name == "cat" else 1)
            count += 1
            if count >= max_images:
                break
    return np.array(features), np.array(labels)

# Load cats
X_cats, y_cats = load_images_hog(DATA_DIR, "cat", MAX_IMAGES_PER_CLASS)
# Load dogs
X_dogs, y_dogs = load_images_hog(DATA_DIR, "dog", MAX_IMAGES_PER_CLASS)

# Combine
X = np.vstack((X_cats, X_dogs))
y = np.concatenate((y_cats, y_dogs))

print(f"Loaded {X.shape[0]} samples with feature dimension {X.shape[1]}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional: reduce dimensionality
# Comment out if you want full HOG features
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Grid Search for best C and gamma
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.001, 0.0001],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=2, n_jobs=-1)
print("Starting grid search...")
grid.fit(X_train_pca, y_train)

print("Best parameters:", grid.best_params_)

# Predict
y_pred = grid.predict(X_test_pca)

# Save the pipeline components
joblib.dump(grid.best_estimator_, "svm_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(pca, "pca.joblib")

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))