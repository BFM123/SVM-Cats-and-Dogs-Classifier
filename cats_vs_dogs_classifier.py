import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# CONFIG
IMAGE_SIZE = (64, 64)   # Resize images to 64x64
DATA_DIR = "/path/to/dogs-vs-cats/train"
MAX_IMAGES_PER_CLASS = 2000  # To avoid memory issues

# Load images
def load_images(data_dir, label_name, max_images):
    images = []
    labels = []
    count = 0
    for fname in tqdm(os.listdir(data_dir)):
        if label_name in fname:
            img_path = os.path.join(data_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.flatten())
            labels.append(0 if label_name == "cat" else 1)
            count += 1
            if count >= max_images:
                break
    return np.array(images), np.array(labels)

# Load cats
X_cats, y_cats = load_images(DATA_DIR, "cat", MAX_IMAGES_PER_CLASS)
# Load dogs
X_dogs, y_dogs = load_images(DATA_DIR, "dog", MAX_IMAGES_PER_CLASS)

# Combine
X = np.vstack((X_cats, X_dogs))
y = np.concatenate((y_cats, y_dogs))

print(f"Loaded {X.shape[0]} images with shape {X.shape}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction (PCA)
pca = PCA(n_components=150)  # Adjust as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_pca, y_train)

# Predict
y_pred = svm.predict(X_test_pca)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))