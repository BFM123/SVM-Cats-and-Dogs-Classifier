import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from skimage.feature import hog

# CONFIG
IMAGE_SIZE = (256, 256)   # Larger size for better HOG features
DATA_DIR = "Data/train"

MAX_IMAGES_PER_CLASS = 5000  # Adjust based on RAM

# Load images and extract HOG
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
            # HOG descriptor
            hog_feature = hog(
                img_gray,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optionally reduce dimensionality
# HOG features are often compact enough, but you can try PCA:
# pca = PCA(n_components=100)
# X_train_scaled = pca.fit_transform(X_train_scaled)
# X_test_scaled = pca.transform(X_test_scaled)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))