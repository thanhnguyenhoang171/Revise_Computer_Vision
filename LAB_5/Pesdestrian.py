import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog
from sklearn.model_selection import train_test_split


# Paths to extracted folders
positive_images_path = "datasets/positive"
negative_images_path = "datasets/negative"


# Image size and HOG parameters
image_size = (64, 128)


# Feature extraction function
def extract_hog_features(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) != 2:
        raise ValueError("Only grayscale images with two spatial dimensions are supported.")
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    )


# Load and preprocess images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, image_size)
        images.append(img)
    return images


positive_images = load_images(positive_images_path)
negative_images = load_images(negative_images_path)

# Feature extraction
X = np.array([extract_hog_features(img) for img in positive_images + negative_images])
y = np.array([1] * len(positive_images) + [0] * len(negative_images))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the SVM classifier
svm_classifier = SVC(kernel="rbf", C=10, gamma=0.01)
svm_classifier.fit(X_train, y_train)

# Testing and performance evaluation
y_pred = svm_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Pedestrian detection function
def detect_pedestrians(image):
    # Perform sliding window approach with HOG features and SVM classifier
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect pedestrians
    rects, _ = hog.detectMultiScale(image, winStride=(9, 9), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


# Test pedestrian detection on example images
test_images_path = "datasets/test/img.jpg"
test_images = load_images(test_images_path)

test_image_resized = cv2.resize(test_images, image_size)
detected_image = detect_pedestrians(test_image_resized)

# Display the result
cv2.imshow("Detected Pedestrians", detected_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
