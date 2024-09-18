import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation
positive_images_path = "datasets/positive"  # Thư mục chứa hình ảnh có người đi bộ
negative_images_path = "datasets/negative"  # Thư mục chứa hình ảnh không có người đi bộ

# Step 2: Image Preprocessing
image_size = (96, 160)


# Step 3: Feature Extraction
def extract_hog_features(image):
    if len(image.shape) == 3:  # Kiểm tra nếu ảnh có kênh màu
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )  # Chuyển sang ảnh đen trắng nếu có kênh màu
    if len(image.shape) != 2:  # Kiểm tra nếu ảnh không có hai kích thước không gian
        raise ValueError(
            "Only grayscale images with two spatial dimensions are supported."
        )
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    )


# Step 4: Load and preprocess images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, image_size)
        images.append(img)
    return images


positive_images = load_images(positive_images_path)
negative_images = load_images(negative_images_path)

# Step 5: Feature Extraction
X = np.array([extract_hog_features(img) for img in positive_images + negative_images])
y = np.array([1] * len(positive_images) + [0] * len(negative_images))

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Training the SVM classifier
svm_classifier = SVC(
    kernel="rbf", C=10, gamma=0.01
)  # RBF kernel with tuned C and gamma
svm_classifier.fit(X_train, y_train)

# Step 8: Testing and Performance Evaluation
y_pred = svm_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 9: Pedestrian Detection
def detect_pedestrians(image):
    # Perform sliding window approach with HOG features and SVM classifier
    rects, _ = cv2.HOGDescriptor().detectMultiScale(
        image, winStride=(8, 8), padding=(32, 32), scale=1.05
    )
    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


# Test pedestrian detection on an example image
test_image = cv2.imread("datasets/test/img.jpg")
test_image_resized = cv2.resize(test_image, image_size)
detected_image = detect_pedestrians(test_image_resized)

# Display the result
cv2.imshow("Detected Pedestrians", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
