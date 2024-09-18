import os
import glob
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Các tham số
image_size = (64, 128)
positive_img_path = "datasets/positive"
negative_img_path = "datasets/negative"

# Khởi tạo danh sách chứa dữ liệu và nhãn
X = []
Y = []


# Hàm để tải ảnh và tính toán các đặc trưng HOG
def load_images(img_path, label):
    for filename in glob.glob(os.path.join(img_path, "*.png")):
        img = cv2.imread(filename, 0)  # Đọc ảnh dưới dạng ảnh xám
        img = cv2.resize(img, image_size)  # Thay đổi kích thước ảnh
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm="L2-Hys",
            transform_sqrt=True,
        )
        X.append(hog_features)  # Thêm đặc trưng HOG vào danh sách
        Y.append(label)  # Thêm nhãn vào danh sách


# Tải ảnh có người và không có người
load_images(positive_img_path, 1)
load_images(negative_img_path, 0)

# Chuyển đổi danh sách thành mảng numpy
X = np.float32(X)
Y = np.array(Y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình SVM
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")  # Ma trận nhầm lẫn
print(conf_matrix)
print("\nClassification Report:")  # Báo cáo phân loại
print(classification_report(y_test, y_pred))

# Lưu mô hình đã huấn luyện
model_file = "models.dat"
joblib.dump(model, model_file)
print(f"Model saved: {model_file}")  # Lưu mô hình vào tệp tin


import cv2
import joblib
import numpy as np
from imutils.object_detection import non_max_suppression  # Non-maxima suppression
import matplotlib.pyplot as plt  # Thư viện hiển thị ảnh
from skimage import color
from skimage.feature import hog  # Tính năng HOG
from skimage.transform import pyramid_gaussian  # Image pyramid

# Các tham số
model_file = "models.dat"  # Đường dẫn đến tệp tin mô hình đã lưu
input_file = "datasets/test/img1.jfif"  # Đường dẫn đến tệp tin ảnh thử nghiệm
image_size = (64, 128)
step_size = (9, 9)  # Kích thước bước di chuyển cửa sổ
downscale = 1.05  # Hệ số giảm kích thước ảnh

# Tải mô hình đã huấn luyện
model = joblib.load(model_file)


# Hàm tạo các cửa sổ trượt trên ảnh
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


# Tải ảnh đầu vào
image = cv2.imread(input_file)
image = cv2.resize(image, (600, 400))  # Thay đổi kích thước ảnh

# Danh sách để lưu trữ các phát hiện
detections = []
scale = 0

# Phát hiện người bằng cách sử dụng cửa sổ trượt và image pyramid
for im_scaled in pyramid_gaussian(image, downscale=downscale):
    if im_scaled.shape[0] < image_size[1] or im_scaled.shape[1] < image_size[0]:
        break

    for x, y, window in sliding_window(im_scaled, image_size, step_size):
        if window.shape[0] != image_size[1] or window.shape[1] != image_size[0]:
            continue

        window = color.rgb2gray(window)  # Chuyển đổi cửa sổ sang ảnh xám
        fd = hog(
            window,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm="L2-Hys",
        )
        fd = fd.reshape(1, -1)
        pred = model.predict(fd)

        if pred == 1 and model.decision_function(fd) > 0.5:
            detections.append(
                (
                    int(x * (downscale**scale)),
                    int(y * (downscale**scale)),
                    model.predict(fd),
                    int(image_size[0] * (downscale**scale)),
                    int(image_size[1] * (downscale**scale)),
                )
            )

    scale += 1

# Vẽ các phát hiện lên ảnh
clone = image.copy()
clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang RGB để hiển thị
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
sc = np.array(sc)
pick = non_max_suppression(
    rects, probs=sc, overlapThresh=0.45
)  # Áp dụng non-maxima suppression

for x1, y1, x2, y2 in pick:
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật
    cv2.putText(
        clone,
        "Pedestrian",  # Văn bản ghi "Pedestrian"
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2,
    )

plt.imshow(clone)  # Hiển thị ảnh
plt.axis("off")
plt.show()
