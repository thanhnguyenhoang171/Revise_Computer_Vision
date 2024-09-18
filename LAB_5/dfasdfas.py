import cv2
import matplotlib.pyplot as plt

# Đường dẫn đến tệp ảnh đầu vào
input_image_path = "datasets/test/img1.jfif"

# Tải ảnh đầu vào
image = cv2.imread(input_image_path)
image = cv2.resize(image, (600, 400))  # Thay đổi kích thước ảnh
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang ảnh xám

# Khởi tạo HOG descriptor và SVM detector được huấn luyện sẵn
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Phát hiện người đi bộ
rects, weights = hog.detectMultiScale(
    gray_image, winStride=(8, 8), padding=(8, 8), scale=1.05
)

# Vẽ các khung chữ nhật xung quanh người đi bộ được phát hiện
for x, y, w, h in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Hiển thị ảnh kết quả
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
