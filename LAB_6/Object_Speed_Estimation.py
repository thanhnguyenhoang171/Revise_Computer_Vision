import cv2
import numpy as np


# Hàm theo dõi đối tượng giữa hai khung hình I1 và I2 trong vùng ROI
def trackObject(I1, I2, roi):
    # Tham số cho hàm Lucas-Kanade Optical Flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Chuyển đổi hai khung hình sang ảnh xám
    grayI1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    grayI2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Tính toán Optical Flow cho vùng ROI
    optical_flow, status, err = cv2.calcOpticalFlowPyrLK(
        grayI1, grayI2, roi, None, **lk_params
    )

    # Tính toán tốc độ di chuyển của đối tượng
    speedMagnitude = np.sqrt(optical_flow[0][0][0] ** 2 + optical_flow[0][0][1] ** 2)

    # Lấy các điểm tốt trong optical flow (những điểm có status == 1)
    good_points = optical_flow[status == 1]

    # Cập nhật ROI với các điểm tốt
    roi = good_points.reshape(-1, 1, 2)
    return roi, speedMagnitude


# Mở video
cap = cv2.VideoCapture("Datasets/video.mp4")

# Đọc khung hình đầu tiên
ret, prev = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Lựa chọn ROI bằng tay
x, y, w, h = cv2.selectROI("Select ROI", prev, False)

# Tạo vùng ROI ban đầu dưới dạng tọa độ trung tâm của khung chữ nhật đã chọn
roi = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)

# Đóng cửa sổ chọn ROI
cv2.destroyWindow("Select ROI")

# Vòng lặp để xử lý từng khung hình trong video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Theo dõi đối tượng giữa khung hình trước và khung hình hiện tại
    roi, speedMagnitude = trackObject(prev, frame, roi)

    # Lấy tọa độ của đối tượng
    x, y = roi[0][0]

    # Vẽ khung chữ nhật xung quanh đối tượng
    cv2.rectangle(
        frame,
        (int(x - w / 2), int(y - h / 2)),
        (int(x + w / 2), int(y + h / 2)),
        (0, 255, 0),
        2,
    )

    # Hiển thị tốc độ di chuyển của đối tượng
    cv2.putText(
        frame,
        f"Speed: {speedMagnitude:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # Hiển thị khung hình hiện tại
    cv2.imshow("Object Tracking", frame)

    # Lưu lại khung hình hiện tại cho lần xử lý tiếp theo
    prev = frame.copy()

    # Thoát khỏi vòng lặp nếu nhấn phím 'ESC'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Giải phóng tài nguyên và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
