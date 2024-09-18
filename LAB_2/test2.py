import cv2
import numpy as np


# Hàm để load và chuẩn bị logo
def prepare_logo(logo_path):
    logo = cv2.imread(logo_path)
    logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
    # Tính histogram của logo
    roi_hist = cv2.calcHist([logo_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # Chuẩn hóa histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


# Hàm để nhận dạng logo trong frame
def detect_logo(frame, roi_hist):
    # Chuyển frame sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tính histogram back projection
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    # Áp dụng phép lọc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)
    # Thresholding
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Đường dẫn đến video
video_path = "LAB_2/data/video1.mp4"
# Đường dẫn đến logo
logo_path = "LAB_2/data/Untitled.png"

# Load logo
roi_hist = prepare_logo(logo_path)

# Mở video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Nhận dạng logo trong frame
        contours = detect_logo(frame, roi_hist)
        # Vẽ contours lên frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        # Hiển thị frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
