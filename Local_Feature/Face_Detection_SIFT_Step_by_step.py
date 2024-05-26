import cv2
import numpy as np

# Đọc ảnh cho img0
img0 = cv2.imread("datasets/face3.jpg")
img0 = cv2.resize(img0, (270, 356))

# Chuyển ảnh sang không gian màu xám
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# Tạo đối tượng SIFT
sift = cv2.SIFT_create()

# Phát hiện keypoints và tính toán descriptors cho img0
keypoints_0, descriptors_0 = sift.detectAndCompute(gray0, None)

# Tạo đối tượng BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Khởi tạo camera
cap = cv2.VideoCapture("datasets/video3.mp4")
while cap.isOpened():
    # Lấy khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển khung hình sang không gian màu xám
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện keypoints và tính toán descriptors cho khung hình từ camera
    keypoints_cam, descriptors_cam = sift.detectAndCompute(gray_frame, None)

    # So khớp descriptors sử dụng knnMatch
    matches = bf.knnMatch(descriptors_0, descriptors_cam, k=2)

    # Áp dụng kiểm tra tỷ lệ
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Kiểm tra nếu có đủ các điểm khớp tốt để ước lượng homography
    if len(good_matches) >= 10:
        # Trích xuất các điểm keypoints khớp nhau
        matched_keypoints_0 = np.float32(
            [keypoints_0[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        matched_keypoints_cam = np.float32(
            [keypoints_cam[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Tìm ma trận homography
        H, _ = cv2.findHomography(
            matched_keypoints_0, matched_keypoints_cam, method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            mask=None,
            maxIters = 2000,
            confidence=0.995
        )

        # Lấy các góc của đối tượng trong img0
        h, w = gray0.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        # Biến đổi các góc để lấy bounding box trong khung hình từ camera
        transformed_corners = cv2.perspectiveTransform(corners, H)

        # Vẽ bounding box trên khung hình từ camera
        cv2.polylines(gray_frame, [np.int32(transformed_corners)], True, (255, 0, 0), 2)

    # Vẽ các điểm khớp trên img_with_matches
    img_with_matches = cv2.drawMatches(
        gray0,
        keypoints_0,
        gray_frame,
        keypoints_cam,
        good_matches[:50],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=0,
    )
    img_with_matches = cv2.resize(img_with_matches, (1380, 720))
    # Hiển thị kết quả
    cv2.imshow("Face Detection with SIFT", img_with_matches)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
