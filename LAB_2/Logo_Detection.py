import cv2
import numpy as np

# Read logo image and resize it to match video frame size
roi = cv2.imread("LAB_2/data/pepsi.png")
hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

# Histogram of logo
M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])


cap = cv2.VideoCapture("LAB_2/data/pepsi.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsvt = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Calculate back projection
    B = cv2.calcBackProject([hsvt], [0, 1], M, [0, 180, 0, 256], 1)

    # Thresholding
    ret, thresh = cv2.threshold(B, 10, 255, 0)

    # Create result image
    result = cv2.merge((thresh, thresh, thresh))
    result = cv2.bitwise_and(frame, result)

    # Display result
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
