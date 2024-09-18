import cv2
import numpy as np

# Load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create SIFT object
sift = cv2.SIFT_create()

# Load image or video
cap = cv2.VideoCapture("datasets/video2.mp4")  # Or specify the path to your image

while cap.isOpened():
    # Capture frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using cascade classifier
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100)
    )
    
    # # Detect keypoints and compute descriptors for frame
    # keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

    # Draw green rectangles around detected faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # # Draw keypoints on frame
    # frame_with_keypoints = cv2.drawKeypoints(
    #     frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # )

    # Display the result
    cv2.imshow("Face Detection with SIFT and Cascade Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
