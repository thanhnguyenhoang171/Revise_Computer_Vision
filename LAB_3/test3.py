import cv2

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Open video
cap = cv2.VideoCapture("datasets/video2.mp4")

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade classifier
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Loop over each detected face
    for x, y, w, h in faces:
        # Crop the face region from the frame
        face_region = frame[y : y + h, x : x + w]

        # Detect keypoints and descriptors in the face region using SIFT
        keypoints, descriptors = sift.detectAndCompute(face_region, None)

        # Match descriptors with pre-defined facial keypoints or descriptors
        # Apply geometric verification to filter out false matches

        # Draw bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
