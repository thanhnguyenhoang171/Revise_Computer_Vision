import cv2
import numpy as np

# Load image for img0
img0 = cv2.imread("datasets/face3.jpg")
img0 = cv2.resize(img0, (270, 356))

# Convert image to grayscale
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for img0
keypoints_0, descriptors_0 = sift.detectAndCompute(gray0, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Initialize camera
cap = cv2.VideoCapture("datasets/video3.mp4")
while cap.isOpened():
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for camera frame
    keypoints_cam, descriptors_cam = sift.detectAndCompute(gray_frame, None)

    # Match descriptors using knnMatch
    matches = bf.match(descriptors_0, descriptors_cam)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = bf.knnMatch(descriptors_0, descriptors_cam, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Check if there are enough good matches to estimate homography
    if len(good_matches) >= 10:
        # Extract matched keypoints
        matched_keypoints_0 = np.float32(
            [keypoints_0[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        matched_keypoints_cam = np.float32(
            [keypoints_cam[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Find homography matrix
        H, _ = cv2.findHomography(
            matched_keypoints_0, matched_keypoints_cam, cv2.RANSAC
        )

        # Get corners of the object in img0
        h, w = gray0.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        # Transform corners to get the bounding box in the camera frame
        transformed_corners = cv2.perspectiveTransform(corners, H)

        # Draw the bounding box on the camera frame
        cv2.polylines(gray_frame, [np.int32(transformed_corners)], True, (255, 0, 0), 2)

    # Draw matches on img_with_matches
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
    # Display the result
    cv2.imshow("Face Detection with SIFT", img_with_matches)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
