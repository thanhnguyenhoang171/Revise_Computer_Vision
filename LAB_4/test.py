import cv2
import numpy as np


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    listPoints1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(
        -1, 1, 2
    )
    tempPoints2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )
    listPoints2 = cv2.perspectiveTransform(tempPoints2, H)
    listPoints = np.concatenate((listPoints1, listPoints2), axis=0)
    [xMin, yMin] = np.int32(listPoints.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(listPoints.max(axis=0).ravel() + 0.5)
    translationDist = [-xMin, -yMin]
    HTranslation = np.array(
        [[1, 0, translationDist[0]], [0, 1, translationDist[1]], [0, 0, 1]]
    )
    output_img = cv2.warpPerspective(
        img2, HTranslation.dot(H), (xMax - xMin, yMax - yMin)
    )
    output_img[
        translationDist[1] : rows1 + translationDist[1],
        translationDist[0] : cols1 + translationDist[0],
    ] = img1
    return output_img


def stitchImages(img1, img2):
    # Detect keypoints
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Find best matches
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches found.")
        return None
    # Match keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    # Calculate homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Make the function resistant to outliers

    # Warp perspective of the second image
    result = warpImages(img1, img2, H)
    return result


# Load images
img1 = cv2.imread("datasets/img2.jpg")


img2 = cv2.imread("datasets/img1.jpg")

panorama = stitchImages(img1, img2)

# Display the result
cv2.imshow("Stitched Image", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
