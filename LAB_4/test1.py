import cv2
import numpy as np


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Tính toán các điểm đỉnh của ảnh đầu ra
    corners1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(
        -1, 1, 2
    )
    corners2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # Tính toán kích thước ảnh đầu ra dựa trên tất cả các điểm được biến đổi
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    output_width, output_height = x_max - x_min, y_max - y_min

    # Dịch chuyển ảnh thứ hai
    translation_dist = [-x_min, -y_min]
    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )
    img2_warped = cv2.warpPerspective(
        img2, H_translation.dot(H), (output_width, output_height)
    )

    # Ghép ảnh
    output_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_img[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = img1
    output_img = cv2.add(output_img, img2_warped)

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
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Warp perspective of the second image
    result = warpImages(img1, img2, H)
    return result


# Load images
img1 = cv2.imread("datasets\img1.jpg")


img2 = cv2.imread("datasets\img2.jpg")


# Stitch images
panorama = stitchImages(img1, img2)

# Display the result
cv2.imshow("Stitched Image", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
