import cv2

img0 = cv2.imread("datasets/pic0.jpg")
img0 = cv2.resize(img0, (356, 270))

img1 = cv2.imread("datasets\pic1.jpg")
img1 = cv2.resize(img1, (960, 1280))


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

sift.setContrastThreshold(0.03)
sift.setEdgeThreshold(5)

keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)
keypoints_0, descriptors_0 = sift.detectAndCompute(gray0, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.match(descriptors_0, descriptors_1)
matches = sorted(matches, key=lambda x: x.distance)
img2 = cv2.drawMatches(
    gray0,
    keypoints_0,
    gray1,
    keypoints_1,
    matches[:50],
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    flags=0,
)

matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)

good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(
    gray0,
    keypoints_0,
    gray1,
    keypoints_1,
    good,
    None,
    matchColor=(0, 255, 0),
    matchesMask=None,
    singlePointColor=(255, 0, 0),
    flags=0,
)
img3 = cv2.resize(img3, (1380, 720))

cv2.imshow("SIFT Matching", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
