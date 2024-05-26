import cv2

# đọc ảnh 1 và ảnh 2
img1 = cv2.imread("datasets/pic0.jpg")
img2 = cv2.imread("datasets/pic1.jpg")

# Chuyển ảnh sang không gian màu xám
grayimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Tạo SIFT
sift = cv2.SIFT_create()

# các tham số của SIFT
sift.setContrastThreshold(0.03) # Low-Constrast Extrema
sift.setEdgeThreshold(5) # Edge-Like Extrema

# Tính các keypoint và descriptor của ảnh 1 và 2
keypoints1, descriptors1 = sift.detectAndCompute(grayimg1, None) # None là mask (mặt nạ)
keypoints2, descriptors2 = sift.detectAndCompute(grayimg2, None) # None là mask (mặt nạ)

# Cách 1: Dùng BF để matching
# Tạo đối tượng BFMatcher để so khớp keypoints và tính toán matches
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Sử dụng thuật toán k-nearest neighbors để tìm các điểm khớp
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

#----------------------------------------------------------
# # Cách 2: Dùng FLANN để matching
# # Tạo đối tượng FLANN Matcher
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#-------------------------------------------------------------

# Lọc các điểm khớp dựa trên mức độ giống nhau của chúng
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

# Vẽ các điểm khớp được lọc và lưu vào img 3
img3 = cv2.drawMatchesKnn(grayimg1, keypoints1, grayimg2, keypoints2, good, None, matchColor=(0, 255, 0), matchesMask=None, singlePointColor=(255, 0, 0), flags=0)

# result
img3 = cv2.resize(img3, (1380, 720))

cv2.imshow("SIFT Matching", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

