from skimage.feature import hog
import joblib, glob, os, cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Training model
X = []
Y = []

positive_img_path = "datasets/positive"
negative_img_path = "datasets/negative"

# load positive
for filename in glob.glob(os.path.join(positive_img_path, "*.png")):
    file_load = cv2.imread(filename, 0)
    file_load = cv2.resize(file_load, (64, 128))
    hog_features = hog(
        file_load,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm="L2-Hys",
        transform_sqrt=True,
    )
    X.append(hog_features)  # Append the HOG features
    Y.append(1)

# load negative
for filename in glob.glob(os.path.join(negative_img_path, "*.png")):
    file_load = cv2.imread(filename, 0)
    file_load = cv2.resize(file_load, (64, 128))
    hog_features = hog(
        file_load,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm="L2-Hys",
        transform_sqrt=True,
    )
    X.append(hog_features)  # Append the HOG features
    Y.append(0)


X = np.float32(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2
)
print("Train Data: ", len(X_train))
print("Train Labels (1, 0)", len(y_train))

model = SVC (kernel='rbf')
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

joblib.dump(model, "models.dat")
print("Model save: {}'.format ('models.dat)")

# # Detect pedestrian
# from imutils.object_detection import non_max_suppression
# import imutils
# from skimage import color
# from skimage.transform import pyramid_gaussian

# model_path = "models.dat"
# input_path = "datasets/test/img.jpg"
# output_path = "datasets/test/img_out.jpg"
# image = cv2.imread(input_path)
# image = cv2.resize(image, (474, 343))
# size = (192, 320)
# step_size = (9, 9)
# downscale = 1.05

# detections = []

# scale = 0

# model = joblib.load(model_path)


# def sliding_window(image, window_size, step_size):
#     step_x, step_y = step_size
#     for y in range(0, image.shape[0], step_y):
#         for x in range(0, image.shape[1], step_x):
#             yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


# for im_scaled in pyramid_gaussian(image, downscale=downscale):
#     if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
#         break

#     for x, y, window in sliding_window(im_scaled, size, step_size):
#         if window.shape[0] != size[1] or window.shape[1] != size[0]:
#             continue
#         window_gray = color.rgb2gray(window)  # Convert to grayscale
#         fd = hog (
#             window_gray,  # Use grayscale image
#             orientations=9,
#             pixels_per_cell=(8, 8),
#             visualize=False,
#             cells_per_block=(2, 2),
#             block_norm="L2-Hys",
#             transform_sqrt=True,
#         )
#         fd = fd.reshape(1, -1)

#     predict = model.predict(fd)

#     if predict == 1:
#         if model.decision_function(fd) > 0.5:
#             detections.append(
#                 (
#                     int(x * (downscale * scale)),
#                     int(y * (downscale * scale)),
#                     model.predict(fd),
#                     int(size[0] * (downscale**scale)),
#                     int(size[1] * (downscale**scale)),
#                 )
#             )

#     scale += 1

# clone = image.copy()
# clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
# rects = np.array([[x, y + w, y + h] for (x, y, _, w, h) in detections])
# sc = [score[0] for (x, y, score, w, h) in detections]
# print("sc: ", sc)
# sc = np.array(sc)
# pick = non_max_suppression(rects, probs=sc, overlapThresh=0.5)
# for x1, y1, x2, y2 in pick:
#     cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0))
#     cv2.putText(clone, "Human", (x1 - 2, y1 - 2, 1, 0.75, (255, 255, 0), 1))

# cv2.imwrite(output_path, clone)
# plt.imshow(clone)
