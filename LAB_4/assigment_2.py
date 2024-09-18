import cv2
def load_images(img_path1, img_path2, img_path3):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img3 = cv2.imread(img_path3)
    return img1, img2, img3


def stitch_images(img1, img2, img3):
    stitcher = cv2.Stitcher_create()

    status, result = stitcher.stitch([img1, img2, img3])

    if status == cv2.Stitcher_OK:
        result = cv2.resize(result, (1080, 600))
        cv2.imshow("Stitched Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed.")


img1_path = "datasets/img1.jpg"
img2_path = "datasets/img2.jpg"
img3_path = "datasets/img3.jpg"
img1, img2, img3 = load_images(img1_path, img2_path, img3_path)
stitch_images(img1, img2, img3)
