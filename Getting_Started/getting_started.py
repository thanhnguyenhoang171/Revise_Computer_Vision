import cv2

videoPath = "datasets/video_book"
imgPath = "datasets/logo-ou.png"
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
resized_img = cv2.resize(img, (30, 30))
overlay = resized_img

cap = cv2.VideoCapture(videoPath)
#cap = cv2.VideoCapture(0)


#if not cap.isOpened():
    #print("Không thể kết nối camera. Vui lòng kiểm tra lại.")
    #exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 768))
    cv2.putText(
        frame,
        "Minh Khue",
        (10, 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,  # Adding the missing fontScale argument
        color=(0, 0, 0),  # Adding font color (white in this case)
        thickness=2,  # Adding text thickness
    )
    height, width, _ = overlay.shape
    roi = frame[100 : 100 + height, 150 : 150 + width]
    overlay = cv2.resize(overlay, (roi.shape[1], roi.shape[0]))

    frame[25 : 25 + height, 25 : 25 + width] = overlay
    # Hiển thị frame
    cv2.imshow("Frame", frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng các tài nguyên
cap.release()
cv2.destroyAllWindows()
