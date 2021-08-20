import cv2

src = 1 # 0- internal camera, 1 - USB??? 1- failed (driver?)
cap = cv2.VideoCapture(src)
cap.set(3, 1280)
cap.set(4,720)

while True:
    _, img = cap.read()
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key ==27:
        break

cv2.cap.release()
cv2.destroyAllWindows()

