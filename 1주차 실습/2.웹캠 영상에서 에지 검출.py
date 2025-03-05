import cv2 as cv
import sys

cap=cv.VideoCapture(0, cv.CAP_DSHOW)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL) 
while True:
    ret, frame = cap.read()
    if not ret:
        print("이미지를 불러오는데 실패했습니다.")
        break
    key=cv.waitKey(1)
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    extratedEdge=cv.Canny(gray_img, 110, 350)

    tmp = np.stack( (extratedEdge, extratedEdge, extratedEdge), axis=2)
    both_imgs = np.hstack((img, tmp))

    key=cv.waitKey(1)
    if key==ord('q'):
        cv.destroyAllWindows()
        break
    cv2.imshow('frame', both_image)
cap.release()


    
