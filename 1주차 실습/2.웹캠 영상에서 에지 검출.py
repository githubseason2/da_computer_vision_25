import cv2 as cv
import numpy as np
import sys

cap=cv.VideoCapture(0, cv.CAP_DSHOW)
cv.namedWindow('frame') 
while True:
    ret, frame = cap.read()
    if not ret:
        print("이미지를 불러오는데 실패했습니다.")
        break
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    extratedEdge=cv.Canny(gray_img, 50, 50)

    tmp = np.stack( (extratedEdge, extratedEdge, extratedEdge), axis=2)
    both_imgs = np.hstack((frame, tmp))

    key=cv.waitKey(1)
    if key==ord('q'):
        cv.destroyAllWindows()
        break
    cv.imshow('frame', both_imgs)
cap.release()


    
