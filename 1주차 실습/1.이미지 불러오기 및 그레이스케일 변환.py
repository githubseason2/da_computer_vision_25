import cv2 as cv
import numpy as np
import sys

img = cv.imread('./a.jpg')

if img is None:
    sys.exit("파일 로딩 오류")
    
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

tmp = np.stack( (gray_img, gray_img, gray_img), axis=2)
both_imgs = np.hstack((img, tmp))

cv.imshow(both_imgs)

cv.waitKey()
cv.destroyAllWindows()
