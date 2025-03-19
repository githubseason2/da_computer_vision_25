import cv2 as cv
import numpy as np
import sys



img = cv.imread('./JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)
if img is None:
    sys.exit("이미지를 불러올 수 없습니다.")
#print(img.shape)

t, binary=cv.threshold(img[:,:,3],0,255, cv.THRESH_BINARY)

#print(binary.shape)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
dilation = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel)
erosion = cv.morphologyEx(binary, cv.MORPH_ERODE, kernel)
opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

result = np.hstack([binary, dilation, erosion, opening, closing])
result_bgr=cv.cvtColor(result, cv.COLOR_GRAY2BGR)
cv.imshow("morphology", result)
cv.waitKey(0)
cv.destroyAllWindows()
