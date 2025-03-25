import cv2 as cv
import numpy as np
import sys

img = cv.imread('tree.png')
if img is None:
    print("이미지를 찾을 수 없습니다.")
    sys.exit("파일 로딩 오류")


rows, cols = img.shape[:2]

M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.5)


rotated_scaled = cv.warpAffine(img, M, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)

padded_original = cv.copyMakeBorder(img, 0, int(rows * 1.5) - rows, 0, int(cols * 1.5) - cols,
                                     borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

combined = np.hstack((padded_original, rotated_scaled))

cv.imshow("image", combined)
cv.waitKey(0)
cv.destroyAllWindows()
