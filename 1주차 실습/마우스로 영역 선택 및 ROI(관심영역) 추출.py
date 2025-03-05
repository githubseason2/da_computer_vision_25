import cv2 as cv
import sys

img=cv.imread('a.jpg')

if img is None:
    sys.exit('파일을 불러올 수 없었습니다...')

def draw(event, x, y, flags, param):
    global ix, iy
    if event==cv.EVENT_LBUTTONDOWN:
        ix, iy=x, y
    elif event==cv.EVENT_LBUTTONUP:
        print(ix, iy, x, y)

cv.namedWindow("ROI")
cv.imshow('ROI',img)
cv.setMouseCallback('ROI', draw)

