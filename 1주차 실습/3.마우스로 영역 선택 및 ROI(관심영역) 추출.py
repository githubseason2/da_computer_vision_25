import cv2 as cv
import sys

img=cv.imread('a.jpg')
if img is None:
    sys.exit('파일을 불러올 수 없었습니다...')
img_ori=img.copy()
img_tmp=img.copy()

def draw(event, x, y, flags, param):
    global ix, iy, img, img_tmp, ismoved
    if event==cv.EVENT_LBUTTONDOWN:
        ix, iy=x, y
        ismoved=False
    elif event==cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        ismoved=True
        img = img_tmp.copy()
        cv.rectangle(img,(ix, iy), (x, y), (255,0,0), 2)
        cv.imshow('ROI', img)
    elif event==cv.EVENT_LBUTTONUP:
        if(ismoved):
            img = img_tmp[min(iy, y):max(iy, y), min(ix, x):max(ix, x)]
            img_tmp= img.copy()
            cv.imshow('ROI', img)
    
cv.namedWindow("ROI")
cv.imshow('ROI',img)
cv.setMouseCallback('ROI', draw)

while(1):
    key= cv.waitKey(1)
    if key==ord('r'):
        img=img_ori.copy()
        img_tmp= img_ori.copy()
        cv.imshow('ROI',img)
    elif key==ord('s'):
        cv.imwrite('b.png', img)    

