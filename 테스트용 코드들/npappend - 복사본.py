import cv2 as cv

drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, img, img_copy

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img = img_copy.copy()

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        w = abs(x - ix)
        h = abs(y - iy)
        x0 = min(ix, x)
        y0 = min(iy, y)
        rect = (x0, y0, w, h)
        print(f"선택한 영역 (x, y, width, height): {rect}")
        cv.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

# 이미지 불러오기
img = cv.imread('a.jpg')
img_copy = img.copy()

cv.namedWindow('Drag to Select')
cv.setMouseCallback('Drag to Select', draw_rectangle)

while True:
    cv.imshow('Drag to Select', img)
    if cv.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cv.destroyAllWindows()
