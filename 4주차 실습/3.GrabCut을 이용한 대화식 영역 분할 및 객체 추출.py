import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('tower.jpg')
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

mask = np.zeros(img.shape[:2], np.uint8)


bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


x = int(input("사각형의 x 좌표를 입력하세요: "))
y = int(input("사각형의 y 좌표를 입력하세요: "))
width = int(input("사각형의 width를 입력하세요: "))
height = int(input("사각형의 height를 입력하세요: "))

rect = (x, y, width, height)


cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)


mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')


result = img * mask2[:, :, np.newaxis]


img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mask_rgb = mask2 * 255
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Mask Image")
plt.imshow(mask_rgb, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Background Removed")
plt.imshow(result_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
