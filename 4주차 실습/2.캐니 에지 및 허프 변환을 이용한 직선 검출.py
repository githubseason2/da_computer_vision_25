import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('Tower.jpg')
if(img is None):
    print("File Not Found")
    exit()
img_original=img.copy()
if(img is None):
    exit()
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray,100,200)

lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=125, minLineLength=100, maxLineGap=5)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

img_rgb = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
img_with_lines_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image with Detected Lines')
plt.imshow(img_with_lines_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
