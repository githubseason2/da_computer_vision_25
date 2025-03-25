import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt



img = cv.imread('./mistyroad.jpg')
if img is None:
    sys.exit("이미지를 불러올 수 없습니다.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

t, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

hist_gray = cv.calcHist([gray], [0], None, [256], [0, 256])
hist_binary = cv.calcHist([binary], [0], None, [256], [0, 256])

plt.figure(figsize=(16, 7))

plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original img")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale img")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary img")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(hist_gray, color='blue')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(2, 3, 6)
plt.plot(hist_binary, color='black')
plt.title("Binary Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
