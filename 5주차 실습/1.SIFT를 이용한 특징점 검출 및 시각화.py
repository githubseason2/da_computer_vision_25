#sift 동그라미 크기 설명

import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('mot_color70.jpg')  # 영상 읽기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(1000, 3, 0.09, 10 ,1.5)
kp, des = sift.detectAndCompute(gray, None)

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_with_kp_rgb = cv.cvtColor(gray, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('SIFT Keypoints')
plt.imshow(img_with_kp_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
