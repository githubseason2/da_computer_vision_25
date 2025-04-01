import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기 (BGR)
img = cv.imread('c.png')
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# OpenCV 방식 (가중치: 0.114 B, 0.587 G, 0.299 R)
gray_cv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 사용자 방식 (균등 가중치: 1/3씩)
# BGR에서 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# R, G, B 채널 분리
print(img.shape)
R = img_rgb[:, :, 0].astype(np.float32)
G = img_rgb[:, :, 1].astype(np.float32)
B = img_rgb[:, :, 2].astype(np.float32)

# 평균 계산 후 uint8로 변환
gray_uniform = ((R + G + B) / 3).astype(np.uint8)

print(img[150][150],gray_cv[150][150], gray_uniform[150][150])

# 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('OpenCV Grayscale')
plt.imshow(gray_cv, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Uniform Grayscale (0.3333)')
plt.imshow(gray_uniform, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
