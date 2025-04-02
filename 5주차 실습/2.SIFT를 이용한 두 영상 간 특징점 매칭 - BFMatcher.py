import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 및 일부 영역 크롭 (모델 영상)
img1 = cv.imread('mot_color70.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 장면 영상 읽기
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create(500, 3, 0.09, 10 ,1.5)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('특징점 개수:', len(kp1), len(kp2))

# BFMatcher 기반 매칭 (crossCheck=True 사용)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
# 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)
print('매칭 수:', len(matches))

# 매칭 결과 이미지 생성
img_matches = cv.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# matplotlib로 시각화 (BGR -> RGB 변환)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_rgb)
plt.axis('off')
plt.title('BFMatcher 기반 매칭 (crossCheck=True)')
plt.show()
