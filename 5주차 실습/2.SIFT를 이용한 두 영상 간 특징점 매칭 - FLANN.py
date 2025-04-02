import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img1 = cv.imread('mot_color70.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 장면 영상
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create(1000, 3, 0.09, 10 ,1.5)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수:', len(kp1), len(kp2))

# FLANN 기반 매칭 (FlannBasedMatcher() 사용)
FLANN_INDEX_KDTREE = 1  # SIFT는 KD-트리 사용
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(des1, des2, k=2)

# 최근접 이웃 거리 비율 테스트
T = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < T * n.distance:
        good_matches.append(m)

print('좋은 매칭 수:', len(good_matches))

# 매칭 결과 이미지 생성
img_matches = cv.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# matplotlib으로 시각화 (BGR -> RGB 변환)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_rgb)
plt.axis('off')
plt.title('FLANN based match')
plt.show()
