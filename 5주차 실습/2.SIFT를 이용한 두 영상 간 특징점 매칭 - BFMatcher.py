import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(500, 3, 0.09, 10 ,1.5)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('특징점 개수:', len(kp1), len(kp2))

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print('매칭 수:', len(matches))

img_matches = cv.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_rgb)
plt.axis('off')
plt.title('BFMatcher 기반 매칭 (crossCheck=True)')
plt.show()
