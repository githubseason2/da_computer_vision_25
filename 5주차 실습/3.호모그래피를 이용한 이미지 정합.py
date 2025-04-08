import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')


gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print("특징점 개수:", len(kp1), len(kp2))


bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
knn_matches = bf.knnMatch(des1, des2, k=2)
T = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < T * n.distance:
        good_matches.append(m)
print("좋은 매칭 수:", len(good_matches))

points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])


H, mask = cv.findHomography(points2, points1, cv.RANSAC)


h1, w1 = img1.shape[:2]


aligned_img2 = cv.warpPerspective(img2, H, (w1, h1))



gray_aligned = cv.cvtColor(aligned_img2, cv.COLOR_BGR2GRAY)
_, mask_aligned = cv.threshold(gray_aligned, 1, 255, cv.THRESH_BINARY)
mask_aligned_3c = cv.cvtColor(mask_aligned, cv.COLOR_GRAY2BGR)
composite_overlay = np.where(mask_aligned_3c == 255, aligned_img2, img1)

orig1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
orig2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
aligned_rgb = cv.cvtColor(aligned_img2, cv.COLOR_BGR2RGB)
overlay_rgb = cv.cvtColor(composite_overlay, cv.COLOR_BGR2RGB)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].imshow(orig1_rgb)
axs[0, 0].set_title("Original img1")
axs[0, 0].axis("off")
axs[0, 1].imshow(orig2_rgb)
axs[0, 1].set_title("Original img2")
axs[0, 1].axis("off")

axs[1, 0].imshow(aligned_rgb)
axs[1, 0].set_title("Aligned img2")
axs[1, 0].axis("off")
axs[1, 1].imshow(overlay_rgb)
axs[1, 1].set_title("Composite Overlay")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
