import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 이미지 읽기 ---
# 원본 이미지들은 720×480 크기라고 가정
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

# --- 2. SIFT 특징점 검출 ---
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print("특징점 개수:", len(kp1), len(kp2))

# --- 3. BFMatcher와 knnMatch, 비율 테스트 ---
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

# --- 4. 호모그래피 계산 (img2 → img1 좌표계) ---
H, mask = cv.findHomography(points2, points1, cv.RANSAC)

# --- 5. 간단한 캔버스 설정 및 img2 정합 ---
h1, w1 = img1.shape[:2]
canvas_width = 2 * w1   # 가로 2배
canvas_height = 2 * h1  # 세로 2배

# 이동 없이 H만 적용
aligned_img2 = cv.warpPerspective(img2, H, (canvas_width, canvas_height))

# --- 6. img1을 동일 캔버스에 배치 (좌측 상단에 그대로 위치) ---
canvas_img1 = np.zeros((canvas_height, canvas_width, 3), dtype=img1.dtype)
canvas_img1[0:h1, 0:w1] = img1

# --- 7. 오버레이 합성 ---
# aligned_img2의 유효한 영역(픽셀 값이 0이 아닌 부분)은 그대로, 나머지는 canvas_img1의 원본 값을 사용
gray_aligned = cv.cvtColor(aligned_img2, cv.COLOR_BGR2GRAY)
_, mask_aligned = cv.threshold(gray_aligned, 1, 255, cv.THRESH_BINARY)
mask_aligned_3c = cv.cvtColor(mask_aligned, cv.COLOR_GRAY2BGR)
composite_overlay = np.where(mask_aligned_3c==255, aligned_img2, canvas_img1)

# --- 8. 결과 이미지들을 matplotlib으로 출력 ---
# 상단: 원본 이미지들 (img1, img2) → 그대로 출력 (720×480)
# 하단: 정합된 img2와 오버레이 합성 결과 (캔버스 크기: 2*w1 x 2*h1)
orig1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
orig2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
aligned_rgb = cv.cvtColor(aligned_img2, cv.COLOR_BGR2RGB)
overlay_rgb = cv.cvtColor(composite_overlay, cv.COLOR_BGR2RGB)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# 상단 행: 원본 이미지
axs[0, 0].imshow(orig1_rgb)
axs[0, 0].set_title("Original img1")
axs[0, 0].axis("off")
axs[0, 1].imshow(orig2_rgb)
axs[0, 1].set_title("Original img2")
axs[0, 1].axis("off")

# 하단 행: 정합된 이미지와 오버레이 합성 결과
axs[1, 0].imshow(aligned_rgb)
axs[1, 0].set_title("Aligned img2")
axs[1, 0].axis("off")
axs[1, 1].imshow(overlay_rgb)
axs[1, 1].set_title("Composite Overlay")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
