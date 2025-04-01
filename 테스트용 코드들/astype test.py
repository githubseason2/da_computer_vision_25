import numpy as np

# RGB 값 (예시)
R, G, B = 186, 215, 230

# 1. Uniform grayscale
uniform_gray = int((R + G + B) / 3)  # → 210

# 2. OpenCV 방식 (BGR 순서)
BGR = [230, 215, 186]
cv_gray = int(0.114 * BGR[0] + 0.587 * BGR[1] + 0.299 * BGR[2])  # → 약 208

print(f"Uniform grayscale: {uniform_gray}")
print(f"OpenCV grayscale: {cv_gray}")
