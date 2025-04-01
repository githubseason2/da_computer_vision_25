import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('edgeDetectionImage.jpg')
if(img is None):
    print("파일 확인")
    exit()
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x=cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
grad_y=cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)

edge_strength=cv.magnitude(grad_x, grad_y)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)
edge_strength_visual=cv.convertScaleAbs(edge_strength)

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.title('Original (Gray)')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel X')
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Sobel Y')
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')


plt.subplot(2, 2, 4)
plt.title('Edge Strength visual')
plt.imshow(edge_strength_visual, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
