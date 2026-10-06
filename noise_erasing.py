import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# 1. Image load karo (grayscale me)
img = cv2.imread('salt_pepper_noise.png', 0)

# Agar image nahi hai to upload karna padega
# Files > Upload me salt_pepper_noise.png dal do

# 2. 5x5 ka kernel banao
kernel = np.ones((5,5), np.uint8)
# ya: kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# 3. MORPH_OPEN apply karo
# OPEN = Erosion phir Dilation -> background ke white specks hatata hai
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 4. Save karo
cv2.imwrite('clean_image.png', opened)

# 5. Bada dikhao
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original - salt_pepper_noise.png")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Clean - MORPH_OPEN 5x5")
plt.imshow(opened, cmap='gray')
plt.axis('off')
plt.show()
