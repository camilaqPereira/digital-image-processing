# Importing packages
import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks


# Reading the image
img = cv.imread('/content/drive/MyDrive/digital_image_processing_fundamentals/Avaliacao_2/lena256x256.tif', cv.IMREAD_GRAYSCALE)

# Getting the size of the image
height = img.shape[0] #M
width = img.shape[1] #N
max_pixel_intensity = 256

# Applying the negative transformation
negative_img = np.ones((height,width))

for i in range(height):
  for j in range(width):
    negative_img[i][j] = max_pixel_intensity - 1 - img[i][j]

negative_img = negative_img.astype(np.uint8)

# Showing image with Google Colab or Jupyter Notebook
cv2_imshow(img)
cv2_imshow(negative_img)

# OpenCV
#cv.show('Lena original',img)
#cv.show('Lena negativo',negative_img)