# Importing packages
import cv2 as cv
import numpy as np
import math
from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks

# Reading the image

img = cv.imread('/content/drive/MyDrive/digital_image_processing_fundamentals/Avaliacao_2/lena256x256.tif',cv.IMREAD_GRAYSCALE)
# Google Colab
cv2_imshow(img)

# OpenCV
#cv.show('Lena original',img)

# Getting the size of the image
height = img.shape[0] #M
width = img.shape[1] #N


# Adding the periodic noise to the original image
noise_values = np.ones((256,256))

fre = 100/height

for i in range(height):
  for j in range(width):
    noise_values[j][i] = 400*(math.sin( (2*math.pi*fre)*i))


for i in range(height):
  for j in range(width):
    img[i][j] = float(img[i][j])+noise_values[i][j]

# Plotting image

# showing image with Google Colab or Jupyter Notebook
cv2_imshow(img)


# OpenCV
#cv.show('Lena negativo',negative_img)