# Importing packages
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
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
intensities = 256

img = img.astype(float)

# Adding the periodic noise to the original image
noise_values, noisy_img = np.zeros((height,width)), np.zeros((height,width))

fre = 100/height

for i in range(height):
  for j in range(width):
    noise_values[j][i] = 400*(math.sin( (2*math.pi*fre)*i))

noisy_img = np.add(img, noise_values)

# Plotting image

# showing image with Google Colab or Jupyter Notebook
cv2_imshow(noisy_img)

# OpenCV
#cv.show('Lena com ruído',negative_img)