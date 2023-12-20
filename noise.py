# Importing packages
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks

# Reading the image #

img = cv.imread('lena256x256.tif',cv.IMREAD_GRAYSCALE)

# Showing the image #

# Google Colab
#cv2_imshow(img)

# OpenCV
cv.imshow('Lena original',img)

# Getting the size of the image #
height = img.shape[0] #M
width = img.shape[1] #N


# Adding the sinusoidal noise to the original image #

img = img.astype(np.float32)

noise_values, noisy_img = np.zeros((height,width), dtype=np.float32), np.zeros((height,width),dtype=np.float32)

fre = 100/height

for i in range(height):
  for j in range(width):
    noise_values[j][i] = 400*(math.sin( (2*math.pi*fre)*i))

noisy_img = np.add(img, noise_values)


# Plotting image #
plt.imshow(noisy_img,cmap=plt.cm.gray, interpolation='none')
plt.axis('off')
plt.show()

cv.waitKey(0)