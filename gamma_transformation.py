# Importing packages #
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks

# Reading the image #
img = cv.imread('gradiente_cinza.jpg',cv.IMREAD_GRAYSCALE)


# Getting the size of the image #
height = img.shape[0] #M
width = img.shape[1] #N

# Applying the gamma transformation #
gamma_img = np.zeros((height,width)).astype(np.float32)

c = 1
gamma = 2

for i in range(height):
  for j in range(width):
   gamma_img[i][j] = c*( (img[i][j])**gamma )


# Converting to 0 to 255 grayscale
img_max = gamma_img.max()

for i in range(height):
  for j in range(width):
    gamma_img[i][j] = round((255*gamma_img[i][j])/img_max)

gamma_img = gamma_img.astype(np.uint8)

# Plotting the original image and the image with the transformation #

#Google Cola
#cv2_imshow(img)
#cv2_imshow(gamma_img)

# OpenCV
cv.imshow('Original', img)
cv.imshow('Gamma', gamma_img)


# Ploting the images in a different color map #
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gamma_img)
plt.axis('off')
plt.show()

cv.waitKey(0)