# Importing packages
import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks

# Reading the image

img = cv.imread('/content/drive/MyDrive/digital_image_processing_fundamentals/Avaliacao_2/lena256x256.tif', cv.IMREAD_GRAYSCALE)

# Showing original image
cv2_imshow(img)
#OPENCV
#cv.imshow|('Lena original', img)


# Getting the size of the image
height = img.shape[0] #M
width = img.shape[1] #N

# Applying the gamma transformation
c = 1
gamma = 0.5

for i in range(height):
  for j in range(width):
   img[i][j] = c*( (img[i][j])**gamma )

img = img.astype(np.uint8)

# Plotting image
# showing image with Google Colab or Jupyter Notebook
cv2_imshow(img)

# OpenCV
#cv.show('Lena - Gamma transformation',img)