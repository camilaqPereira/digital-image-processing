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


# Applying the fourier transform to the original and noisy images #

#Original image
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
Fabs = np.abs(Fshift)

#Noisy image
Fn = np.fft.fft2(noisy_img)
FNshift = np.fft.fftshift(Fn)
FNabs = np.abs(FNshift)


# plotting the magnitudes of the original and noisy transforms #
# on a logarithmic scale #

#Original image
c = 15/math.log(Fabs.max(),10)
D = [[c*math.log(1+num) for num in line] for line in Fabs]
z = np.abs(D)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(z, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")
plt.show()


#Noisy image
cn = 15/math.log(FNabs.max(),10)
Dn = [[cn*math.log(1+num) for num in line] for line in FNabs]
zn = np.abs(Dn)

plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.imshow(zn, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")
plt.show()



# Finding the values causing the noise #
FNmax = FNabs.max()

positions = np.where(FNabs==FNmax)
positions = zip(positions[0],positions[1])

for position in positions:
  FNshift[position[0]][position[1]] = 0


# Plotting the new magnitude
FNabs = abs(FNshift)

cn2 = 15/math.log(FNabs.max(),10)
Dn2 = [[cn2*math.log(1+num) for num in line] for line in FNabs]
zn2 = np.abs(Dn2)

plt.figure(figsize=(5,5))
plt.imshow(zn2, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")
plt.show()


# Showing final image #
img_recons = abs(np.fft.ifft2(FNshift))
cv2_imshow(img_recons)

# OpenCV
#cv.show('Lena sem ruído',img_recons)

