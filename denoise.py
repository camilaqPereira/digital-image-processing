# Importing packages #
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow #for Google Colab or Jupyter Notebooks

# Reading the image #

img = cv.imread('lena256x256.tif',cv.IMREAD_GRAYSCALE)

# Google Colab
#cv2_imshow(img)

# OpenCV
cv.imshow('Lena original',img)

# Getting the size of the image
height = img.shape[0] #M
width = img.shape[1] #N

# Adding the periodic noise to the original image
img = img.astype(np.float32)

noise_values, noisy_img = np.zeros((height,width),dtype=np.float32), np.zeros((height,width),dtype=np.float32)

fre = 100/height

for i in range(height):
  for j in range(width):
    noise_values[j][i] = 400*(math.sin( (2*math.pi*fre)*i))

noisy_img = np.add(img, noise_values)

# Plotting image with noise #

# OpenCV
plt.imshow(noisy_img,cmap=plt.cm.gray, interpolation='none')
plt.axis('off')
plt.show()


# Applying the fourier transform to the original and noisy images #

#Original image
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
Fabs = np.abs(Fshift)

#Noisy image
Fn = np.fft.fft2(noisy_img)
FNshift = np.fft.fftshift(Fn)
FNabs = np.abs(FNshift)


# Plotting the magnitudes of the original and noisy transforms #
# on a logarithmic scale #

#Original image
c = 15/math.log(Fabs.max(),10)
D = [[c*math.log(1+num) for num in line] for line in Fabs]
z = np.abs(D)

#Noisy image
cn = 15/math.log(FNabs.max(),10)
Dn = [[cn*math.log(1+num) for num in line] for line in FNabs]
zn = np.abs(Dn)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(Fabs, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(FNabs, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(z, cmap=plt.cm.gray, interpolation="none")
plt.axis("off")

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


# Plotting the new magnitude #
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
#cv2_imshow(img_recons)

# OpenCV
#cv.imshow('Lena sem ruido',img_recons.astype(np.uint8))
plt.imshow(img_recons,cmap=plt.cm.gray, interpolation="none")
plt.axis('off')
plt.show()

cv.waitKey(0)

