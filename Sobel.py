import scipy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches
import random
import copy
import itertools


import math
from statistics import mean


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from collections.abc import Sequence
from itertools import chain, count

from hulpfunctie_sobel import *

# Open the image
name_image = '1_rechthoeken.png'
img = np.array(Image.open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\" + name_image))#.astype(np.uint8)


# Apply gray scale
gray_img = grayscale(img)

print("Checkpoint 1")

# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1
print(img[0,0])

print("Checkpoint 2")

# define images with 0s
newhorizontalImage = np.zeros((h, w))
newverticalImage = np.zeros((h, w))
newgradientImage = np.zeros((h, w))

print("Checkpoint 3")

# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        newverticalImage[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag
plt.imsave('C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\Sobel_foto.jpg', newgradientImage, cmap='gray', format='jpg')

print("Checkpoint 4")

#############
#toepassen van hysteresis
#MET DEZE WAARDEN VALT NOG TE EXPERIMENTEREN

iar = hyst(np.array(gem_kleur_van_pixels('Sobel_foto.jpg')),15,20)

iar_boolToNum = [0 if i == False else 255 for i in iar]

np_array_iar_reconverted = np.array(iar_reconverted(iar_boolToNum,h,w))
np_array_to_float = np_array_iar_reconverted.astype(np.uint8)

print("Checkpoint 5")
###############
plt.figure()
plt.title('Sobel_met_Hyst.jpg')
plt.imsave('C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\' + 'Sobel_met_Hystfoto.jpg', np_array_to_float, cmap='gray', format='jpg')
plt.imshow(np_array_to_float, cmap='gray')
plt.show()
