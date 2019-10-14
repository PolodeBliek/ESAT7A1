timer = True
if timer:
    import time
    t0 = time.time()

import math #do we need this?
from statistics import mean
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from hulpfunctie_sobel import *


activateCheckpoints = True         #Whether it will print the checkpoints (mainly for timing purposes)
if activateCheckpoints:
    print("Checkpoint 1")
if timer:
    t1 = time.time()
#TestVariables
name_image      = '1_rechthoeken.png'    #Which photo
gaussianAmount  = 1                  #How many times Gaussian blur is done on image, must be natural number
demonstration   = False               #Whether it will show figure at the end
currentDir      = os.path.dirname(os.path.abspath(__file__))
directory       = currentDir + "\\testImages\\" + name_image

#Open Image
img = np.array(Image.open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\" + name_image))#.astype(np.uint8)
sys.exit()


# Apply gray scale
gray_img = grayscale(img)


# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

if activateCheckpoints:
    print("Checkpoint 2")

if timer:
    t2 = time.time()

#Apply Gaussian Blur
for index in range(gaussianAmount):
    gray_img = gaussian(gray_img)
    plt.imsave(directory + '\\outputImages\\Blur.jpg', gray_img, cmap='gray', format='jpg')
# define images with 0s
newHorizontalImage = np.zeros((h, w))
newVerticalImage = np.zeros((h, w))
newGradientImage = np.zeros((h, w))

if activateCheckpoints:
    print("Checkpoint 3")
if timer:
    t3 = time.time()

if timer:
    t5 = time.time()
    print("0 -> 1", t1-t0)
    print("1 -> 2", t2-t1)
    print("2 -> 3", t3-t2)
    print("Total:", t3-t0)
sys.exit()


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

        newHorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        newVerticalImage[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newGradientImage[i - 1, j - 1] = mag
plt.imsave(directory + "\\outputImages\\Sobel_foto.jpg", newGradientImage, cmap='gray', format='jpg')

if activateCheckpoints:
    print("Checkpoint 4")
if timer:
    t4 = time.time()

#############
#toepassen van hysteresis
#MET DEZE WAARDEN VALT NOG TE EXPERIMENTEREN

iar = hyst(np.array(gem_kleur_van_pixels('Sobel_foto.jpg')),15,20)

iar_boolToNum = [0 if i == False else 255 for i in iar]

np_array_iar_reconverted = np.array(iar_reconverted(iar_boolToNum,h,w))
np_array_to_float = np_array_iar_reconverted.astype(np.uint8)

if activateCheckpoints:
    print("Checkpoint 5")

if timer:
    t5 = time.time()
    print("0 -> 1", t1-t0)
    print("1 -> 2", t2-t1)
    print("2 -> 3", t3-t2)
    print("3 -> 4", t4-t3)
    print("4 -> 5", t5-t4)
    print("Total:", t5-t0)
###############
if demonstration:
    plt.figure()
    plt.title('Sobel_met_Hyst.jpg')
    plt.imsave(directory + '\\outputImages\\' + 'Sobel_met_Hystfoto.jpg', np_array_to_float, cmap='gray', format='jpg')
    plt.imshow(np_array_to_float, cmap='gray')
    plt.show()
