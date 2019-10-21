timer = True
if timer:
    import time
    t0 = time.time()

import math #do we need this?
from statistics import mean
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def diff_then_background(pix):
    global backgroundColor
    if abs(pix - backgroundColor) < 0.2:
        return False
    else:
        return True

from hulpfunctie_sobel import *
files = ["1_rechthoeken.png", "2_rechthoeken_dicht_op_elkaar.png", "2objecten_1lichtbron.PNG", "3_willekeurige_vormen.png", "4_rechthoeken_in_kleur.png", "5_rechthoeken_niet_opgevuld.png", "6_rechthoeken_deels_opgevuld.png", "7_rechthoeken_in_kleur_deels_opgevuld.png", "8_vormen_vaag_met_schaduw.png", "9_vormen_gekleurd.png", "10_vormen_opgedeeld.png", "11_zwart.png"]

activateCheckpoints = False         #Whether it will print the checkpoints (mainly for timing purposes)
if activateCheckpoints:
    print("Checkpoint 1")
if timer:
    t1 = time.time()
#TestVariables
for file in files:
    name_image = file    #Which photo
    print(name_image)
    gaussianAmount = 1                  #How many times Gaussian blur is done on image, must be natural number
    demonstration = False               #Whether it will show figure at the end

    #Open Image
    img = np.array(Image.open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\testImages\\" + name_image))#.astype(np.uint8)


    # Apply gray scale
    gray_img = grayscale(img)
    backgroundColor = gray_img[0,0]


    # Sobel Operator
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1
    #Bounding box
    t7 = time.time()
    for rowIndex in range(0, h - 1):
        if len(list(filter(diff_then_background, list(gray_img[rowIndex])))) != 0:
            upperRow = rowIndex
            break
    for rowIndex in range(0, h - 1):
        if len(list(filter(diff_then_background, list(gray_img[(h - 1) - rowIndex])))) != 0:
            lowerRow = (h - 1) - rowIndex
            break

    t8 = time.time()

    if activateCheckpoints:
        print("Checkpoint 2")

    if timer:
        t2 = time.time()
    # define images with 0s
    newHorizontalImage = np.zeros((h, w))
    newVerticalImage = np.zeros((h, w))
    newGradientImage = np.zeros((h, w))
    newHorizontalImage2 = np.zeros((h, w))
    newVerticalImage2 = np.zeros((h, w))
    newGradientImage2 = np.zeros((h, w))
    gray_img2 = 2*gray_img
    gray_img_neg = -1*gray_img
    gray_img_neg_2 = -1*gray_img2


    if activateCheckpoints:
        print("Checkpoint 3")
    # offset by 1
    if (upperRow-10) < 0:
        upperRow = -10
    if (lowerRow + 10) > (h - 1):
        lowerRow = h - 11
    t3 = time.time()
    newHorizontalImage2 = ndimage.convolve(gray_img, horizontal)
    newVerticalImage2 = ndimage.convolve(gray_img, vertical)
    t4 = time.time()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Edge Magnitude
            mag = np.sqrt(pow(newHorizontalImage2[i - 1, j - 1], 2.0) + pow(newVerticalImage2[i - 1, j - 1], 2.0))
            newGradientImage2[i - 1, j - 1] = mag
    t5 = time.time()
    newVerticalImage2 = np.square(newVerticalImage2)
    newHorizontalImage2 = np.square(newHorizontalImage2)
    newSum = newHorizontalImage2 + newVerticalImage2
    newSum = np.sqrt(newSum)
    t6 = time.time()
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
    for i in range(upperRow - 10, lowerRow + 10):
        for j in range(1, w - 1):
            overlapGrad     = (gray_img_neg[i - 1, j - 1] + gray_img[i + 1, j + 1])
            negOverlapGrad  = (gray_img[i - 1, j + 1] +  gray_img_neg[i + 1, j - 1])
            horizontalGrad  = overlapGrad + negOverlapGrad + gray_img_neg_2[i, j - 1] + gray_img2[i, j + 1]
            verticalGrad    = overlapGrad - negOverlapGrad + gray_img_neg_2[i - 1, j] + gray_img2[i + 1, j]
            # Edge Magnitude
            mag = np.sqrt(pow(overlapGrad, 2.0) + pow(negOverlapGrad, 2.0))
            newGradientImage2[i - 1, j - 1] = mag
    plt.imsave('C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\Sobel_foto.jpg', newGradientImage, cmap='gray', format='jpg')
    print(False in newGradientImage2 == newSum)
    if activateCheckpoints:
        print("Checkpoint 4")
    t7 = time.time()
    print("CLASSIC METHOD: ", t7 - t6)
    print("NEW METHOD:     ", t5 - t3, (t5-t4)/(t5-t3))
    print("NEW METHOD 2:   ", t4 - t3 + (t6 - t5))
    print("FOR LOOP:       ", t5 - t4)
    print("NUMPY:          ", t6 - t5)
    print("========================")
