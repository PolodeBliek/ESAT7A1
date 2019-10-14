from Kinect_to_pc import *
from Image_processing import *
from Image_detection import *

import os
import math
from statistics import mean
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

#1: van kinect-camera naar pc via hulpscript Kinect_to_pc.py
kinect_to_pc(1080, 1920,4)
################

name_image = 'KinectPicture.png'    #Which photo
gaussianAmount = 1                  #How many times Gaussian blur is done on image, must be natural number
currentDir      = os.path.dirname(os.path.abspath(__file__))
directory       = currentDir + "\\" + name_image
#Open Image
img = np.array(Image.open("/Users/Brecht/PycharmProjects/Peno3/input_images" + name_image))#.astype(np.uint8)

# Apply gray scale
gray_img = grayscale(img)

# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

#Apply Gaussian Blur
for index in range(gaussianAmount):
    gray_img = gaussian(gray_img)
    plt.imsave(directory + 'Gaussian_Blur.png', gray_img, cmap='gray', format='png')# define images with 0s
newHorizontalImage = np.zeros((h, w))
newVerticalImage = np.zeros((h, w))
newGradientImage = np.zeros((h, w))


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
plt.imsave("/Users/Brecht/PycharmProjects/Peno3/output_images/Sobel_foto.png", newGradientImage, cmap='gray', format='png')

#############
#toepassen van hysteresis
#MET DEZE WAARDEN VALT NOG TE EXPERIMENTEREN

iar = hyst(np.array(gem_kleur_van_pixels("/Users/Brecht/PycharmProjects/Peno3/output_images/Sobel_foto.png")),10,27)

iar_boolToNum = [0 if i == False else 255 for i in iar]

np_array_iar_reconverted = np.array(iar_reconverted(iar_boolToNum,h,w))
np_array_to_float = np_array_iar_reconverted.astype(np.uint8)
plt.imsave("/Users/Brecht/PycharmProjects/Peno3/output_images/Processed_image.png", newGradientImage, cmap='gray', format='png')

###############

# This is a program to detect and count object from a binary matrix,
#  matrix with 0 en 1, where 1 stands for pixel from an object and 0 for empty, the background.
matrix,nb_columns,nb_rows = make_matrix_voor_annelies(iar, h, w)
nb_object_collision = []

# TODO: find a better way to substract the colliding pixels from the same objects from the total count

def get_label(row, column):
    """
    :return: the value of the element located at the given row and given column.
    """
    return matrix[row][column]

def get_label_left(row, column):
    """
    :return: the value of the element left of the element located at the given row and the given column.
    """
    return matrix[row][column - 1]

def get_label_above(row, column):
    """
    :return: the value of the element above of the element located at the given row and the given column.
    """
    return matrix[row - 1][column]

def label_is_one(row, column):
    """
    :return: true if the element at the given row and given column has a value of 1.
    """
    return get_label(row, column) == 1

def is_labeled(row, column):
    """
    :return: true if the element at the given row and given column has a value different from 0.
            Meaning the element is a pixel as part of an object.
    """
    return get_label(row, column) != 0

def is_connected(row, column):
    """
    :return: true if the element at the given row and the given column is next to or under
                an element that is not zero
    """
    if row == 0 and column == 0:
        return False
    if column == 0:
        return is_labeled(row - 1, column)
    if row == 0:
        return is_labeled(row, column - 1)
    else:
        return is_labeled(row, column - 1) or is_labeled(row - 1, column)

def solve_collision_detection(lowest_label, highest_label):
    """
    :param lowest_label:
    :param highest_label:
    :return: a matrix where the elements with a value equal to the given highest_label are replaced by
            elements with a value equal to the given lowest_label
    """
    for i in range(nb_rows):
        for (e, element) in enumerate(matrix[i]):
            if element == highest_label:
                matrix[i][e] = lowest_label
                # print(matrix[i])

# TODO: Divide this function into two !!!
def get_lowest_adjacent_label(row, column):
    """
    :return: the lowest value of the adjacent labels.
    """
    label_left = get_label_left(row, column)
    label_above = get_label_above(row, column)
    if row == 0 or label_above == 0:
        return label_left
    if column == 0 or label_left == 0:
        return label_above
    if label_above != label_left:
        solve_collision_detection(min(label_left, label_above), max(label_left, label_above))
        nb_object_collision.append(1)
    return min(label_left, label_above)

def main():
    counter = 0
    for row in range(nb_rows):
        for column in range(nb_columns):
            if label_is_one(row, column):
                if is_connected(row, column):
                    matrix[row][column] = get_lowest_adjacent_label(row, column)
                else:
                    matrix[row][column] = counter + 1
                    counter += 1

    print("Number of objects: ", counter - len(nb_object_collision))
    plt.imshow(matrix)
    plt.show()


main()
