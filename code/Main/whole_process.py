import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import numpy as np
from statistics import mean
from PIL import Image
import os
import matplotlib.pyplot as plt


###### HULPFUNCTIES #######
def gem_kleur_van_pixels(picture):
    image = np.array(picture).astype(np.uint8)

    gem_kleur_van_pixel = []
    for eachRow in image:
        for eachPix in eachRow:
            avgColor = mean(eachPix[:3])  # eerste 3 getallen vd array die de kleur geven
            gem_kleur_van_pixel.append(avgColor)

    return gem_kleur_van_pixel


def flatten_matrix(matrix):
    image = np.array(matrix).astype(np.uint8)

    array = []
    for rows in image:
        for pixel in rows:
            array.append(pixel)

    return array


def reconvert_to_img(hyst_array, height, width, name_image):
    """
    :return: a saveable reconstructed image of the array that comes out of hysteresis
    """
    arary_Bool2Num = [0 if i == False else 255 for i in hyst_array]

    # reconstruct the (h,w,3)-matrix
    reconverted_array = np.zeros((height, width))
    index = 0
    for i in range(height):
        for j in range(width):
            reconverted_array[i][j] = arary_Bool2Num[index]
            index += 1

    reconverted_image = reconverted_array.astype(np.uint8)  # the values need to be uint8 types

    # save and show the image
    # plt.figure()
    # plt.title('Processed image')
    # plt.imsave('../processed_images/' + name_image[:-4] + '_processed.png', reconverted_image, cmap='gray', format='png')
    # plt.imshow(reconverted_image, cmap='gray')
    #
    # plt.show()

    return reconverted_array


def make_detection_matrix(hyst_array, h, w):
    """
    :return: a matrix consisting of 0's and 1's for the object detection
    """
    array_Bool2Bin = [0 if i == False else 1 for i in hyst_array]

    matrix = []
    index = 0
    for i in range(h):
        matrix.append([])
        for j in range(w):
            k = array_Bool2Bin[index]
            matrix[i].append(k)
            index += 1

    return matrix


####### TAKING PICTURES #######
def kinect_to_pc(width, height, dimension):
    # https://github.com/daan/calibrating-with-python-opencv/blob/02c90e4291adfb2426072f8f0837033754fc3a55/kinect-v2/color.py
    print("connecting with kinect...")
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    print("connected to kinect")
    noPicture = True

    color_flipped = None  # give them a default value
    colorized_frame = None

    while noPicture:
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            noPicture = False

            color_frame = color_frame.reshape((width,height,dimension))
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            color_flipped = cv2.flip(color_frame, 1)

            cv2.imwrite("../input_images/KinectPicture.png", color_flipped)  # Save

    depth_image_size = (424, 512)

    kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    noDepth = True

    while noDepth:
        if kinect2.has_new_depth_frame():
            depth_frame = kinect2.get_last_depth_frame()
            noDepth = False

            depth_frame = depth_frame.reshape(depth_image_size)
            depth_frame = depth_frame * (256.0 / np.amax(depth_frame))
            colorized_frame = cv2.applyColorMap(np.uint8(depth_frame), cv2.COLORMAP_JET)
        #cv2.imshow('depth', colorized_frame)
            cv2.imwrite("../input_images/kinectDepthPicture.png", colorized_frame)  # Save

    print("pictures taken")
    return color_flipped, colorized_frame


####### IMAGE PROCESSING #######
def grayscale(image):
    return (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)


def gaussian_blur(image):

    h, w = image.shape
    GaussianKernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    newImg = np.zeros((h, w))
    newImg = ndimage.convolve(newImg, GaussianKernel)
    return newImg


def sobel(image):
    # get dimensions
    h, w = image.shape

    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # initialize new images
    newHorizontalImage = np.zeros((h, w))
    newVerticalImage = np.zeros((h, w))
    newGradientImage = np.zeros((h, w))
    newVerticalImage2 = np.square(newVerticalImage2)
    newHorizontalImage2 = np.square(newHorizontalImage2)
    newSum = newHorizontalImage2 + newVerticalImage2
    newSum = np.sqrt(newSum)
    return newSum


def hysteresis(sobel_image, th_lo, th_hi, initial=False):
    """
    x : Numpy Array
        Series to apply hysteresis to.
    th_lo : float or int
        Below this threshold the value of hyst will be False (0).
    th_hi : float or int
        Above this threshold the value of hyst will be True (1).
    """

    # convert the image to an array
    # x = np.array(gem_kleur_van_pixels(sobel_image))  # enkel als ge het met een opgeslagen afbeelding moet doen
    x = np.array(flatten_matrix(sobel_image))  # sobel returns a 2D matrix now instead of an image

    if th_lo > th_hi: # If thresholds are reversed, x must be reversed as well
        x = x[::-1]
        th_lo, th_hi = th_hi, th_lo
        rev = True
    else:
        rev = False

    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi

    ind = np.nonzero(lo_or_hi)[0]  # Index für alle darunter oder darüber
    if not ind.size:  # prevent index error if ind is empty
        x_hyst = np.zeros_like(x, dtype=bool) | initial
    else:
        cnt = np.cumsum(lo_or_hi)  # from 0 to len(x)
        x_hyst = np.where(cnt, hi[ind[cnt-1]], initial)

    if rev:
        x_hyst = x_hyst[::-1]

    return x_hyst


# start the processing/filtering loop
def process_image(image):
    print("start image processing")
    # if the image doesn't come straight from the Kinect, but is a selected picture, open the selected picture
    if isinstance(image, str):
        name_image = image  # 'image' is a string
        image = np.array(Image.open('../input_images/' + image))  # .astype(np.uint8)
    else:
        name_image = "KinectPicture"

    # constants
    gauss_repetitions = 1  # number of times the gaussian filter is applied
    low_threshold, high_threshold = 10, 27  # the thresholds for hysteresis
    h, w, d = image.shape  # the height, width and depth of the image
    print(h,w)

    # image processing/filtering process
    gray_image = grayscale(image)
    for i in range(gauss_repetitions):
        blurred_image = gaussian_blur(gray_image)
    sobel_image = sobel(blurred_image)
    hyst_image = hysteresis(sobel_image, low_threshold, high_threshold)
    reconvert_to_img(hyst_image, h, w, name_image)  # if you want to save the image (as e.g. a .png)

    return make_detection_matrix(hyst_image, h, w)


####### DETECTING OBJECTS #######
# TODO: find a better way to substract the colliding pixels from the same objects from the total count

def get_label(matrix, row, column):
    """
    :return: the value of the element located at the given row and given column.
    """
    return matrix[row][column]


def get_label_left(matrix, row, column):
    """
    :return: the value of the element left of the element located at the given row and the given column.
    """
    return matrix[row][column - 1]


def get_label_above(matrix, row, column):
    """
    :return: the value of the element above of the element located at the given row and the given column.
    """
    return matrix[row - 1][column]


def label_is_one(matrix, row, column):
    """
    :return: true if the element at the given row and given column has a value of 1.
    """
    return get_label(matrix, row, column) == 1


def is_labeled(matrix, row, column):
    """
    :return: true if the element at the given row and given column has a value different from 0.
            Meaning the element is a pixel as part of an object.
    """
    return get_label(matrix, row, column) != 0


def is_connected(matrix, row, column):
    """
    :return: true if the element at the given row and the given column is next to or under
                an element that is not zero
    """
    if row == 0 and column == 0:
        return False
    if column == 0:
        return is_labeled(matrix, row - 1, column)
    if row == 0:
        return is_labeled(matrix, row, column - 1)
    else:
        return (is_labeled(matrix, row, column-1)) or (is_labeled(matrix, row-1, column))


def solve_collision_detection(matrix, lowest_label, highest_label):
    """
    :param lowest_label:
    :param highest_label:
    :return: a matrix where the elements with a value equal to the given highest_label are replaced by
            elements with a value equal to the given lowest_label
    """
    nb_rows = len(matrix)
    for i in range(nb_rows):
        for (e, element) in enumerate(matrix[i]):
            if element == highest_label:
                matrix[i][e] = lowest_label
                # print(matrix[i])


# TODO: Divide this function into two !!!
def get_lowest_adjacent_label(matrix, row, column, nb_object_collision):
    """
    :return: the lowest value of the adjacent labels.
    """
    label_left = get_label_left(matrix, row, column)
    label_above = get_label_above(matrix, row, column)
    if row == 0 or label_above == 0:
        return label_left
    if column == 0 or label_left == 0:
        return label_above
    if label_above != label_left:
        solve_collision_detection(matrix, min(label_left, label_above), max(label_left, label_above))
        nb_object_collision.append(1)
    return min(label_left, label_above)


# start the object detection loop
def detect_objects(matrix):
    # print("object detecting started")
    nb_rows, nb_columns = len(matrix), len(matrix[0])
    nb_object_collision = []
    print("nb_rows, nb_columns =", nb_rows, nb_columns)

    counter = 0
    for row in range(nb_rows):
        for column in range(nb_columns):
            if label_is_one(matrix, row, column):
                # print(row,column)
                if is_connected(matrix, row, column):
                    matrix[row][column] = get_lowest_adjacent_label(matrix, row, column, nb_object_collision)
                else:
                    matrix[row][column] = counter + 1
                    counter += 1

    print("Number of objects: ", counter - len(nb_object_collision))
    plt.imshow(matrix)
    plt.show()


####### MAIN #######
def main():
    # 1) take a picture
    color_image, depth_image = kinect_to_pc(1080, 1920, 4)

    # 2) start the image processing
    matrix = process_image(color_image)

    # 3) start looking for objects
    detect_objects(matrix)


if __name__ == '__main__':

    main()
