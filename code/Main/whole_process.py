#import cv2
#from pykinect2 import PyKinectV2
#from pykinect2 import PyKinectRuntime

import numpy as np
from statistics import mean
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import time

#Test Variables
timed               = False
kinectFreeTesting   = True
printing            = False
show                = False
gauss_repetitions   = 1
grayscale_save      = False
gauss_save          = False
sobel_save          = False
hysteresis_save     = False
detect              = True
currentDir          = os.path.dirname(os.path.abspath(__file__)).replace("code\\Main", "")


if timed:
    time_gem_kleur  = 0
    time_flatten    = 0
    time_reconvert  = 0
    time_detection  = 0
    time_grayscale  = 0
    time_sobel      = 0
    time_Gaussian   = 0
    time_Hysteris   = 0
    time_processing = 0



###### HULPFUNCTIES #######
def gem_kleur_van_pixels(picture):
    if timed:
        t0 = time.time()
    image = np.array(picture).astype(np.uint8)

    gem_kleur_van_pixel = []
    for eachRow in image:
        for eachPix in eachRow:
            avgColor = mean(eachPix[:3])  # eerste 3 getallen vd array die de kleur geven
            gem_kleur_van_pixel.append(avgColor)
    if timed:
        t1 = time.time()
        global time_gem_kleur
        time_gem_kleur = t1 - t0
        print("IMPORTANT:", t1-t0)
    return gem_kleur_van_pixel


def flatten_matrix(matrix):
    if timed:
        t0 = time.time()
    image = np.array(matrix).astype(np.uint8)
    if timed:
        t1 = time.time()
        global time_flatten
        time_flatten = t1 - t0
    return image.flatten()


def reconvert_to_img(hyst_array, height, width, name_image):
    if timed:
        t0 = time.time()
    """
    :return: a saveable reconstructed image of the array that comes out of hysteresis
    """
    arary_Bool2Num = 255*hyst_array.astype(np.uint8)
    # reconstruct the (h,w,3)-matrix
    reconverted_array = np.reshape(arary_Bool2Num, (height, width))
    reconverted_image = reconverted_array.astype(np.uint8)  # the values need to be uint8 types

    # save and show the image
    # plt.figure()
    # plt.title('Processed image')
    # plt.imsave('../processed_images/' + name_image[:-4] + '_processed.png', reconverted_image, cmap='gray', format='png')
    # plt.imshow(reconverted_image, cmap='gray')
    #
    # plt.show()
    if timed:
        t1 = time.time()
        global time_reconvert
        time_reconvert = t1 - t0
    return reconverted_array


def make_detection_matrix(hyst_array, h, w):
    if timed:
        t0 = time.time()
    """
    :return: a matrix consisting of 0's and 1's for the object detection
    """
    array_Bool2Bin = hyst_array.astype(int)
    matrix = np.reshape(array_Bool2Bin, (h, w))
    if timed:
        t1 = time.time()
        global time_detection
        time_detection = t1 - t0
    return matrix


####### TAKING PICTURES #######
def kinect_to_pc(width, height, dimension):
    if timed:
        t0 = time.time()
    # https://github.com/daan/calibrating-with-python-opencv/blob/02c90e4291adfb2426072f8f0837033754fc3a55/kinect-v2/color.py
    if printing:
        print("connecting with kinect...")
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    if printing:
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
    if printing:
        print("pictures taken")
    if timed:
        t1 = time.time()
        print("kinect_to_pc", t1-t0)
    return color_flipped, colorized_frame


####### IMAGE PROCESSING #######
def grayscale(image):
    if timed:
        t0 = time.time()
    Image = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)
    if timed:
        t1 = time.time()
        global time_grayscale
        time_grayscale = t1 - t0
    return Image


def gaussian_blur(image):
    if timed:
        t0 = time.time()
    h, w = image.shape
    GaussianKernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    newImg = ndimage.convolve(image, GaussianKernel)
    if timed:
        t1 = time.time()
        global time_Gaussian
        time_Gaussian = t1 - t0
    return newImg


def sobel(image):
    if timed:
        t0 = time.time()
    # get dimensions
    h, w = image.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # initialize new images
    newHorizontalImage = ndimage.convolve(image, horizontal)
    newVerticalImage = ndimage.convolve(image, vertical)
    newVerticalImage = np.square(newVerticalImage)
    newHorizontalImage = np.square(newHorizontalImage)
    newSum = newHorizontalImage + newVerticalImage
    newSum = np.sqrt(newSum)
    if timed:
        t1 = time.time()
        global time_sobel
        time_sobel = t1 - t0
    return newSum


def hysteresis(sobel_image, th_lo, th_hi, initial = False):
    if timed:
        t0 = time.time()
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
    if timed:
        t1 = time.time()
        global time_Hysteris
        time_Hysteris = t1 - t0
    return x_hyst


# start the processing/filtering loop
def process_image(image):
    if timed:
        t0 = time.time()
    if printing:
        print("start image processing")
    # if the image doesn't come straight from the Kinect, but is a selected picture, open the selected picture
    if isinstance(image, str):
        name_image = image  # 'image' is a string
        image_path = currentDir + "\\testImages\\" + image
        image = np.array(Image.open(image_path))  # .astype(np.uint8)
    else:
        name_image = "KinectPicture"
    global gauss_repetitions, gauss_save, grayscale_save, sobel_save, hysteresis_save
    # constants
    low_threshold, high_threshold = 10, 27  # the thresholds for hysteresis
    h, w, d = image.shape  # the height, width and depth of the image
    if printing:
        print(h,w)

    # image processing/filtering process
    gray_image = grayscale(image)                                               #153
    if grayscale_save:
        plt.imsave(currentDir + 'Gray.jpg', gray_image, cmap='gray', format='jpg')
    for i in range(gauss_repetitions):
        blurred_image = gaussian_blur(gray_image)                               #163
    if gauss_save:
        plt.imsave(currentDir + 'Gauss.jpg', blurred_image, cmap='gray', format='jpg')
    sobel_image = sobel(blurred_image)                                          #176
    if sobel_save:
        plt.imsave(currentDir + 'Sobel.jpg', sobel_image, cmap='gray', format='jpg')
    hyst_image = hysteresis(sobel_image, low_threshold, high_threshold)         #200
    reconvert_to_img(hyst_image, h, w, name_image)                              #050 , if you want to save the image (as e.g. a .png)
    if hysteresis_save:
        plt.imsave(currentDir + 'Hyst.jpg', hyst_image, cmap='RGB', format='jpg')
    tbReturned = make_detection_matrix(hyst_image, h, w)                        #081
    if timed:
        t1 = time.time()
        global time_processing
        time_processing = t1 - t0
    return tbReturned


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
    return matrix[row][column] == 1


def is_labeled(matrix, row, column):
    """
    :return: true if the element at the given row and given column has a value different from 0.
            Meaning the element is a pixel as part of an object.
    """
    return matrix[row][column] != 0


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
    t0 = time.time()
    # print("object detecting started")
    nb_rows, nb_columns = len(matrix), len(matrix[0])
    nb_object_collision = []
    if printing:
        print("nb_rows, nb_columns =", nb_rows, nb_columns)
    counter = 0
    for row in range(nb_rows):
        for column in range(nb_columns):
            if matrix[row][column]:
                if is_connected(matrix, row, column):
                    matrix[row][column] = get_lowest_adjacent_label(matrix, row, column, nb_object_collision)
                else:
                    matrix[row][column] = counter + 1
                    counter += 1
    counter2 = 0
    print("Counter:     ", counter)
    flattened_matrix = flatten_matrix(matrix)
    print("Max found:   ", max(flattened_matrix))
    t1 = time.time()
    if printing:
        print("Number of objects: ", counter - len(nb_object_collision))
        print("Detect Objects:      ", t1-t0)


####### MAIN #######
def main():
    t0 = time.time()
    # 1) take a picture
    if kinectFreeTesting:
        color_image = "1_rechthoeken.png"
    else:
        color_image, depth_image = kinect_to_pc(1080, 1920, 4)
    # 2) start the image processing
    matrix = process_image(color_image) #241

    t1 = time.time()
    # 3) start looking for objects
    if detect:
        detect_objects(matrix)
    t2 = time.time()
    if timed:
        print("PROCESSING:              ", time_processing)
        print("|-> Grayscale:           ", time_grayscale)
        print("|-> Gaussian :           ", time_Gaussian)
        print("|-> Sobel    :           ", time_sobel)
        print("|-> Hysteris :           ", time_Hysteris)
        print("    |-> Flatten Matrix:  ", time_flatten)
        print("|-> Reconvert:           ", time_reconvert)
        print("|-> Make Detection :     ", time_detection)
        print("DETECTION OBJECTS:       ", t2-t1)
        print("==========================================")
        print("TOTAL:                   ", t2-t0)


if __name__ == '__main__':

    main()
