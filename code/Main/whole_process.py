import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import numpy as np
from statistics import mean
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import time
import pickle
from sklearn.cluster import DBSCAN

import ntpath

#Test Variables
timed                  = True
kinectFreeTesting      = False
printing               = True
show                   = False
gauss_repetitions      = 1
low_threshold          = 40  # hysteresis
high_threshold         = 120  # hysteresis
grayscale_save         = True
gauss_save             = True
sobel_save             = True
hysteresis_save        = True
detection_matrix_save   = True
detect                 = False
currentDir             = os.path.dirname(os.path.abspath(__file__)).replace("code\\Main", "")
# print(currentDir)

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
def get_globals():
    # for the GUI
    return globals()


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

            cv2.imwrite(currentDir + "KinectColorPicture.png", color_flipped)  # Save

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
            flipped = cv2.flip(colorized_frame, 1)
        #cv2.imshow('depth', colorized_frame)
            cv2.imwrite(currentDir + "KinectDepthPicture.png", flipped)  # Save

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

    gray_image = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)

    if timed:
        t1 = time.time()
        global time_grayscale
        time_grayscale = t1 - t0

    return gray_image


def gaussian_blur(image, reps):
    if timed:
        t0 = time.time()

    if reps != 0:
        GaussianKernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
        for i in range(reps):
            image = ndimage.convolve(image, GaussianKernel)

    if timed:
        t1 = time.time()
        global time_Gaussian
        time_Gaussian = t1 - t0

    return image


def sobel(image):
    if timed:
        t0 = time.time()

    image = image.astype(np.int32)  # heel belangrijk, anders doet convolve vreemde dingen
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # initialize new images
    newHorizontalImage = ndimage.convolve(image, horizontal)
    newVerticalImage = ndimage.convolve(image, vertical)
    newVerticalImage = np.square(newVerticalImage)
    newHorizontalImage = np.square(newHorizontalImage)

    gradient = newHorizontalImage + newVerticalImage
    gradient = np.sqrt(gradient)

    max_grads = list(map(max, gradient))
    max_grad = max(max_grads)
    print(f"max grad is {max_grad}")
    # gradient *= 255/max_grad

    gradient = gradient.astype(np.uint8)  # reconvert range to (0, 255)

    if timed:
        t1 = time.time()
        global time_sobel
        time_sobel = t1 - t0

    # get the max value of gradient, just for testing

    return gradient


def hysteresis(image, low, high):
    """
    returns a matrix of 0s and 1s indicating the edges after thresholding
    """
    if timed:
        t0 = time.time()

    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image >= high
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]  # .astype(np.uint8)
    # transform from True/False to 1/0
    thresholded = thresholded.astype(np.uint8)

    if timed:
        t1 = time.time()
        global time_Hysteris
        time_Hysteris = t1 - t0
    return thresholded


# start the processing/filtering loop
def process_image(image):
    if timed:
        t0 = time.time()
    if printing:
        print("start image processing")

    # if the image doesn't come straight from the Kinect, but is a selected picture, open the selected picture
    if isinstance(image, str):  # 'image' is an absolute path
        name_image = ntpath.basename(image)
        image = np.array(Image.open(image))  # .astype(np.uint8)
    else:
        name_image = "KinectColorPicture"

    global gauss_repetitions, low_threshold, high_threshold, gauss_save, grayscale_save, sobel_save, hysteresis_save, \
        detetion_matrix_save, currentDir

    h, w, d = image.shape  # the height, width and depth of the image
    if printing:
        print("image.shape = ", h, w)

    # image processing/filtering process
    gray_image = grayscale(image)                                               #162
    if grayscale_save:
        plt.imsave(currentDir + 'Gray.jpg', gray_image, cmap='gray', format='jpg')
    blurred_image = gaussian_blur(gray_image, gauss_repetitions)                #176
    if gauss_save:
        plt.imsave(currentDir + 'Gauss.jpg', blurred_image, cmap='gray', format='jpg')
    sobel_image = sobel(blurred_image)                                          #193
    if sobel_save:
        plt.imsave(currentDir + 'Sobel.jpg', sobel_image, cmap='gray', format='jpg')
    hyst_matrix = hysteresis(sobel_image, low_threshold, high_threshold)         #228
    hyst_image = hyst_matrix*255                 #076
    if hysteresis_save:
        plt.imsave(currentDir + 'Hyst.jpg', hyst_image, cmap='gray', format='jpg')
    # detection_matrix = ndimage.binary_fill_holes(hyst_matrix)  # gaten vullen, mocht je willen
    if detection_matrix_save:
        pickle.dump(hyst_matrix, open(currentDir + "DetectionMatrix.pkl", "wb"))
    if timed:
        t1 = time.time()
        global time_processing
        time_processing = t1 - t0
    return hyst_matrix


####### DETECTING OBJECTS #######
def matrix_to_coordinates(matrix):
    nb_rows, nb_columns = matrix.shape
    d = []
    for row in range(nb_rows):
        for column in range(nb_columns):
            if matrix[row][column] == 1:
                d.extend(np.array([[row, column]]))

    return d


def plot_image(d, matrix, db):
    for i in range(len(d)):
        row = d[i][0]
        column = d[i][1]
        matrix[row][column] = db.labels_[i]
    plt.imshow(matrix)
    plt.show()


# start the object detection loop
def detect_objects(matrix):
    t0 = time.time()

    image = Image.fromarray(matrix)
    image = image.resize(size=(int(len(matrix) / 2), int(len(matrix[0]) / 2)))
    matrix = np.array(image)

    d = matrix_to_coordinates(matrix)
    db = DBSCAN(eps=3, min_samples=5).fit(d)

    t1 = time.time()

    # plot_image(d, matrix, db)
    if printing:
        print("NUMBER OF OBJECTS:", max(db.labels_))
        print("Detect Objects:      ", t1-t0)


####### MAIN #######
def main():
    t0 = time.time()
    # 1) take a picture
    if kinectFreeTesting:
        color_image = "1_rechthoeken.png"
    else:
        color_image, depth_image = kinect_to_pc(1080, 1920, 4)  #110
    # 2) start the image processing
    matrix = process_image(color_image)  #272

    t1 = time.time()
    # 3) start looking for objects
    if detect:
        detect_objects(matrix)  #339
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
