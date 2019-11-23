# DON'T FORGET TO MAKE A DIRECTORY FOR THE IMAGES
import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import numpy as np
from statistics import mean
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
import time
import pickle
from sklearn.cluster import DBSCAN
import math

import ntpath

# path to ESAT7A1 directory of the github
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Main", "")

# Algorithm variables:
gauss_repetitions      = 1  # gaussian blur
low_threshold          = 10  # hysteresis
high_threshold         = 80  # hysteresis

# Test Variables
timed                  = True
printing               = True
show_imgs              = True
save_imgs              = False
detection_matrix_save  = False


###### HULPFUNCTIES #######
def get_globals():
    # for the GUI
    return globals()


def show_images(d: dict):
    """
    plot all the images,
    expects a dictionary where the key is the title and the value is a tuple of the img and the mapping colors
    """
    n = len(d)
    k = int(math.sqrt(n)) + 1
    f = plt.figure(constrained_layout=True, num='Results')
    spec = gridspec.GridSpec(ncols=k, nrows=n//k+1, figure=f)
    for i, key, vals in zip(range(n), d.keys(), d.values()):
        img, map_ = vals
        ax = f.add_subplot(spec[i//k, i % k])
        ax.imshow(img, cmap=map_)
        ax.set_xlabel(f"{key}")
        # plt.axis('off')
    plt.show()


def save_images(d: dict):
    """
    save all the images of the dictionary in their respective directories
    Gray is the only constant in life
    """
    if "Gray" in d.keys():
        storage_limit = 100
        grayname = "Gray"
        _, graypath = d[grayname]

        if os.path.exists(graypath + grayname + f"_{storage_limit}.jpg"):
            print("!!!STORAGE FULL!!!\nYou have stored the maximum capacity of pictures in your Image directory.\n"
                  "Delete some previous pictures or change 'storage_limit' in 'save_images'.")

        for i in range(1, storage_limit+1):
            # if Gray_{i} doesn't exist, save all the images with index 'i'
            if not os.path.exists(graypath + grayname + f"_{i}.jpg"):
                for name, vals in zip(d.keys(), d.values()):
                    img, path = vals
                    plt.imsave(path + name + f"_{i}.jpg", img, cmap='gray', format='jpg')
            break
    else:
        print("ERROR:\n'Gray' not found in the save dictionary keys, \nnone of the images have been saved.")


####### TAKING PICTURES #######
def kinect_to_pc(width, height, dimension):
    # https://github.com/daan/calibrating-with-python-opencv/blob/02c90e4291adfb2426072f8f0837033754fc3a55/kinect-v2/color.py
    color_flipped = None  # give them a default value
    depth_flipped = None

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    noPicture = True
    while noPicture:
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            noPicture = False

            color_frame = color_frame.reshape((width,height,dimension))
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            color_flipped = cv2.flip(color_frame, 1)

            # cv2.imwrite(currentDir + "KinectColorPicture.png", color_flipped)  # Save

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
            depth_flipped = cv2.flip(colorized_frame, 1)
            #cv2.imshow('depth', colorized_frame)
            # cv2.imwrite(currentDir + "KinectDepthPicture.png", depth_flipped)  # Save

    return color_flipped, depth_flipped


####### IMAGE PROCESSING #######
def grayscale(image):
    """
    :return: a grayscaled version of the input image
    """
    if isinstance(image, str):  # 'image' is an absolute path
        image = np.array(Image.open(image))  # .astype(np.uint8)

    gray_image = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)
    return gray_image


def gaussian_blur(image, reps):
    """
    blur the image 'reps' times with a chosen kernel
    :return: a slightly blurred version of the input image
    """
    if reps != 0:
        GaussianKernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
        for i in range(reps):
            image = ndimage.convolve(image, GaussianKernel)

    return image


def sobel(image):
    """
    filter out the edges of the image
    :return: a matrix with the magnitude of the color gradient of the input image, rescaled to range(0, 255)
    """
    image = image.astype(np.int32)  # heel belangrijk, anders doet convolve vreemde dingen
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # initialize new images
    newHorizontalImage = ndimage.convolve(image, horizontal)
    newVerticalImage = ndimage.convolve(image, vertical)

    # calculate magnitude gradient
    newVerticalImage = np.square(newVerticalImage)
    newHorizontalImage = np.square(newHorizontalImage)
    gradient = newHorizontalImage + newVerticalImage
    gradient = np.sqrt(gradient)

    gradient = np.interp(gradient, (gradient.min(), gradient.max()), (0, 255))  # rescale to (0, 255)
    gradient = gradient.astype(np.uint8)  # reconvert to uint8

    return gradient


def hysteresis(image, low, high):
    """
    :return: a matrix consisting of 0s and 1s indicating the edges after thresholding
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low  # seperate 'weak/strong' values (mask_low[*,*]==True) from 'noise' values (=False)
    mask_high = image >= high  # seperate 'strong' values (=True) from 'weak/noise' values (=False)
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low)  # detecting and naming islands, and also returns nb of islands (making a map of the islands)
    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))  # list of how many high's each island has
    connected_to_high = sums > 0  # seperate islands with at least one 'high' (=True) from islands without highs(=False)
    thresholded = connected_to_high[labels_low]  # special np thing, projecs the values of 'connected_to_high' onto the island of which the name (1,2,3...) is the same as the index of 'connected to high'
    # transform from True/False to 1/0
    thresholded = thresholded.astype(np.uint8)

    return thresholded


# start the processing/filtering loop
def process_image(image):
    # if the image doesn't come straight from the Kinect, but is a selected picture, open the selected picture
    if isinstance(image, str):  # 'image' is an absolute path
        image = np.array(Image.open(image))  # .astype(np.uint8)

    global gauss_repetitions, low_threshold, high_threshold

    # image processing/filtering process
    gray_image = grayscale(image)                                               #162
    blurred_image = gaussian_blur(gray_image, gauss_repetitions)                #176
    sobel_image = sobel(blurred_image)                                          #193
    hyst_matrix = hysteresis(sobel_image, low_threshold, high_threshold)         #228
    hyst_matrix = ndimage.binary_fill_holes(hyst_matrix)
    # hyst_image = hyst_matrix*255                 #076
    # plt.imsave('../images/hysteresis_images/Hyst.jpg', hyst_image, cmap='gray')
    # detection_matrix = ndimage.binary_fill_holes(hyst_matrix)  # gaten vullen, mocht je willen

    # pickle.dump(hyst_matrix, open(currentDir + "HystTest.pkl", "wb"))

    return gray_image, blurred_image, sobel_image, hyst_matrix


####### DETECTING OBJECTS #######
def detect_objects(matrix):
    matrix = matrix[::2, ::2].copy().astype(np.int32)  # new matrix with every other element of the rows and cols
    d = np.transpose(np.nonzero(matrix))  # get the indices of the ones
    db = DBSCAN(eps=3, min_samples=5).fit(d)  # DBSCAN
    np.place(matrix, matrix, db.labels_+1)  # map/place the labels of db_scan over matrix (where matrix==1)

    return matrix


####### MAIN #######
def main():
    global timed, show_imgs
    t0 = time.time()

    # 1) take a picture
    color_image, depth_image = kinect_to_pc(1080, 1920, 4)  #110
    t1 = time.time()

    # 2) start the image processing
    gray, gauss, sobel, hyst_matrix = process_image(color_image)  #272
    hyst_img = hyst_matrix * 255  # for saving and displaying the hysteresis matrix
    t2 = time.time()

    # 3) start looking for objects
    dbScan_img = detect_objects(hyst_matrix)  #339
    t3 = time.time()
    if timed:
        print("TAKE PICTURES:           ", t1-t0)
        print("PROCESSING:              ", t2-t1)
        print("DETECTION OBJECTS:       ", t3-t2)
        print("==========================================")
        print("TOTAL:                   ", t3-t0)

    if show_imgs:
        show_images(color_image, depth_image, gray, gauss, sobel, hyst_img)

    if save_imgs:
        save_images(color_image, depth_image, gray, gauss, sobel, hyst_img, dbScan_img)


if __name__ == '__main__':

    main()
