print("IMPORTING")
import time
t0 = time.time()
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
t1 = time.time()
import numpy as np
from statistics import mean
t2 = time.time()
import os
import pickle
t3 = time.time()
import matplotlib.pyplot as plt
import cv2 #Moeten we dit echt allemaal importen?
from scipy import ndimage
#from sklearn.cluster import DBSCAN
from PIL import Image
import ntpath
t4 = time.time()

#Test Variables
timed                  = True
kinectFreeTesting      = True
printing               = True
show                   = False
gauss_repetitions      = 1
low_threshold          = 10  # hysteresis
high_threshold         = 120  # hysteresis
grayscale_save         = False
gauss_save             = False
sobel_save             = True
hysteresis_save        = True
detection_matrix_save  = True
detect                 = False
currentDir             = os.path.dirname(os.path.abspath(__file__)).replace("code\\Main", "")


if timed:
    time_import     = t1 - t0
    time_gem_kleur  = 0
    time_flatten    = 0
    time_reconvert  = 0
    time_detection  = 0
    time_grayscale  = 0
    time_sobel      = 0
    time_Gaussian   = 0
    time_Hysteris   = 0
    time_processing = 0
    time_save_gray  = 0
    time_save_hyst  = 0
    time_save_detec = 0
    time_save_gauss = 0
    time_save_sobel = 0
    time_fill       = 0
    time_kinect_imp = t1-t0
    time_math_imp   = t2-t1
    time_misc_imp   = t3-t2
    time_img_imp    = t4-t3


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

    gradient = np.interp(gradient, (gradient.min(), gradient.max()), (0, 255))  # rescale to (0, 255)
    gradient = gradient.astype(np.uint8)  # reconvert floats between 0 and 255 to uint8's

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
        print("START IMAGE PROCESSING")

    # if the image doesn't come straight from the Kinect, but is a selected picture, open the selected picture
    global gauss_repetitions, low_threshold, high_threshold, gauss_save, grayscale_save, sobel_save, hysteresis_save, \
    detetion_matrix_save
    global time_save_gray, time_save_hyst, time_save_detec, time_save_gauss, time_save_sobel, time_fill
    if isinstance(image, str):  # 'image' is an absolute path
        name_image = ntpath.basename(image)
        image = np.array(Image.open(currentDir + "testImages\\"  + image))  # .astype(np.uint8)

    else:
        name_image = "KinectColorPicture"


    h, w, d = image.shape  # the height, width and depth of the image

    # image processing/filtering process
    gray_image = grayscale(image)                                               #162
    t_a = time.time()
    if grayscale_save:
        plt.imsave(currentDir + 'Gray.jpg', gray_image, cmap='gray', format='jpg')
    t_b = time.time()
    time_save_gray = t_b - t_a
    blurred_image = gaussian_blur(gray_image, gauss_repetitions)                #176
    t_a = time.time()
    if gauss_save:
        plt.imsave(currentDir + 'Gauss.jpg', blurred_image, cmap='gray', format='jpg')
    t_b = time.time()
    time_save_gauss = t_b - t_a
    sobel_image = sobel(blurred_image)                                          #193
    t_a = time.time()
    if sobel_save:
        plt.imsave(currentDir + 'Sobel.jpg', sobel_image, cmap='gray', format='jpg')
    t_b = time.time()
    time_save_sobel = t_b - t_a
    hyst_matrix = hysteresis(sobel_image, low_threshold, high_threshold)         #228
    hyst_image = hyst_matrix*255                 #076
    t_a = time.time()
    if hysteresis_save:
        plt.imsave(currentDir + 'Hyst.jpg', hyst_image, cmap='gray', format='jpg')
    t_b = time.time()
    time_save_hyst = t_b - t_a
    # detection_matrix = ndimage.binary_fill_holes(hyst_matrix)  # gaten vullen, mocht je willen
    if detection_matrix_save:
        t_a = time.time()
        pickle.dump(hyst_matrix, open(currentDir + "DetectionMatrix.pkl", "wb"))
        t_b = time.time()
        time_save_detec = t_b - t_a
    t3 = time.time()
    matrix = ndimage.morphology.binary_fill_holes(hyst_matrix)
    matrix = np.where(matrix, 1, 0)
    plt.imsave(currentDir + "Filled.png", matrix, cmap = "gray", format = "png")
    matrix = sobel(matrix)
    pickle.dump(matrix, open(currentDir + "FilledMatrix.pkl", "wb"))
    plt.imsave(currentDir + "Fill.jpg", matrix, cmap = 'gray', format = 'jpg')
    t4 = time.time()
    time_fill = t4 - t3
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
    matrix = ndimage.morphology.binary_fill_holes(matrix)
    t1 = time.time()
    plt.imshow(matrix)
    plt.show()
    plt.imsave(currentDir + "Fill.jpg", matrix, cmap = 'gray', format = 'jpg')
    #image = Image.fromarray(matrix)
    #image = image.resize(size=(int(len(matrix) / 2), int(len(matrix[0]) / 2)))
    #matrix = np.array(image)

    #d = matrix_to_coordinates(matrix)
    #db = DBSCAN(eps=3, min_samples=5).fit(d)

    # plot_image(d, matrix, db)
    if printing:
        pass
        #print("NUMBER OF OBJECTS:", max(db.labels_))


####### MAIN #######
def main():
    if printing:
        print("START MAIN")
    t0 = time.time()
    # 1) take a picture
    if kinectFreeTesting:
        color_image = "kinectColor\\Gauss_3.jpg"
    else:
        color_image, depth_image = kinect_to_pc(1080, 1920, 4)  #110
    # 2) start the image processing
    matrix = process_image(color_image)  #272
    #times = pickle.load(open(currentDir + "/Times.pkl", "rb"))
    #times = [times[0] + time_processing, times[1] + time_grayscale, times[2] + time_Gaussian, times[3] + time_sobel, times[4] + time_Hysteris, times[5] + time_flatten, times[6] + time_reconvert, times[7] + time_detection]
    #pickle.dump(times, open(currentDir + "/Times.pkl", "wb"))


    t1 = time.time()
    # 3) start looking for objects
    if detect:
        detect_objects(matrix)  #339
    t2 = time.time()
    if timed:
        print("\n")
        print("IMPORTING:               ", time_import)
        print("|-> Kinect   Modules:    ", time_kinect_imp)
        print("|-> Math     Modules:    ", time_math_imp)
        print("|-> Misc     Modules:    ", time_misc_imp)
        print("|-> Image    Modules:    ", time_img_imp)
        print("PROCESSING:              ", time_processing)
        print("|-> Grayscale:           ", time_grayscale)
        print("    |-> Grayscale Save:  ", time_save_gray if grayscale_save else "TURNED OFF")
        print("|-> Gaussian :           ", time_Gaussian)
        print("    |-> Gaussian Save:   ", time_save_gauss if gauss_save else "TURNED OFF")
        print("|-> Sobel    :           ", time_sobel)
        print("    |-> Sobel Save:      ", time_save_sobel if sobel_save else "TURNED OFF")
        print("|-> Hysteris :           ", time_Hysteris)
        print("    |-> Hysteris Save:   ", time_save_hyst if hysteresis_save else "TURNED OFF")
        print("    |-> Matrix Save :    ", time_save_detec if detection_matrix_save else "TURNED OFF")
        print("|-> Fill & Save :        ", time_fill)
        print("DETECTION OBJECTS:       ", t2-t1 if detect else "TURNED OFF")
        print("==========================================")
        print("TOTAL:                   ", t2-t0 + time_import)
        time_list = (time_grayscale, time_Gaussian, time_sobel, time_Hysteris, time_flatten, time_reconvert, time_detection, time_fill, time_save_gray, time_save_hyst, time_save_detec, time_save_gauss, time_save_sobel)
        if abs(sum(time_list) - time_processing) > 0.1:
            print("ERROR! SOME PROCESSES SEEM TO HAVE GONE UNDETECTED BY THE TIMER")
            print("Difference:  ", abs(sum(time_list) -time_processing))
    if printing:
        print("\n")
        print("FINISHED")

if __name__ == '__main__':

    main()
