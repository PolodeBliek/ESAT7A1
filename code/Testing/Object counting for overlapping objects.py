from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import numpy as np
import time
import peakutils
import copy
from scipy import ndimage

start_time = time.time()


### INITIALIZE
kinectDepth            = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
currentDir             = os.path.dirname(os.path.abspath(__file__))
takeGroundImage        = False
kinectCamera           = True

### SHOW GRAPHS
plotGraphs             = False

### CONSTANTS
pd                     = 4     # distance between pixels
iv                     = 3     # interval distance / 2
md                     = 15    # minimum distance to count as new object
nb_overlaps_for_object = 30    # minimum overlaps to count as an object

nb_objects_color = None


def get_depth_frame():
    noDepth = True
    depth_image_size = (424, 512)
    while noDepth:
        if kinectDepth.has_new_depth_frame():
            noDepth = False
            depth_frame = kinectDepth.get_last_depth_frame()
            depth_frame = depth_frame.reshape(depth_image_size)
            depth_frame_flipped = cv2.flip(depth_frame, 1)

            if takeGroundImage:
                pickle.dump(depth_frame_flipped, open(currentDir + "/GroundDepth.pkl", "wb"))

            else:
                groundMatrix = pickle.load(open(currentDir + "/GroundDepth.pkl", "rb"))
                diff = np.subtract(groundMatrix.astype(np.int16), depth_frame_flipped.astype(np.int16))
                diff = abs(diff)
                diff = np.where(diff <= 5, 0, diff)
                diff = np.where(diff > 300, 0, diff)
                return diff


def has_overlapping_objects(frame):
    frame = copy.deepcopy(frame)
    nb_cols = len(frame)
    nb_rows = len(frame[0])
    nb_overlaps = 0

    for col in range(0, nb_cols - pd * 3):
        for row in range(0, nb_rows):
            value1 = frame[col][row]
            value2 = frame[col + pd][row]
            value3 = frame[col + 2 * pd][row]
            value4 = frame[col + 3 * pd][row]

            if value1 != 0 and value2 != 0 and value3 != 0 and value4 != 0:
                if value1 - iv <= value2 <= value1 + iv:
                        if not (value1 - md <= value3 <= value1 + md):
                            if value3 - iv <= value4 <= value3 + iv:
                                nb_overlaps += 1
                                frame[(col + int(3 * pd / 2))][row] = 500

    for col in range(0, nb_cols):
        for row in range(0, nb_rows - 3 * pd):
            value1 = frame[col][row]
            value2 = frame[col][row + pd]
            value3 = frame[col][row + 2 * pd]
            value4 = frame[col][row + 3 * pd]

            if value1 != 0 and value2 != 0 and value3 != 0 and value4 != 0:
                if value1 - iv <= value2 <= value1 + iv:
                    if not (value1 - md <= value3 <= value1 + md):
                        if value3 - iv <= value4 <= value3 + iv:
                            nb_overlaps += 1
                            frame[col][row + int(3 * pd / 2)] = 500

    return nb_overlaps, frame


def get_object_list(all_coords):
    chained_pixels, nb_chains = ndimage.label(all_coords)

    possible_objects = []
    for label in range(1, nb_chains + 1):
        possible_objects.append(np.transpose(np.nonzero(chained_pixels == label)))

    objects = []
    for area in possible_objects:
        if len(area) > 100:
            objects.append(area)

    return objects


def depth_counting(objects_c, depth_frame, nb_objects_color= None):
    depth_frameCopy = copy.deepcopy(depth_frame)
    depth_frame2 = copy.deepcopy(depth_frame)

    object_counter = len(objects_c)
    colorCounter = 0
    plottedPixels = set()

    for area in objects_c:
        z_values = []
        for pixel in area:
            z_values.append(depth_frame[tuple(pixel)])

        for nb in range(0, max(z_values)):
            z_values.append(nb)

        different_z_values = list(set(z_values))

        z_value_count = []
        for element in different_z_values:
            n = z_values.count(element)
            z_value_count.append(n)

        peaks = peakutils.indexes(np.array(z_value_count), thres=0.1, min_dist=15)

        if len(peaks) > 1:
            object_counter += (len(peaks) - 1)

        for peak in peaks:
            colorCounter += 1
            for pixel in area:
                pixel = tuple(pixel)
                if peak - 10 <= depth_frame[pixel] <= peak + 10:
                    plottedPixels.add(pixel)
                    depth_frameCopy[pixel] = 50*(colorCounter + 1)
                    depth_frame2[pixel] = 0
                else:
                    if pixel not in plottedPixels:
                        depth_frameCopy[pixel] = 0

        if plotGraphs:
            plt.plot(z_value_count, label='y = x')
            plt.title('Depth values and numbers')
            plt.ylabel('Y Axis')
            plt.yticks(different_z_values)
            plt.show()

    return object_counter, depth_frameCopy, depth_frame2


def plot_results(depth_frame, overlap, result, result2, nb):
    f = plt.figure()
    n = 4
    images = [depth_frame, overlap, result, result2]
    names = ["Depth frame", "Overlap visualized", "Depth detection, objects =" + str(nb), "Remaining objects"]
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(names[i])
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show(block=True)


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


def main():
    if kinectCamera:
        depth_frame = get_depth_frame()[100:310,:450]

    else:
        depth_frame = pickle.load(open(currentDir + "/DepthDiffTest1.pkl", "rb"))[50:350,:350]

    if not takeGroundImage:
        nb_overlaps, overlapFrame = has_overlapping_objects(depth_frame)
        if nb_overlaps >= nb_overlaps_for_object - 30:
            print('OVERLAP')
            object_estimate = get_object_list(depth_frame)
            result, resultFrame, resultFrame2 = depth_counting(object_estimate, depth_frame, nb_objects_color)

            resultFrame2 = cv2.medianBlur(resultFrame2, 5)
            resultFrame2 = sobel(resultFrame2)
            resultFrame2 = ndimage.binary_fill_holes(resultFrame2)
            objectsLeft = get_object_list(resultFrame2)
            remainingResult = len(objectsLeft)
            print('Detected:', result, '| Undetected:', remainingResult, '| TOTAL OBJECTS:', result + remainingResult)
            print('Time:', time.time() - start_time)

            plot_results(depth_frame, overlapFrame, resultFrame, resultFrame2, result)

        else:
            print('NO OVERLAP')
            plt.imshow(depth_frame)
            plt.title('test')
            plt.show()



main()
