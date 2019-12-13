#### OBJECT COUNTING ALGORITHM ESAT7A1

import matplotlib.gridspec as gridspec
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import numpy as np
import peakutils
import pickle
import math
import copy
import cv2
import os



# Used directory (GitHub)
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Main", "")

# Used variables in counting algorithm:
gauss_repetitions      = 1   # Times Gaussian blur
low_threshold          = 10  # Low threshold hysteresis
high_threshold         = 80  # High threshold hysteresis
pd                     = 4   # Distance pixels to compare
iv                     = 3   # Maximum height difference for same object
md                     = 15  # Minimum height difference for 2nd object
nb_overlaps_for_object = 30  # Minimum overlapping points for overlap


## 1. Auxiliary functions for plotting and GUI ##

def get_globals():
    return globals()

def show_images(d: dict):
    """
        Plot all the images in dict
    """
    n = len(d)
    k = int(math.sqrt(n)) + 1
    f = plt.figure(constrained_layout = True, num = f'Results:')
    spec = gridspec.GridSpec(ncols=k, nrows=n//k+1, figure=f)
    for i, key, vals in zip(range(n), d.keys(), d.values()):
        img, map_ = vals
        ax = f.add_subplot(spec[i//k, i % k])
        ax.imshow(img, cmap=map_)
        plt.xticks([])
        plt.yticks([])
        ax.set_xlabel(f"{key}")

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def save_images(d: dict):
    """
    Save all the images of the dictionary in their respective directories
    """
    if "Gray" in d.keys():
        storage_limit = 100
        grayname = "Gray"
        _, graypath = d[grayname]

        if os.path.exists(graypath + grayname + f"_{storage_limit}.jpg"):
            print("!!!STORAGE FULL!!!\nYou have stored the maximum capacity of pictures in your Image directory.\n"
                  "Delete some of the previous pictures or change 'storage_limit' in 'save_images'.")

        for i in range(1, storage_limit+1):
            # If Gray_{i} doesn't exist, save all the images with index 'i'
            if not os.path.exists(graypath + grayname + f"_{i}.jpg"):
                for name, vals in zip(d.keys(), d.values()):
                    img, path = vals
                    plt.imsave(path + name + f"_{i}.jpg", img, cmap='gray', format='jpg')
                print(f"saved as _{i}")
                break
    else:
        print("ERROR:\n'Gray' not found in the save dictionary keys, \nnone of the images have been saved.")


## 2. Take an RGB and a depth image ##

def is_connected():
    """
    Check the connection between the Kinect and computer. Return 'False' when not connected
    """
    cap = cv2.VideoCapture(1)
    return cap.isOpened()


def kinect_to_pc(kinectColor, kinectDepth, takeGroundImage):
    """
        Take an RGB and depth frame. When takeGroundImage = True, save ground image
    """
    noColor = True
    noDepth = True
    while noColor:
        if kinectColor.has_new_color_frame():
            color_frame = kinectColor.get_last_color_frame()
            noColor = False
            color_image_shape = (1080, 1920, 4)
            color_frame = color_frame.reshape(color_image_shape)
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            color_flipped = cv2.flip(color_frame, 1)

    while noDepth:
        if kinectDepth.has_new_depth_frame():
            noDepth = False
            depth_image_shape = (424, 512)
            depth_frame = kinectDepth.get_last_depth_frame()
            depth_frame = depth_frame.reshape(depth_image_shape)
            depth_frame_flipped = cv2.flip(depth_frame, 1)

            if takeGroundImage:
                pickle.dump(depth_frame_flipped, open(currentDir + "/GroundDepth.pkl", "wb"))
                return
            else:
                groundMatrix = pickle.load(open(currentDir + "/GroundDepth.pkl", "rb"))
                diff = np.subtract(groundMatrix.astype(np.int16), depth_frame_flipped.astype(np.int16))
                diff = np.where(diff <= 5, 0, diff)
                diff = np.where(diff > 300, 0, diff)

    return color_flipped, diff


## 3. Process the RGB image ##

def grayscale(image):
    """
        Grayscale the image
    """
    return (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)


def gaussian_blur(image, reps):
    """
        Blur the image 'reps' times with a chosen kernel
    """
    if reps != 0:
        GaussianKernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
        for i in range(reps):
            image = ndimage.convolve(image, GaussianKernel)
    return image


def sobel(image):
    """
        Filter out the edges of the image
        :return: a matrix with the magnitude of the color gradient of the input image, rescaled to range(0, 255)
    """
    image = image.astype(np.int32)                               # Important for convolution
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Define horizontal filter
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])    # Define vertical filter

    newHorizontalImage = ndimage.convolve(image, horizontal)     # Initialize new horizontal image
    newVerticalImage = ndimage.convolve(image, vertical)         # Initialize new vertical image

    gradient = np.sqrt(np.square(newVerticalImage) + np.square(newHorizontalImage))

    gradient = np.interp(gradient, (gradient.min(), gradient.max()), (0, 255))  # Rescale gradient
    gradient = gradient.astype(np.uint8)                                        # Reconvert to uint8

    return gradient


def hysteresis(image, low, high):
    """
        Returns a matrix consisting of zeros and ones indicating the edges after thresholding
    """
    low = np.clip(low, a_min=None, a_max=high)  # Ensure low always below high
    mask_low = image > low  # Seperate 'weak/strong' values (mask_low[*,*] == True) from 'noise' values (= False)
    mask_high = image >= high  # Seperate 'strong' values (= True) from 'weak/noise' values (= False)
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low)  # Detects connected pixels, returns groups and nb of groups
    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))  # List of how many high's each group has
    connected_to_high = sums > 0  # Seperate groups with at least one 'high' (=True) from groups without highs(=False)
    thresholded = connected_to_high[labels_low]  # Projecs the values of 'connected_to_high' onto the group of which the name (1,2,3...) is the same as the index of 'connected to high'
    thresholded = thresholded.astype(np.uint8) # Transform from True/False to 1/0

    return thresholded


## 4. Object counting 1 ##

def db_scan(matrix):
    matrix = matrix[::2, ::2].copy().astype(np.int32)          # New matrix with every other element of the rows and cols
    d = np.transpose(np.nonzero(matrix))                       # Get the indices of the ones
    db = DBSCAN(eps=6, min_samples=78,n_jobs=-1).fit(d)        # DBSCAN
    nb_objects = max(db.labels_) +1
    np.place(matrix, matrix, (db.labels_+1).astype(np.int32))  # Map the labels of DBSCAN over matrix (where matrix == 1)

    return matrix, nb_objects


## 5. Object counting 2 ##

def is_part_of_an_object(matrix):
    """
        Returns a list which contains pixels in the given matrix being a part of an object
    """
    nonzerolist = np.nonzero(matrix)
    objects = list(zip(nonzerolist[0], nonzerolist[1]))
    matrix = np.where(matrix != 0, 0, 0)
    return objects, matrix


def are_part_of_same_object(pixels, matrix, counter, min_nb_pixels):
    cluster = [pixels[0]]
    pixels.remove(pixels[0])
    connected_pixels = set()
    end = False

    while not end:
        found_next_pixel = False
        for pixel_c in cluster:

            (row, col) = pixel_c
            neighbours = {(row - 1, col - 1), (row - 1, col), (row - 1, col + 1), (row, col - 1),
                          (row, col + 1), (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)}.intersection(pixels)

            for pixel_n in neighbours:
                if pixel_n not in connected_pixels:
                    cluster.append(pixel_n)
                    pixels.remove(pixel_n)
                    found_next_pixel = True
            cluster.remove(pixel_c)
            connected_pixels.add(pixel_c)

        if found_next_pixel is False:
            end = True
            if len(connected_pixels) > min_nb_pixels:
                for element in connected_pixels:
                    (c_row, c_column) = element
                    matrix[c_row][c_column] = 40 * (counter + 1)  # Make spotted pixels visible in end result
            else:
                for element in connected_pixels:
                    (c_row, c_column) = element
                    matrix[c_row][c_column] = 0  # Make spotted pixels visible in end result

    return pixels, len(connected_pixels), matrix


def object_counting_from_scratch(matrix, min_nb_pixels):
    counter = 0
    objects, matrix_visual = is_part_of_an_object(matrix)
    end = False
    while not end:
        objects, len_object, matrix = are_part_of_same_object(objects, matrix_visual, counter, min_nb_pixels)
        if len_object > min_nb_pixels:
            counter += 1

        if len(objects) <= min_nb_pixels:
            end = True
            for element in objects:
                (c_row, c_column) = element
                matrix[c_row][c_column] = 0  # Make spotted pixels visible in end result

    return matrix, counter


## 6. Draw boxes ##

def draw_boxes(og_image, obj_image):
    """
        Draw boxes around the objects
    """
    image = Image.fromarray(og_image)
    coord = find_corners(obj_image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 100)

    for i in range(0, len(coord)):
        rechts_boven = (coord[i][1][1]*2, coord[i][0][0]*2)
        links_boven = (coord[i][2][1]*2, coord[i][0][0]*2)
        rechts_onder = (coord[i][1][1]*2, coord[i][3][0]*2)
        links_onder = (coord[i][2][1]*2, coord[i][3][0]*2)

        draw.line([rechts_boven, links_boven, links_onder, rechts_onder, rechts_boven], fill=(0, 0, 225), width=5)
        PixelToDistanceRatio = None
        if PixelToDistanceRatio == None:
            draw.text(links_onder, str(rechts_boven[0] - links_boven[0]) + "px x " + str(abs(rechts_boven[1] - rechts_onder[1])) + "px", font = font)
        else:
            draw.text(links_onder, str((rechts_boven[0] - links_boven[0])*PixelToDistanceRatio) + "cm x " + str(abs(rechts_boven[1] - rechts_onder[1])*PixelToDistanceRatio)+ "cm", font = font)

    return image


def find_corners(corner_img):
    """
        Returns the corners of the objects
    """
    number_of_elements = corner_img.max()
    allcoord = []

    for index in range(1, number_of_elements + 1):
        coord = np.where(corner_img == index)
        minx = min(coord[0])
        maxx = max(coord[0])
        miny = min(coord[1])
        maxy = max(coord[1])
        listCoord = list(zip(coord[0], coord[1]))
        Links = [x for x in listCoord if x[0] == minx]
        Rechts = [x for x in listCoord if x[0] == maxx]
        Boven = [x for x in listCoord if x[1] == miny]
        Onder = [x for x in listCoord if x[1] == maxy]

        allcoord.append([Links[0], Onder[0], Boven[0], Rechts[0]])

    return allcoord


## 7. Depth counting full algorithm ##

def has_overlapping_objects(frame):
    """
        Checks for overlapping points, return 'True' or 'False' if overlap is found/not found.
        Also returns the depth frame on which overlapping objects are marked.
    """
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

    if nb_overlaps >= nb_overlaps_for_object:
        overlap = True
    else:
        overlap = False

    return overlap, frame


def get_object_list(all_coords):
    """
        Returns groups of connected pixels from depth frame
    """
    chained_pixels, nb_chains = ndimage.label(all_coords)

    possible_objects = []
    for label in range(1, nb_chains + 1):
        possible_objects.append(np.transpose(np.nonzero(chained_pixels == label)))

    objects = []
    for area in possible_objects:
        if len(area) > 100:
            objects.append(area)

    return objects


def depth_counting(objects_c, depth_frame, nb_objects_color):
    """
        Counts mainly flat objects on top of each other, returns nb of counted objects, depth frame in which
        counted objects are indicated and depth frame with pixels which aren't counted
    """
    depth_frameCopy = copy.deepcopy(depth_frame)
    depth_frame2 = copy.deepcopy(depth_frame)

    object_counter = nb_objects_color
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
                if peak - md + 3 < depth_frame[pixel] < peak + md - 3:
                    plottedPixels.add(pixel)
                    depth_frameCopy[pixel] = 50*(colorCounter + 1)
                    depth_frame2[pixel] = 0
                else:
                    if pixel not in plottedPixels:
                        depth_frameCopy[pixel] = 0

    return object_counter, depth_frameCopy, depth_frame2


def remaining_objects(all_coords):
    """
        Returns groups of connected pixels from depth frame, currently not used
    """
    chained_pixels, nb_chains = ndimage.label(all_coords)

    possible_objects = []
    for label in range(1, nb_chains + 1):
        possible_objects.append(np.transpose(np.nonzero(chained_pixels == label)))

    objects = []
    for area in possible_objects:
        if len(area) > 100:
            objects.append(area)

    frame = np.zeros((len(all_coords), len(all_coords[0])))
    for object in objects:
        for pixel in object:
            pixel = tuple(pixel)
            frame[pixel] = 50

    return objects, frame


def depth_general(depth_frame, nb_objects_color):
        object_estimate = get_object_list(depth_frame)
        result, resultFrame, resultFrame2 = depth_counting(object_estimate, depth_frame, nb_objects_color)
        # Code below could be used to detect non-flat objects
        #resultFrame2 = cv2.medianBlur(resultFrame2, 3)
        #objectsLeft, resultFrame2 = remaining_objects(resultFrame2)
        #remainingResult = len(objectsLeft)

        return depth_frame, resultFrame, resultFrame2, result