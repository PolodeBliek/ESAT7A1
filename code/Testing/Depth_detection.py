from PIL import Image
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

img = Image.open("test1.png") # Use image

## CONSTANTS ##
counter = 0
min_nb_pixels = 3
min_depth_value = 600
max_depth_value = 700
resizeValue = 2
resizeImage = False

if resizeImage:
    (width, height) = img.size
    img = img.resize((int(width/resizeValue), int(height/resizeValue))) #Go faster, lose image quality

### IMAGE PROCESSING ###

def get_depth_frame():

    kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    noDepth = True
    depth_image_size = (424, 512)

    while noDepth:
        if kinect2.has_new_depth_frame():
            depth_frame = kinect2.get_last_depth_frame()
            noDepth = False

            depth_frame = depth_frame.reshape(depth_image_size)
            depth_frame = depth_frame * (256.0 / np.amax(depth_frame))
            colorized_frame = cv2.applyColorMap(np.uint8(depth_frame), cv2.COLORMAP_JET)
    return colorized_frame



### OBJECT COUNTING ###
def collapse_items():
    pass
    # def total_surrounding_value(row, column, matrix):
    #     """
    #     :return: the sum of the value of the surrounding pixels + given pixel in a 9-block area
    #     """
    #     sum = 0
    #     for delta_row in [-1, 0, 1]:
    #         for delta_column in [-1, 0, 1]:
    #             if row + delta_row <= len(matrix) and column + delta_column <= len(matrix[0]):
    #                 sum += matrix[row + delta_row][column + delta_column]
    #     return sum
    #
    # def is_pixel_neighbour(pixel, list_of_pixels):
    #     """
    #     :return: true if pixel is a surrounding element
    #     """
    #     (row, col) = pixel
    #     for row_dif in [1, 0, -1]:
    #         for col_dif in [1, 0, -1]:
    #             if (row + 4 * row_dif, col + 4 * col_dif) in list_of_pixels:
    #                 return True
    #     return False

def return_pixel_neighbours(pixel):
    """
        Returns a set of pixels surrounding the given pixel (could be written with for-loop = slower)
    """
    (row, col) = pixel
    return {(row - 4, col - 4), (row - 4, col), (row -4, col + 4),
            (row, col - 4), (row, col + 4), (row + 4, col - 4),
            (row + 4, col), (row + 4, col + 4)}

def is_part_of_an_object(nb_rows, nb_columns, matrix):
    """
        Returns a list which contains pixels in the given matrix being a part of an object
    """
    objects = []
    for row in range(nb_rows):
        for column in range(nb_columns):
            c_row = 4 * row + 2
            c_column = 4 * column + 2
            if min_depth_value <= matrix[c_row][c_column] <= max_depth_value:
                objects.append((c_row, c_column))
    return objects

def are_part_of_same_object(pixels, min, matrix, counter):

    cluster = [pixels[0]]
    pixels.remove(pixels[0])
    verified = set()
    end = False

    while not end:
        found_next_object = False
        for pixel1 in cluster:
            neighbours = return_pixel_neighbours(pixel1)
            for pixel2 in neighbours:
                if pixel2 not in verified:
                    if pixel2 in pixels:
                        cluster.append(pixel2)
                        pixels.remove(pixel2)
                        found_next_object = True
                        (c_row, c_column) = pixel2
                        matrix[c_row][c_column] = 40 * (counter + 2)  # Make spotted pixels visible in end result
            verified.add(pixel1)
            cluster.remove(pixel1)
        if len(cluster) <= min and pixels != []:
            cluster = [pixels[0]]
            pixels.remove(pixels[0])
        elif found_next_object is False:
            end = True
        for element in verified:
            (c_row, c_column) = element
            matrix[c_row][c_column] = 40*(counter + 2)  # Make spotted pixels visible in end result
    return pixels

def main(counter):

    matrix = np.array(get_depth_frame())

    n_columns = int(len(matrix[0]) / 4)
    n_rows = int(len(matrix) / 4)
    start_time = time.time()
    objects = is_part_of_an_object(n_rows, n_columns, matrix)

    while len(objects) > 3:
        objects = are_part_of_same_object(objects, min_nb_pixels, matrix, counter)
        counter += 1

    print("Number of objects: ", counter, "/ Time: ", time.time() - start_time)
    plt.imshow(matrix)
    plt.show()
    plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/test_result_depth.png")

main(counter)




