import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from PIL import Image

pkl = "FilledMatrix.pkl"
m = np.array(pickle.load(open("C:/Users/Polo/Documents/GitHub/ESAT7A1/" + pkl, "rb")))


## CONSTANTS ##
counter         = 0
min_nb_pixels   = 300
show            = True
save            = True
timed           = True
whileloop       = 0
Rest            = 0

def return_pixel_neighbours(pixel):
    """
        Returns a set of pixels surrounding the given pixel (could be written with for-loop = slower)
    """
    (row, col) = pixel
    return {(row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
            (row, col - 1), (row, col + 1), (row + 1, col - 1),
            (row + 1, col), (row + 1, col + 1)}

def is_part_of_an_object(nb_rows, nb_columns, matrix):
    """
        Returns a list which contains pixels in the given matrix being a part of an object
    """
    #matrix = np.where(matrix < 500, 0, matrix)
    nonzerolist = np.nonzero(matrix)
    objects = list(zip(nonzerolist[0], nonzerolist[1]))
    return objects


def are_part_of_same_object(pixels, matrix, counter):
    global whileloop, Rest
    cluster = [pixels[0]]
    pixels.remove(pixels[0])
    verified = set()
    end = False
    while not end:
        t_0 = time.time()
        found_next_pixel = False
        for pixel1 in cluster:
            (row, col) = pixel1
            neighbours = {(row - 1, col - 1), (row - 1, col), (row - 1, col + 1),(row, col - 1), (row, col + 1), (row + 1, col - 1),(row + 1, col), (row + 1, col + 1)}
            if len(neighbours.intersection(pixels)) == 0:
                pass
            else:
                for pixel2 in neighbours:
                    if (pixel2 not in verified) and (pixel2 in pixels):
                        cluster.append(pixel2)
                        pixels.remove(pixel2)
                        found_next_pixel = True

            verified.add(pixel1)
            cluster.remove(pixel1)
        t_1 = time.time()
        if found_next_pixel is False:
            end = True
            if len(verified) > min_nb_pixels:
                for element in verified:
                    (c_row, c_column) = element
                    matrix[c_row][c_column] = 40 * (counter + 2)  # Make spotted pixels visible in end result
            else:
                for element in verified:
                    (c_row, c_column) = element
                    matrix[c_row][c_column] = 0  # Make spotted pixels visible in end result

        for element in verified:
            (c_row, c_column) = element
            matrix[c_row][c_column] = 40*(counter + 2)  # Make spotted pixels visible in end result
        t_2 = time.time()
        whileloop += t_1 - t_0
        Rest += t_2 - t_1
    t_c = time.time()
    return pixels, len(verified)

def main(counter, matrix):
    t0 = time.time()
    objects = is_part_of_an_object(len(matrix), len(matrix[0]), matrix)
    t1 = time.time()
    while len(objects) > min_nb_pixels:
        objects, len_object = are_part_of_same_object(objects, matrix, counter)
        if len_object > min_nb_pixels:
            counter += 1
    t2 = time.time()

    print("Number of objects: ", counter)
    print("TOTAL:       ", t2-t0)
    if show:
        plt.imshow(matrix)
        plt.show()
    if save:
        plt.savefig("C:/Users/Polo/Documents/GitHub/ESAT7A1/test_result.png")

main(counter, m)
print("FIRST PART:  ", whileloop)
print("SECOND PART: ", Rest)
