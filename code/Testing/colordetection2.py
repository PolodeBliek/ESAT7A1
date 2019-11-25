import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

pkl = "TEST.pkl"
m = 1*np.array(pickle.load(open("C:/Users/Administrator/PycharmProjects/ESAT7A1/" + pkl, "rb")))

## CONSTANTS ##
counter = 0
min_nb_pixels = 100

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
    matrix = np.where(matrix < 500, 0, matrix)

    objects = []
    for row in range(nb_rows):
        for column in range(nb_columns):
            if matrix[row][column] != 0:
                objects.append((row, column))
    return objects


def are_part_of_same_object(pixels, matrix, counter):
    cluster = [pixels[0]]
    pixels.remove(pixels[0])
    verified = set()
    end = False

    while not end:
        found_next_pixel = False
        for pixel1 in cluster:
            neighbours = return_pixel_neighbours(pixel1)
            for pixel2 in neighbours:
                if pixel2 not in verified:
                    if pixel2 in pixels:
                        cluster.append(pixel2)
                        pixels.remove(pixel2)
                        found_next_pixel = True

            verified.add(pixel1)
            cluster.remove(pixel1)
        if found_next_pixel is False:
            end = True
            if len(verified) > min_nb_pixels:
                print("OBJECT NR.", len(verified), counter + 1, " : ", verified)
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

    return pixels, len(verified)

def main(counter, matrix):

    start_time = time.time()
    objects = is_part_of_an_object(len(matrix), len(matrix[0]), matrix)
    print(len(objects))

    while len(objects) > min_nb_pixels:
        objects, len_object = are_part_of_same_object(objects, matrix, counter)
        if len_object > min_nb_pixels:
            counter += 1

    print("Number of objects: ", counter, "/ Time: ", time.time() - start_time)
    plt.imshow(matrix)
    plt.show()
    plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/test_result.png")

main(counter, m)




