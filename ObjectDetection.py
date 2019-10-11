# This is a program to detect and count object from a binary matrix,
#  matrix with 0 en 1, where 1 stands for pixel from an object and 0 for empty, the background.

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

start_time = time.time()

matrix = pickle.load(open("matrix_vr_annelies.pkl", "rb"))
nb_columns = len(matrix[0])
nb_rows = len(matrix)
nb_object_collision = []


# TODO: find a better way to substract the colliding pixels from the same objects from the total count

def get_label(row, column):
    """
    :return: the value of the element located at the given row and given column.
    """
    return matrix[row][column]


def get_label_left(row, column):
    """
    :return: the value of the element left of the element located at the given row and the given column.
    """
    return matrix[row][column - 1]


def get_label_above(row, column):
    """
    :return: the value of the element above of the element located at the given row and the given column.
    """
    return matrix[row - 1][column]


def label_is_one(row, column):
    """
    :return: true if the element at the given row and given column has a value of 1.
    """
    return get_label(row, column) == 1


def is_labeled(row, column):
    """
    :return: true if the element at the given row and given column has a value different from 0.
            Meaning the element is a pixel as part of an object.
    """
    return get_label(row, column) != 0


def is_connected(row, column):
    """
    :return: true if the element at the given row and the given column is next to or under
                an element that is not zero
    """
    if row == 0 and column == 0:
        return False
    if column == 0:
        return is_labeled(row - 1, column)
    if row == 0:
        return is_labeled(row, column - 1)
    else:
        return is_labeled(row, column - 1) or is_labeled(row - 1, column)


def solve_collision_detection(lowest_label, highest_label):
    """
    :param lowest_label:
    :param highest_label:
    :return: a matrix where the elements with a value equal to the given highest_label are replaced by
            elements with a value equal to the given lowest_label
    """
    for i in range(nb_rows):
        for (e, element) in enumerate(matrix[i]):
            if element == highest_label:
                matrix[i][e] = lowest_label
                # print(matrix[i])


# TODO: Divide this function into two !!!

def get_lowest_adjacent_label(row, column):
    """
    :return: the lowest value of the adjacent labels.
    """
    label_left = get_label_left(row, column)
    label_above = get_label_above(row, column)
    if row == 0 or label_above == 0:
        return label_left
    if column == 0 or label_left == 0:
        return label_above
    if label_above != label_left:
        solve_collision_detection(min(label_left, label_above), max(label_left, label_above))
        nb_object_collision.append(1)
    return min(label_left, label_above)


def main():
    counter = 0
    for row in range(nb_rows):
        for column in range(nb_columns):
            if label_is_one(row, column):
                if is_connected(row, column):
                    matrix[row][column] = get_lowest_adjacent_label(row, column)
                else:
                    matrix[row][column] = counter + 1
                    counter += 1

    print("Number of objects: ", counter - len(nb_object_collision))
    plt.imshow(matrix)
    plt.show()


main()
print("--- %s seconds ---" % (time.time() - start_time))
