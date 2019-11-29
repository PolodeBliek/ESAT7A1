import numpy as np
import peakutils

objects = [...]
matrix = ...frame...


def depth_counting(objects_color, depth_frame):

    object_counter = len(objects_color)

    for area in objects_color:
        z_values = []
        for pixel in area:
            z = depth_frame[pixel]
            if z != 0:
                z_values.append(z)

        different_z_values = list(set(z_values))
        z_value_count = []
        for element in different_z_values:
            n = z_values.count(element)
            z_value_count.append(n)

        nb_peaks = len(peakutils.indexes(np.array(z_value_count), min_dist=15))

        if nb_peaks > 0:
            object_counter += (nb_peaks - 1)

    return object_counter

depth_counting(objects, matrix)

