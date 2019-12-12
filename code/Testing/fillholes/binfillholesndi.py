import numpy as np
from scipy import ndimage as ndi

# print(f"log_not mask = \n{mask}")


def fill_holes_(input_array):
    mask = np.logical_not(input_array)  # invert input
    tmp = np.zeros(mask.shape)  # just array of zeros the same size as input
    output = dilation_(tmp, -1, mask, None, 1)
    np.logical_not(output, output)  # invert again
    return output.astype(np.uint8)  # bool to 0 and 1


def dilation_(input_array, iterations=1, mask=None, output=None, border_value=0, origin=0):
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    origin = [0, 0]
    return ndi.morphology._binary_erosion(input_array, structure, iterations, mask, output, border_value, origin, 1, 0)


a1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])


def whole_fill_thing(array):
    inverse_ = fill_holes_(array)
    print(inverse_)
    # dilated_ = dilation_(filled_)


whole_fill_thing(a1)

