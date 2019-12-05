import numpy as np
from scipy import ndimage as ndi

# print(f"log_not mask = \n{mask}")


def fill_holes_(input_array, structure=None, output=None, origin=0):
    mask = np.logical_not(input_array)  # invert input
    tmp = np.zeros(mask.shape)  # just array of zeros the same size as input
    output = ndi.binary_dilation(tmp, structure, -1, mask, None, 1, origin)
    np.logical_not(output, output)  # invert again
    return output.astype(np.uint8)  # bool to 0 and 1


a1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

print(fill_holes_(a1))
