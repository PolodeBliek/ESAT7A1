import numpy as np
from scipy import ndimage as ndi

a1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
               [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def get_bin_edges(array):
    """
    Get the edges of objects in a bitmap.
    The edges are all the pixels with value 1 that are connected to at least one 0
    """
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    conv = ndi.convolve(array, kernel)  # convolve the array, if the pixel is connected to a black one, conv=3 or less
    edgemap = np.where(conv<4, array, 0)  # pull the values of array back over the pixels connected to a black one, the rest becomes 0
    print(edgemap)
    return edgemap


edges = get_bin_edges(a1)
filled = ndi.binary_fill_holes(edges).astype(np.uint8)
print(filled)
