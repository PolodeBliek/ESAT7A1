import numpy as np
from scipy import ndimage

a1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def fill_holes(array, struct):
    print(f"*input array = \n{array}")

    filled = ndimage.binary_fill_holes(a1, structure=struct)
    filled = np.where(filled, 1, 0)
    print(f"*filled = \n{filled}")


struct1 = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0]])

struct2 = np.array([[1]])

struct3 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])
print(struct3)

fill_holes(a1, struct1)
