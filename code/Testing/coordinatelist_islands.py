import numpy as np
from scipy import ndimage

a1 = np.array([[1, 1, 0, 0, 0, 0, 0],
               [1, 1, 0, 1, 1, 1, 0],
               [1, 1, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])


def make_coord_list(boolarr):
    islands, nb_islands = ndimage.label(boolarr)
    coords = []
    # coords = np.array([])
    for label in range(1, nb_islands+1):
        island_coords = np.transpose(np.nonzero(islands == label))  # get the indices of the ones
        # np.append(coords, island_coords)
        coords.append(island_coords)



