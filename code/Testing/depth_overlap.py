# works for max 1 overlap per 'level' of height on each object, if you can resize the depth img to fit the color img
import numpy as np

# color = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                   [1, 1, 0, 2, 2, 2, 2, 0, 0, 0],
#                   [1, 1, 0, 2, 2, 2, 2, 0, 0, 0],
#                   [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
#                   [0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
# depth = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                   [1, 1, 0, 1, 2, 2, 1, 0, 0, 0],
#                   [1, 1, 0, 1, 2, 2, 1, 0, 0, 0],
#                   [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

color = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 2, 2, 0, 0, 0, 0],
                  [1, 1, 0, 2, 2, 2, 2, 0, 0, 0],
                  [1, 1, 0, 2, 2, 2, 2, 0, 0, 0],
                  [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                  [0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

depth = np.array([[1, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                  [1, 2, 2, 0, 2, 2, 0, 0, 0, 0],
                  [1, 1, 0, 1, 2, 2, 1, 0, 0, 0],
                  [1, 1, 0, 1, 3, 2, 1, 0, 0, 0],
                  [0, 0, 0, 0, 3, 2, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 2, 2, 1],
                  [0, 0, 0, 0, 0, 0, 1, 2, 2, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# color = np.array([[0, 1, 1, 0],
#                   [1, 1, 1, 1],
#                   [1, 1, 1, 1],
#                   [0, 1, 1, 0]])
# depth = np.array([[0, 2, 2, 0],
#                   [1, 2, 2, 1],
#                   [1, 2, 2, 1],
#                   [0, 2, 2, 0]])


def overlap(colr, deph):
    nb_colr_obj = np.amax(colr)
    # nb_deph_obj = np.amax(deph)
    deph = deph * nb_colr_obj  # * nb_deph_obj
    cnd = colr + deph
    u, indices = np.unique(cnd, return_index=True)
    nb_objects = len(u) - 1  # zero is also in u and is not an object
    dict1 = dict(zip(u, range(nb_objects+1)))
    reranged = np.vectorize(dict1.__getitem__)(cnd)  # map to range (1, 2, ..., n) where n is the nb of objects

    print(cnd)
    print(u)
    print(reranged)
    print(f"{nb_objects} different objects detected")


overlap(color, depth)
