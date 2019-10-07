import scipy
import pip
import PIL
import matplotlib
import skimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import matplotlib.patches
import random
import copy
import itertools

from scipy.ndimage import gaussian_filter
from scipy import signal
from skimage import data, io
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from skimage.exposure import histogram
import math
from statistics import mean

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def hyst(x, th_lo, th_hi, initial = False):
    """
    x : Numpy Array
        Series to apply hysteresis to.
    th_lo : float or int
        Below this threshold the value of hyst will be False (0).
    th_hi : float or int
        Above this threshold the value of hyst will be True (1).
    """        

    if th_lo > th_hi: # If thresholds are reversed, x must be reversed as well
        x = x[::-1]
        th_lo, th_hi = th_hi, th_lo
        rev = True
    else:
        rev = False

    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi

    ind = np.nonzero(lo_or_hi)[0]  # Index für alle darunter oder darüber
    if not ind.size:  # prevent index error if ind is empty
        x_hyst = np.zeros_like(x, dtype=bool) | initial
    else:
        cnt = np.cumsum(lo_or_hi)  # from 0 to len(x)
        x_hyst = np.where(cnt, hi[ind[cnt-1]], initial)

    if rev:
        x_hyst = x_hyst[::-1]

    return x_hyst