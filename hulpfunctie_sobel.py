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

from collections.abc import Sequence
from itertools import chain, count

def gaussian(img):
    h, w = img.shape
    GaussianKernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    newImg = np.zeros((h,w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gaussianGrad = (GaussianKernel[0, 0] * img[i - 1, j - 1]) + \
                           (GaussianKernel[0, 1] * img[i - 1, j]) + \
                           (GaussianKernel[0, 2] * img[i - 1, j + 1]) + \
                           (GaussianKernel[1, 0] * img[i, j - 1]) + \
                           (GaussianKernel[1, 1] * img[i, j]) + \
                           (GaussianKernel[1, 2] * img[i, j + 1]) + \
                           (GaussianKernel[2, 0] * img[i + 1, j - 1]) + \
                           (GaussianKernel[2, 1] * img[i + 1, j]) + \
                           (GaussianKernel[2, 2] * img[i + 1, j + 1])
            newImg[i - 1, j - 1] = abs(gaussianGrad)
    return newImg

def grayscale(image):
    return (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)

def gem_kleur_van_pixels(picture):
    gem_kleur_van_pixel = []

    image = np.array(Image.open(picture)).astype(np.uint8)

    for eachRow in image:
        for eachPix in eachRow:
            avgColor = mean(eachPix[:3])  # eerste 3 getallen vd array die de kleur geven
            gem_kleur_van_pixel.append(avgColor)

    return gem_kleur_van_pixel

def iar_reconverted(iar_boolToNum,height,width):

    iar_reconverted = []
    index = 0
    for i in range(height):                        #RANGES HANGEN AF VAN DE GROOTTE VAN DE IMAGE
        iar_reconverted.append([])
        for k in range(width):
            j = iar_boolToNum[index]
            iar_reconverted[i].append([j, j, j])
            index += 1
    return iar_reconverted




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
