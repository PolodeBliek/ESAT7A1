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

def iar_reconverted(iar_boolToNum):

    iar_reconverted = []
    index = 0
    for i in range(480):
        iar_reconverted.append([])
        for k in range(640):
            j = iar_boolToNum[index]
            iar_reconverted[i].append([j, j, j])
            index += 1
    return iar_reconverted