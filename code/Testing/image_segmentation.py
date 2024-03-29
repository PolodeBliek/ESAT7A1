import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from PIL import Image
import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
from math import *
import matplotlib
import numpy as np
import platform
from skimage import io

isWin = True if platform.system() == 'Windows' else False
#https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980


currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "") if isWin else os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "").replace("\\", "/")
image = mpimg.imread(currentDir + "testImages\\kinectColor\\kinectfoto.png") if isWin else Image.open(currentDir + "testImages/kinectColor/kinectfoto.png")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

image_gray = rgb2gray(image)

plt.imshow(image_gray, cmap = plt.get_cmap('gray'))

#plt.show()


def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)  # polar co-ordinates
    r = center[0] + radius * np.sin(radians)

    return np.array([c, r]).T


# Exclude last point because a closed path should not have duplicate points
points = circle_points(200, [250, 520], 180)[:-1]

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

fig, ax = image_show(image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

snake = seg.active_contour(image_gray, points,alpha=0.06,beta=0.3)
fig, ax = image_show(image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

plt.show()