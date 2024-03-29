import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from math import *
import matplotlib
import numpy as np
import platform


# Check wich platform it is
isWin = True if platform.system() == 'Windows' else False

"""
 boven = coord[i][0]
 links = coord[i][1]
 rechts = coord[i][2]
 onder = coord[i][3]

"""


def hoekpunten_vinden(matrix_anneloes):
    number_of_elements = matrix_anneloes.max()
    allcoord = []

    for index in range(1, number_of_elements + 1):
        coord = np.where(matrix_anneloes == index)
        minx = min(coord[0])
        maxx = max(coord[0])
        miny = min(coord[1])
        maxy = max(coord[1])

        listCoord = list(zip(coord[0], coord[1]))
        Links = [x for x in listCoord if x[0] == minx]
        Rechts = [x for x in listCoord if x[0] == maxx]
        Boven = [x for x in listCoord if x[1] == miny]
        Onder = [x for x in listCoord if x[1] == maxy]

        allcoord.append([Links[0], Onder[0], Boven[0], Rechts[0]])

    return allcoord


######################
# originele foto om afstanden op aan te duiden
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "") if isWin else os.path.dirname(
    os.path.abspath(__file__)).replace("code\\Testing", "").replace("\\", "/")
im = Image.open(currentDir + "testImages\\kinectColor\\kinectfoto.png") if isWin else Image.open(
    currentDir + "testImages/kinectColor/kinectfoto.png")

draw = ImageDraw.Draw(im)

# matrix van annelies na bewerkingen
matrix_anneloes = pickle.load(open("Dag_lieve_schat.pkl", "rb"))

coord = hoekpunten_vinden(matrix_anneloes)

lijst = [0,1,9,11]

for i in lijst:
    rechts_boven = (coord[i][1][1], coord[i][0][0])
    links_boven = (coord[i][2][1], coord[i][0][0])
    rechts_onder = (coord[i][1][1], coord[i][3][0])
    links_onder = (coord[i][2][1], coord[i][3][0])

    draw.line([rechts_boven, links_boven, links_onder, rechts_onder, rechts_boven], fill=(0, 0, 225), width=5)

rechts_boven = (coord[8][1][1], coord[7][0][0])
links_boven = (coord[8][2][1], coord[7][0][0])
rechts_onder = (coord[8][1][1], coord[8][3][0])
links_onder = (coord[8][2][1], coord[8][3][0])

draw.line([rechts_boven, links_boven, links_onder, rechts_onder, rechts_boven], fill=(0, 0, 225), width=5)
im = np.array(im)
plt.imsave("C:/Users/olivi/untitled/image_boxes.png", im, cmap='gray', format = 'png')
#im.show()


