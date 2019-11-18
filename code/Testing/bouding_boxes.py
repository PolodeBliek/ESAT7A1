import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
from math import *
import matplotlib
import numpy
import platform

#Check wich platform it is
isWin = True if platform.system() == 'Windows' else False

"""
 boven = coord[i][0]
 links = coord[i][1]
 rechts = coord[i][2]
 onder = coord[i][3]

"""
def hoekpunten_vinden(matrix_anneloes):
    number_of_elements = matrix_anneloes.max()

    coord = [[0, ] * 4 for k in range(0, number_of_elements)]

    for i in range(1, number_of_elements + 1):

        for rij in range(len(matrix_anneloes)):
            for kolom in range(len(matrix_anneloes[0])):
                if matrix_anneloes[rij][kolom] == i:

                    if coord[i - 1][0] == 0 or (coord[i - 1][0])[0] > rij:
                        coord[i - 1][0] = (rij, kolom)

                    if coord[i - 1][1] == 0 or (coord[i - 1][1])[1] < kolom:
                        coord[i - 1][1] = (rij, kolom)

                    if coord[i - 1][2] == 0 or (coord[i - 1][2])[1] > kolom:
                        coord[i - 1][2] = (rij, kolom)

                    if coord[i - 1][3] == 0 or (coord[i - 1][3])[0] < rij:
                        coord[i - 1][3] = (rij, kolom)
    return coord

######################
# originele foto om afstanden op aan te duiden
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "") if isWin else os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "").replace("\\", "/")
im = Image.open(currentDir + "testImages\\kinectColor\\kinectfoto.png") if isWin else Image.open(currentDir + "testImages/kinectColor/kinectfoto.png")

draw = ImageDraw.Draw(im)

# matrix van annelies na bewerkingen
matrix_anneloes = pickle.load(open("Dag_lieve_schat.pkl", "rb"))

coord = hoekpunten_vinden(matrix_anneloes)

for i in range(0, len(coord)):

          rechts_boven = (coord[i][1][1], coord[i][0][0])
          links_boven = (coord[i][2][1], coord[i][0][0])
          rechts_onder = (coord[i][1][1], coord[i][3][0])
          links_onder = (coord[i][2][1], coord[i][3][0])

          draw.line([rechts_boven,links_boven,links_onder,rechts_onder,rechts_boven], fill=(0, 0, 225), width=5)

im.show()
