import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
from math import *
import matplotlib
import numpy as np
import platform
import time

#Check wich platform it is
isWin = True if platform.system() == 'Windows' else False


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

def hoekpunten_vinden_V2(matrix_anneloes):
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

# originele foto om afstanden op aan te duiden
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "") if isWin else os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "").replace("\\", "/")
im = Image.open(currentDir + "testImages\\kinectColor\\kinectfoto.png") if isWin else Image.open(currentDir + "/testImages/kinectColor/kinectfoto.png")

draw = ImageDraw.Draw(im)

# matrix van annelies na bewerkingen
matrix_anneloes = pickle.load(open(currentDir + "code\\Testing\\Dag_lieve_schat.pkl", "rb"))

t0 = time.time()
coord = hoekpunten_vinden(matrix_anneloes)
t1 = time.time()
coord2 = hoekpunten_vinden_V2(matrix_anneloes)
t2 = time.time()
print("TIME")
print("OLD METHOD:  ", t1-t0)
print("NEW METHOD:  ", t2-t1)
