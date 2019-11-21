import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
from math import *
import matplotlib
import numpy as np
import platform
import time

t0 = time.time()
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
t1 = time.time()
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "") if isWin else os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "").replace("\\", "/")
im = Image.open(currentDir + "testImages\\kinectColor\\kinectfoto.png") if isWin else Image.open(currentDir + "testImages/kinectColor/kinectfoto.png")

t2 = time.time()
draw = ImageDraw.Draw(im)

# matrix van annelies na bewerkingen
matrix_anneloes = pickle.load(open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\code\\Testing\\Dag_lieve_schat.pkl", "rb"))
t3 = time.time()
coord = hoekpunten_vinden(matrix_anneloes)
t4 = time.time()
for i in range(0, len(coord)):

          rechts_boven = (coord[i][1][1], coord[i][0][0])
          links_boven = (coord[i][2][1], coord[i][0][0])
          rechts_onder = (coord[i][1][1], coord[i][3][0])
          links_onder = (coord[i][2][1], coord[i][3][0])

          draw.line([rechts_boven,links_boven,links_onder,rechts_onder,rechts_boven], fill=(0, 0, 225), width=5)
t5 = time.time()
print("TIME")
print("##################")
print("0->1     ", t1-t0)
print("1->2     ", t2-t1)
print("2->3     ", t3-t2)
print("3->4     ", t4-t3)
print("4->5     ", t5-t4)

#im.show()
