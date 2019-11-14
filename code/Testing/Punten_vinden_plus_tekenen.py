import os
import _osx_support
import pickle
from PIL import Image, ImageDraw, ImageFont
from math import *
import matplotlib

"""
 boven = coord[i][0]
 links = coord[i][1]
 rechts = coord[i][2]
 onder = coord[i][3]
"""


def hoekpunten_vinden(matrix_anneloes):

    number_of_elements = matrix_anneloes.max()

    coord = [[0, ] * 4 for k in range(0, number_of_elements)]

    for i in range(1,number_of_elements+1):

        for rij in range(len(matrix_anneloes)):
            for kolom in range(len(matrix_anneloes[0])):
                if matrix_anneloes[rij][kolom] == i:
                    if coord[i-1][0] == 0 or (coord[i-1][0])[0] > rij:
                        coord[i-1][0] = (rij,kolom)

                    if coord[i-1][1] == 0 or (coord[i-1][1])[1] < kolom:
                        coord[i-1][1] = (rij,kolom)

                    if coord[i-1][2] == 0 or (coord[i-1][2])[1] > kolom:
                        coord[i-1][2] = (rij, kolom)

                    if coord[i-1][3] == 0 or (coord[i-1][3])[0] < rij:
                        coord[i-1][3] = (rij, kolom)
    return coord


def distance_between_pixels_in_pixels(pixel1,pixel2):
    x_distance = abs(pixel2[0]-pixel1[0])
    y_distance = abs(pixel2[1]-pixel1[1])

    direct_distance = sqrt(x_distance**2 + y_distance**2)

    return direct_distance

def pixel_length_to_real_length(pixellength):
    return ("%.1f" % (pixellength/21.0)) #afronding van echte

def midden_lijn(pixel1,pixel2):
    x_midden = (pixel1[0]+pixel2[0])/2
    y_midden = (pixel1[1]+pixel2[1])/2

    return x_midden,y_midden


def draw_line(pixel1,pixel2):
    fnt = ImageFont.truetype("arial.ttf", 30)
    draw.line([pixel1, pixel2], fill=(0, 0, 225), width=5)
    draw.text((midden_lijn(pixel1, pixel2)[0], midden_lijn(pixel1, pixel2)[1]),
              str(pixel_length_to_real_length(distance_between_pixels_in_pixels(pixel1, pixel2))),
              font=fnt, fill=(0, 0, 0))


######################
#originele foto om afstanden op aan te duiden
currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "")
im = Image.open(currentDir + "testImages\\kinectColor\\kinectfoto.png")

draw = ImageDraw.Draw(im)


#matrix van annelies na bewerkingen
matrix_anneloes = pickle.load(open(currentDir + "code\\Testing\\" + "Dag_lieve_schat.pkl","rb"))

coord = hoekpunten_vinden(matrix_anneloes)

for i in range(0,len(coord)):
    #elke i is een object
    pixel_coo_boven = (coord[i][0][1],coord[i][0][0])
    pixel_coo_links = (coord[i][1][1],coord[i][1][0])
    pixel_coo_rechts = (coord[i][2][1],coord[i][2][0])

    #pixel_onder = coord[i][3]
    draw_line(pixel_coo_boven,pixel_coo_links)
    draw_line(pixel_coo_boven,pixel_coo_rechts)

im.show()
