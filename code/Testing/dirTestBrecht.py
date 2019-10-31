"""
binnnekrijgen:

- bovenpixel met (x-coo, y-coo)
- onderpixel '''

- meest linkse pixel(coo)
"""

from PIL import Image, ImageDraw, ImageFont
from math import *

im = Image.open("kinectfoto.png")
draw = ImageDraw.Draw(im)

pixel1 = (377,277)
pixel2 = (610,160)

def distance_between_pixels_in_pixels(pixel1,pixel2):
    x_distance = abs(pixel2[0]-pixel1[0])
    y_distance = abs(pixel2[1]-pixel1[1])

    direct_distance = sqrt(x_distance^2 + y_distance^2)
    return direct_distance

def pixel_length_to_real_length(pixellenght):
#1 cm int echt is 100?? pixels op de foto(!! op 1,5m hoog)

    return format(pixellenght/15.5 ,"12.1f") #afronding van echte

def midden_lijn(pixel1,pixel2):
    x_midden = (pixel1[0]+pixel2[0])/2
    y_midden = (pixel1[1]+pixel2[1])/2

    return x_midden,y_midden

fnt = ImageFont.truetype("arial.ttf", 30)

draw.line([pixel1,pixel2], fill = (0,0,225),width = 5)

draw.text((midden_lijn(pixel1,pixel2)[0],midden_lijn(pixel1,pixel2)[1]),
          str(pixel_length_to_real_length(distance_between_pixels_in_pixels(pixel1,pixel2))),
          font =  fnt,fill = (0,0,0))

im.show()

