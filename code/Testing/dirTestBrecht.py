import matplotlib
from matplotlib.pyplot import text

"""
binnnekrijgten:

- bovenpixel met (x-coo, y-coo)
- onderpixel '''

- meest linkse pixel(coo)
"""
def distance_between_pixels_in_pixels(pixel1,pixel2):
    x_distance = abs(pixel2[0]-pixel1[0])
    y_distance = abs(pixel2[1]-pixel1[1])


    direct_distance = sqrt(x_distance^2 + y_distance^2)
    return direct_distance

def pixel_length_to_real_length(pixellenght):
#1 cm int echt is 100?? pixels op de foto(!! op 1,5m hoog)

    return format(pixellenght/100 ,"12.1f") #afronding van echte


def line_between_outer_points(outer_point1,outer_point2):
    return plt.plot(outer_point1, outer_point2, 'ro-')

def midden_lijn(pixel1,pixel2):
    x_midden = (pixel1[0]+pixel2[0])/2
    y_midden = (pixel1[1]+pixel2[1])/2

    return x_midden,y_midden

def afstand_aan_lijn_plaatsen(pixel1,pixel2):
    text(midden_lijn(pixel1,pixel2)[0],midden_lijn(pixel1,pixel2)[1], str(pixel_length_to_real_length(direct_distance)), rotation=0, verticalalignment='center')

from PIL import Image, ImageDraw
im = Image.open("grid.png")
d = Image.Draw.Draw(im)

top = (150,50)
left = (100,125)
right = (200,125)

line_color = (0,0,225)

d.line([top,left,right,top], fill = line_color,width = 2)

im.save("drawn_grid.png")