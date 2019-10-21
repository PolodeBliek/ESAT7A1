import numpy as np
from PIL import Image
from hulpfunctie_sobel import *
import time
from scipy import ndimage

files = ["1_rechthoeken.png", "2_rechthoeken_dicht_op_elkaar.png", "2objecten_1lichtbron.PNG", "3_willekeurige_vormen.png", "4_rechthoeken_in_kleur.png", "5_rechthoeken_niet_opgevuld.png", "6_rechthoeken_deels_opgevuld.png", "7_rechthoeken_in_kleur_deels_opgevuld.png", "8_vormen_vaag_met_schaduw.png", "9_vormen_gekleurd.png", "10_vormen_opgedeeld.png", "11_zwart.png"]
t_classic = 0
t_new = 0
for file in files:
    print(file)
    name_image = file
    img = np.array(Image.open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\testImages\\" + name_image))#.astype(np.uint8)
    img = grayscale(img)
    h, w = img.shape
    GaussianKernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    t4 = time.time()
    img4 = img/4
    img8 = img4/2
    img16 = img8/2
    t5 = time.time()
    NewImage = np.zeros((h, w))
    NewImage2 = np.zeros((h, w))
    t0 = time.time()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            verticalGrad = (GaussianKernel[0, 0] * img[i - 1, j - 1]) + \
                           (GaussianKernel[0, 1] * img[i - 1, j]) + \
                           (GaussianKernel[0, 2] * img[i - 1, j + 1]) + \
                           (GaussianKernel[1, 0] * img[i, j - 1]) + \
                           (GaussianKernel[1, 1] * img[i, j]) + \
                           (GaussianKernel[1, 2] * img[i, j + 1]) + \
                           (GaussianKernel[2, 0] * img[i + 1, j - 1]) + \
                           (GaussianKernel[2, 1] * img[i + 1, j]) + \
                           (GaussianKernel[2, 2] * img[i + 1, j + 1])
            NewImage[i - 1, j - 1] = abs(verticalGrad)
    t1 = time.time()
    NewImage2 = ndimage.convolve(img, GaussianKernel)
    t2 = time.time()
    print(NewImage == NewImage2)
    t_classic += t1-t0
    t_new += t2-t1 + t5-t4
    print("CLASSIC METHOD: ", t1-t0)
    print("NEW METHOD:     ", t2-t1 + t5-t4)
    print("========================")

print("CLASSIC METHOD, TOTAL:", t_classic)
print("NEW METHOD, TOTAL:    ", t_new)
