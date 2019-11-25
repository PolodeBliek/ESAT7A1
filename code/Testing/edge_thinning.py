import numpy as np
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image
from tkinter import filedialog
from Main.whole_process2 import grayscale, gaussian_blur, detect_objects
import time

timed = True
th_low = 0


def sobel(image):

    image = image.astype(np.int32)  # heel belangrijk, anders doet convolve vreemde dingen
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

    # initialize new images
    Gh = ndimage.convolve(image, horizontal)
    Gv = ndimage.convolve(image, vertical)
    # print(f"* the horizontal gradient is: \n{Gh}\n")
    # print(f"* the vertical gradient is: \n{Gv}\n")

    # this needs to be done here because later on GH and GV change
    theta_rad = np.arctan2(Gv, Gh)
    theta_deg = theta_rad * 180 /math.pi
    theta_deg = theta_deg.astype(np.int32)
    # print(f"* the angles are: \n{theta_deg}\n")

    Magnitude = np.sqrt(np.square(Gh) + np.square(Gv))
    if np.amax(Magnitude) > 255:
        Magnitude = np.interp(Magnitude, (Magnitude.min(), Magnitude.max()), (0, 255))
    Magnitude = Magnitude.astype(np.uint8)  # reconvert range to (0, 255)

    return Magnitude, theta_deg


def thin_edges(magnitude, angle, low):
    """
    thin the edges and delete the values below the low threshold so that it doesn't need to happen in hysteresis anymore
    (or don't delete the low ones, doesn't really matter when it happens)
    """
    # define footprints for the angle cases (1, 2, 3 and 4)
    t1 = time.time()
    f1 = np.array([[1, 0, 1]])
    f2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])  # hier had ik f4 verwacht, ma de hoeken kloppen niet
    f3 = np.array([[1], [0], [1]])
    f4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])  # hier had ik f2 verwacht
    # define conditions
    t2 = time.time()
    cond1 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f1, mode='constant', cval=-np.inf)  # True/False matrix
    cond2 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f2, mode='constant', cval=-np.inf)  # True/False matrix
    cond3 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f3, mode='constant', cval=-np.inf)  # True/False matrix
    cond4 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f4, mode='constant', cval=-np.inf)  # True/False matrixcond1 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f1, mode='constant', cval=-np.inf)  # True/False matrix
    t3 = time.time()
    pos_ang = np.where(angle < 0, angle + 180, angle)  # make the negative angles positive, works for this application
    t4 = time.time()
    # transform the angle matrix to a matrix of 1/0, indicating wether the element is the highest along its gradient
    ang_to_bool = np.where(pos_ang <= 22.5, cond1, np.where(pos_ang <= 67.5, cond2, np.where(pos_ang <= 112.5, cond3, np.where(
                    pos_ang <= 157.5, cond4, np.where(pos_ang > 157.5, cond1, pos_ang)))))
    t5 = time.time()
    filtered = np.where(magnitude > low, ang_to_bool, 0)  # keep only the bools of the values higher than the low th
    t6 = time.time()
    remasked = np.where(filtered, magnitude, 0)  # og waardes er weer over trekken
    t7 = time.time()

    global timed
    if timed:
        print("EDGE THINNING:")
        print(f"Defining footprints:      {t2-t1}s")
        print(f"Defining conditions:      {t3-t2}s")
        print(f"Calculating pos_ang:      {t4-t3}s")
        print(f"Calculating ang_to_bool:  {t5-t4}s")
        print(f"Filtering:                {t6-t5}s")
        print(f"Remasking:                {t7-t6}s")
        print("-----------------------------------")
        print(f"TOTAL TIME:               {t7-t1}s\n")

    return remasked


def hysteresis(image, low, high):
    """
    :return: a matrix consisting of 0s and 1s indicating the edges after thresholding
    """
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low  # seperate 'weak/strong' values (mask_low[*,*]==True) from 'noise' values (=False)
    mask_high = image >= high  # seperate 'strong' values (=True) from 'weak/noise' values (=False)
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low, structure)  # detecting and naming islands, and also returns nb of islands (making a map of the islands)
    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))  # list of how many high's each island has
    connected_to_high = sums > 0  # seperate islands with at least one 'high' (=True) from islands without highs(=False)
    thresholded = connected_to_high[labels_low]  # special np thing, projecs the values of 'connected_to_high' onto the island of which the name (1,2,3...) is the same as the index of 'connected to high'
    # transform from True/False to 1/0
    thresholded = thresholded.astype(np.uint8)

    return thresholded


# def thin_edgess(magnitude, angle, low):
#     # define footprints for the angle cases (1, 2, 3 and 4)
#     f1 = np.array([[1, 0, 1]])
#     f2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
#     f3 = np.array([[1], [0], [1]])
#     f4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
#     # define conditions
#     t3 = time.time()
#     cond1 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f1, mode='constant', cval=-np.inf)  # True/False
#     cond2 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f2, mode='constant', cval=-np.inf)  # True/False
#     cond3 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f3, mode='constant', cval=-np.inf)  # True/False
#     cond4 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f4, mode='constant', cval=-np.inf)  # True/False
#     t4 = time.time()
#     # print(t4-t3)
#
#     ang_pos = np.where(angle < 0, angle + 180, angle)  # make the negative angles positive, works for this application
#     ang_pos_ = ang_pos.copy()  # make the negative angles positive, works for this application
#     t1 = time.time()
#     # ja kijk dit verandert ang_pos en da es nie hoed e, want je gebruikt dat nog als referentie
#     # en nu we dat niet meer doen, gaat het zelfs trager dan de eerste versie
#     np.putmask(ang_pos_, ang_pos > 157.5, cond1)
#     np.putmask(ang_pos_, ang_pos <= 157.5, cond4)
#     np.putmask(ang_pos_, ang_pos <= 112.5, cond3)
#     np.putmask(ang_pos_, ang_pos <= 67.5, cond2)
#     np.putmask(ang_pos_, ang_pos <= 22.5, cond1)
#     t2 = time.time()
#     print(t2-t1)
#
#     filtered = np.where(magnitude > low, ang_pos_, 0)  # keep only the angles of the values higher than the low th
#     remasked = np.where(filtered, magnitude, 0)  # og waardes er weer over trekken
#
#     return remasked



# def thin_edgess(magnitude, angle, low):
#     # print(f"*magnitude:\n{magnitude}\n*angle:\n{angle}")
#     # define footprints for the angle cases (1, 2, 3 and 4)
#     f1 = np.array([[1, 0, 1]])
#     f2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])  # hier had ik f4 verwacht, ma de hoeken kloppen niet
#     f3 = np.array([[1], [0], [1]])
#     f4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])  # hier had ik f2 verwacht
#     # define conditions
#     cond1 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f1, mode='constant', cval=-np.inf)  # True/False
#     cond2 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f2, mode='constant', cval=-np.inf)  # True/False
#     cond3 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f3, mode='constant', cval=-np.inf)  # True/False
#     cond4 = magnitude >= ndimage.maximum_filter(magnitude, footprint=f4, mode='constant', cval=-np.inf)  # True/False
#
#     filter_zero_mag = np.where(magnitude == 0, 181, angle)
#     # print(f"*filter_zero_mag:\n{filter_zero_mag}")
#     ang_pos = np.where(filter_zero_mag < 0, filter_zero_mag + 180, filter_zero_mag)  # make the negative angles positive, works for this application
#     # print(f"*ang_pos: \n{ang_pos}")
#     # ang_to_num = np.where(ang_pos <= 22.5, 1, np.where(ang_pos <= 67.5, 2, np.where(ang_pos <= 112.5, 3, np.where(
#     #                 ang_pos <= 157.5, 4, np.where(ang_pos > 157.5, 1, ang_pos)))))
#     # filter = np.where(magnitude > low, ang_to_num, 0)  # keep only the angles of the values higher than the low th
#
#     ang_to_num = np.where(ang_pos <= 22.5, cond1, np.where(ang_pos <= 67.5, cond2, np.where(ang_pos <= 112.5, cond3, np.where(
#                     ang_pos <= 157.5, cond4, np.where(ang_pos > 157.5, cond1, np.where(ang_pos >180, 0, ang_pos))))))
#     # filtered = np.where(magnitude > low, ang_to_num, 0)  # keep only the angles of the values higher than the low t
#     # print(f"*ang_to_num: \n{ang_to_num}")
#     remasked = np.where(ang_to_num, magnitude, 0)  # og waardes er weer over trekken
#     # print(f"*remasked: \n{remasked}")
#
#     return remasked


# a1 = np.array([[1, 0, 120, 250, 255],
#                [0, 0, 120, 250, 255],
#                [0, 0, 120, 250, 255],
#                [0, 0, 120, 250, 255],
#                [0, 0, 120, 250, 255]])

# a1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# a1 = np.array([[0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 1],
#                [0, 0, 0, 1, 1],
#                [0, 0, 1, 1, 1],
#                [0, 1, 1, 1, 1]])

# mag, ang = sobel(a1)
# thin_edgess(mag, ang, 60)


def sobel_en_thin():
    global th_low
    structure = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1]])
    filepath = filedialog.askopenfilename()
    image = np.array(Image.open(filepath))
    image = grayscale(image)
    image = gaussian_blur(image, 1)
    m, t = sobel(image)
    thin = thin_edges(m, t, th_low)
    t0 = time.time()
    hyst1 = hysteresis(m, 10, 80)
    t1 = time.time()
    hyst_thin = hysteresis(thin, 10, 80)
    t2 = time.time()
    filled1 = ndimage.binary_fill_holes(hyst1)
    t3 = time.time()
    filled_thin = ndimage.binary_fill_holes(hyst_thin)
    t4 = time.time()
    obj = detect_objects(filled1)
    t5 = time.time()
    obj_thin = detect_objects(filled_thin)
    t6 = time.time()
    print(f"hyst:           {t1-t0}")
    print(f"hyst_thinned:   {t2-t1}")
    print(f"filled:         {t3-t2}")
    print(f"filled_thinned: {t4-t3}")
    print(f"db:             {t5-t4}")
    print(f"db_thinned:     {t6-t5}")
    f = plt.figure()
    f.suptitle("zoom in als ge de lijnen wilt zien")
    f.add_subplot(2,4,1)
    plt.imshow(m, cmap='gray')
    plt.title('sobel')
    f.add_subplot(2,4,5)
    plt.imshow(thin, cmap='gray')
    plt.title('thinned edges')
    f.add_subplot(2,4,2)
    plt.imshow(hyst1, cmap='gray')
    plt.title('hysteresis')
    f.add_subplot(2,4,6)
    plt.imshow(hyst_thin, cmap='gray')
    plt.title('hysteresis (thinned)')
    f.add_subplot(2,4,3)
    plt.imshow(filled1, cmap='gray')
    plt.title('filled')
    f.add_subplot(2,4,7)
    plt.imshow(filled_thin, cmap='gray')
    plt.title('filled (thinned)')
    f.add_subplot(2,4,4)
    plt.imshow(obj, cmap='gray')
    plt.title('db')
    f.add_subplot(2,4,8)
    plt.imshow(obj_thin, cmap='gray')
    plt.title('db (thinned)')
    plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    b1 = tk.Button(root, text='thin edges', command=lambda: sobel_en_thin())
    b1.pack()
    root.mainloop()

