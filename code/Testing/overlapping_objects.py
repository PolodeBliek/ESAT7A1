from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import numpy as np


currentDir             = os.path.dirname(os.path.abspath(__file__))
pd                     = 10     # distance between pixels
iv                     = 2     # interval distance / 2
md                     = 12    # minimum distance to count as new object
nb_overlaps_for_object = 5     # mimimum overlaps to count as an object
groundImage            = False
upDown                 = True
leftRight              = True
kinectCamera           = True
blur                   = False


def get_depth_frame():
    kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    noDepth = True
    depth_image_size = (424, 512)
    while noDepth:
        if kinect2.has_new_depth_frame():
            noDepth = False
            depth_frame = kinect2.get_last_depth_frame()
            depth_frame = depth_frame.reshape(depth_image_size)
            depth_frame_flipped = cv2.flip(depth_frame, 1)

            if groundImage:
                pickle.dump(depth_frame_flipped, open(currentDir + "/GroundDepth.pkl", "wb"))
            else:
                groundMatrix = pickle.load(open(currentDir + "/GroundDepth.pkl", "rb"))
                diff = np.subtract(groundMatrix.astype(np.int16), depth_frame_flipped.astype(np.int16))
                diff = abs(diff)
                diff = np.where(diff <= 5, 0, diff)
                diff = np.where(diff > 300, 0, diff)
                pickle.dump(diff, open(currentDir + "/DepthDiff.pkl", "wb"))
                return diff


def has_overlapping_objects(frame):
    nb_cols = int(len(frame)/pd)
    nb_rows = int(len(frame[0])/pd)
    nb_overlaps = 0

    if upDown:
        for col in range(0, nb_cols - 3):
            for row in range(0, nb_rows):
                value1 = frame[pd * col][pd * row]
                value2 = frame[pd * (col + 1)][pd * row]
                value3 = frame[pd * (col + 2)][pd * row]
                value4 = frame[pd * (col + 3)][pd * row]

                if value1 != 0 and value2 != 0 and value3 != 0 and value4 != 0:
                    if value1 - iv <= value2 <= value1 + iv:
                            if not (value1 - md <= value4 <= value1 + md):
                                if value3 - iv <= value4 <= value3 + iv:
                                    nb_overlaps += 1
                                    frame[pd * col][pd * row]       = 900
                                    frame[pd * (col + 1)][pd * row] = 900
                                    frame[pd * (col + 2)][pd * row] = 900
                                    frame[pd * (col + 3)][pd * row] = 900

    if leftRight:
        for col in range(0, len(frame)):
            for row in range(0, nb_rows - 3):
                value1 = frame[col][pd * row]
                value2 = frame[col][pd * (row + 1)]
                value3 = frame[col][pd * (row + 2)]
                value4 = frame[col][pd * (row + 3)]

                if value1 != 0 and value2 != 0 and value3 != 0 and value4 != 0:
                    if value1 - iv <= value2 <= value1 + iv:
                        if not (value1 - md <= value4 <= value1 + md):
                            if value3 - iv <= value4 <= value3 + iv:
                                nb_overlaps += 1
                                frame[col][pd * row]       = 450
                                frame[col][pd * (row + 1)] = 450
                                frame[col][pd * (row + 2)] = 450
                                frame[col][pd * (row + 3)] = 450

    plt.imshow(frame)
    plt.show()
    print(nb_overlaps)
    return nb_overlaps


def main():
    if kinectCamera:
        depth_frame = get_depth_frame()
    else:
        depth_frame = pickle.load(open(currentDir + "/DepthDiff.pkl", "rb"))

    if blur:
        depth_frame = cv2.medianBlur(depth_frame, 7)

    if not groundImage:
        if has_overlapping_objects(depth_frame) >= nb_overlaps_for_object:
            print('OVERLAP')
        else:
            print('NO OVERLAP')


main()




