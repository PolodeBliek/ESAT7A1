from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import os

### IMAGE PROCESSING ###

dir = "C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images"
a = os.listdir("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images")
for element in a:
    os.remove(dir + '/' + str(element))


def get_color_frame():

    color_image_shape = (1080, 1920,4)

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    noPicture = True

    while noPicture:
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            noPicture = False

            color_frame = color_frame.reshape(color_image_shape)
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            color_flipped = cv2.flip(color_frame, 1)
            cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectColorPicture.png", color_flipped)  #

def get_depth_frame():

    kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    noDepth = True
    depth_image_size = (424, 512)

    while noDepth:
        if kinect2.has_new_depth_frame():
            noDepth = False
            depth_frame = kinect2.get_last_depth_frame()
            depth_frame_vis = depth_frame.reshape(depth_image_size)

            filtered_depth_frame = np.where(depth_frame > 696, 0, depth_frame)
            filtered_depth_frame2 = np.where(filtered_depth_frame < 1, 0, filtered_depth_frame)
            depth_frame3 = filtered_depth_frame2.reshape(depth_image_size)
            depth_frame_flipped = cv2.flip(depth_frame3, 1)

            plt.imshow(depth_frame_flipped)
            plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPicture.png")

            for ran in range(620, 720, 2):
                depth_copy = copy.deepcopy(depth_frame)

                filtered_depth_frame = np.where(depth_copy > ran + 5, 0, depth_copy)
                filtered_depth_frame2 = np.where(filtered_depth_frame < ran, 0, filtered_depth_frame)
                depth_frame2 = filtered_depth_frame2.reshape(depth_image_size)
                depth_frame_flipped = cv2.flip(depth_frame2, 1)

                plt.imshow(depth_frame_flipped)
                plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPicture" + str(ran) + ".png")


get_color_frame()
get_depth_frame()




