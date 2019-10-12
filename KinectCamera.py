from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2

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

        cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/kinectPicture.png", color_flipped)  # Save

depth_image_size = (424, 512)

kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
noDepth = True

while noDepth:
    if kinect2.has_new_depth_frame():
        depth_frame = kinect2.get_last_depth_frame()
        noDepth = False

        depth_frame = depth_frame.reshape(depth_image_size)
        depth_frame = depth_frame * (256.0 / np.amax(depth_frame))
        colorized_frame = cv2.applyColorMap(np.uint8(depth_frame), cv2.COLORMAP_JET)
        #cv2.imshow('depth', colorized_frame)
        cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/kinectDepthPicture.png", colorized_frame)  # Save






# https://github.com/daan/calibrating-with-python-opencv/blob/02c90e4291adfb2426072f8f0837033754fc3a55/kinect-v2/color.py