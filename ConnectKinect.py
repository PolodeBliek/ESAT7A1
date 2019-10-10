from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2

color_image_shape = (1080, 1920, 4)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
pictureTaken = True

while pictureTaken:
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape(color_image_shape)

        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        color_flipped = cv2.flip(color_frame, 1)

        cv2.imwrite("C:/Users/olivi/untitled/kinectfoto.png", color_flipped)  # Save

        pictureTaken = False


# https://github.com/daan/calibrating-with-python-opencv/blob/02c90e4291adfb2426072f8f0837033754fc3a55/kinect-v2/color.py


