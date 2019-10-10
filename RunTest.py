import sys, time, cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class AcquisitionKinect():
    #Create a constructor to initialize different types of array and frame objects
    def __init__(self, resolution_mode=1.0):
        self.resolution_mode = resolution_mode

        self._done = False

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

        self._frameRGB = None
        self._frameDepth = None
        self._frameDepthQuantized = None
        self._frameSkeleton = None
        self.frameNum = 0

    def get_frame(self, frame):
        self.acquireFrame()
        frame.ts = int(round(time.time() * 1000))

        self.frameNum += 1

        frame.frameRGB = self._frameRGB
        frame.frameDepth = self._frameDepth
        frame.frameDepthQuantized = self._frameDepthQuantized

    #Get a color frame object
    def get_color_frame(self):
       self._frameRGB = self._kinect.get_last_color_frame()
       self._frameRGB = self._frameRGB.reshape((1080, 1920,-1)).astype(np.uint8)
       self._frameRGB = cv2.resize(self._frameRGB, (0,0), fx=1/self.resolution_mode, fy=1/self.resolution_mode)

    # Acquire the type of frame required
    def acquireFrame(self):
        if self._kinect.has_new_color_frame():
            self.get_color_frame()

    def close(self):
        self._kinect.close()
        self._frameDepth = None
        self._frameRGB = None


class Frame():
        frameRGB = None
        frameDepth = None
        frameDepthQuantized = None
        frame_num = 0


if __name__ == '__main__':

    kinect = AcquisitionKinect()
    frame = Frame()

    kinect.get_frame(frame)
    kinect.get_color_frame()
    image = kinect._frameRGB # Take image
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) # Convert image from RGBA to RGB
    cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/kinect_show.png", image) # Save image