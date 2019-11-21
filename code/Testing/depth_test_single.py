from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import numpy as np
import copy

dir = "C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images"
a = os.listdir("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images")
for element in a:
    os.remove(dir + '/' + str(element))

### IMAGE PROCESSING ###
groundImage = False
layers = 0
currentDir  = os.path.dirname(os.path.abspath(__file__))

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
            pickle.dump(color_flipped, open(currentDir + "/ColorImage.pkl", "wb"))
            cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectColorPicture.png", color_flipped)  #

def get_depth_frame():

    kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    noDepth = True
    depth_image_size = (424, 512)
    while noDepth:
        if kinect2.has_new_depth_frame():
            noDepth = False
            depth_framee = kinect2.get_last_depth_frame()
            depth_framea = kinect2.get_last_depth_frame()
            depth_frameZ = kinect2.get_last_depth_frame()
            depth_frame = (depth_framea + depth_framee + depth_frameZ) / 3
            depth_frame = depth_framee.reshape(depth_image_size)
            depth_frame_flipped = cv2.flip(depth_frame, 1)
            print(depth_frame_flipped[200][240])
            filtered_depth_frame2 = np.where(depth_frame_flipped < 695, 0, depth_frame_flipped)
            print(filtered_depth_frame2[200][240])
            plt.imshow(filtered_depth_frame2)
            plt.show()

            if groundImage:
                pickle.dump(depth_frame_flipped, open(currentDir + "/GroundDepth.pkl", "wb"))
            else:
                groundMatrix = pickle.load(open(currentDir + "/GroundDepth.pkl", "rb"))
                diff = np.subtract(groundMatrix.astype(np.int16), depth_frame_flipped.astype(np.int16))
                diff = abs(diff)
                diff = np.where(diff <= 5, 0, diff)


            if not(groundImage):

                if layers:
                    for ran in range(0, 50, 1):
                        depth_copy = copy.deepcopy(diff)

                        filtered_depth_frame = np.where(depth_copy > ran + 1, 0, depth_copy)
                        filtered_depth_frame2 = np.where(filtered_depth_frame < ran, 0, filtered_depth_frame)
                        depth_frame2 = filtered_depth_frame2.reshape(depth_image_size)

                        plt.imshow(depth_frame2)
                        plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPicture" + str(ran) + ".png")

                diff = np.where(diff > 300, 0, diff)
                print(diff[200][240])
                pickle.dump(diff, open(currentDir + "/DepthDiff.pkl", "wb"))
                plt.imshow(diff)
                cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPictureA.png", diff)
                cropped = diff[150:250, 200:300]
                pickle.dump(cropped, open(currentDir + "/DepthCropped.pkl", "wb"))

                depth_frame = cropped * (256.0 / np.amax(cropped))
                colorized_frame = cv2.applyColorMap(np.uint8(depth_frame), cv2.COLORMAP_JET)
                cv2.imwrite("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPictureB.png", colorized_frame)
                plt.imshow(cropped)
                plt.show()

                plt.savefig("C:/Users/Administrator/PycharmProjects/ESAT7A1/depth_images/kinectDepthPictureC.png")
                #plt.show()


get_color_frame()
get_depth_frame()




