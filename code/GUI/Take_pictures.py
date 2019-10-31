# still gotta decide where to drop the kinect images, I would suggest at least in a different directory

import tkinter as tk
import os
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process
from Main.whole_process import kinect_to_pc

currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\GUI", "")
max_pictures = 100


def take_pics(max_pics):
    kinectColorPicture, kinectDepthPicture = kinect_to_pc(1080, 1920, 4)

    for index in range(1, max_pics+1):
        colorPicName = f"KinectColorPicture{index}.png"

        if not os.path.exists(currentDir + "testImages/kinectColor" + colorPicName):
            depthPicName = f"KinectDepthPicture{index}.png"

            cv2.imwrite(currentDir + "testImages/kinectColor" + colorPicName, kinectColorPicture)
            cv2.imwrite(currentDir + "testImages/kinectDepth" + depthPicName, kinectDepthPicture)

            print(f"images saved as {colorPicName} and {depthPicName}")
            break

    if v1.get():
        p = Process(target=show_pics, args=(kinectColorPicture, kinectDepthPicture))
        p.start()
        p.join()

    return


def show_pics(*args):
    n = len(args)
    f = plt.figure()
    plt.suptitle("Kinect Pictures")
    for i in range(n):
        f.add_subplot(n, 1, i+1)
        plt.imshow(args[i])

    plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("250x150")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    picture_button = tk.Button(root, text="Take Picture", height=5, width=20, bg="green", command=lambda: take_pics(max_pictures))
    picture_button.grid(row=0, column=0, sticky="s")
    v1 = tk.IntVar()
    cb1 = tk.Checkbutton(root, variable=v1, text="show pictures")
    cb1.grid(row=1, column=0, sticky="n")
    root.mainloop()
