from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from PIL import Image
from Main.whole_process import process_image
import matplotlib.pyplot as plt

def plot_image(labels):
    matrix = np.zeros()
    for label in labels:


def image_to_datasetDBSCAN(im):
    X_data = []
    matrix = np.array(im)
    image = Image.fromarray(matrix)
    image = image.resize(size=(int(len(matrix) / 2), int(len(matrix[0]) / 2)))
    matrix = np.array(image)
    nb_columns = len(matrix[0])
    nb_rows = len(matrix)
    for row in range(nb_rows):
        for column in range(nb_columns):
            if matrix[row][column] == 1:
                X_data.extend(np.array([[row, column]]))
    return X_data


def gridsearch(all_im_data, nb_cluster):
    eps_space = np.arange(1, 5, 0.2)
    min_samples_space = np.arange(3, 50, 2)
    best_diff = 1000
    for eps in eps_space:
        for min_samples in min_samples_space:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            total_diff = 0
            for img, exp_clusters in zip(all_im_data, nb_cluster):
                pred_clusters = max(db.fit(img).labels_ + 1)
                diff = abs(exp_clusters - pred_clusters)
                total_diff += diff
            if total_diff < best_diff:
                best_diff = total_diff
                best_comb = [eps, min_samples]
                print(f'found new best {best_diff} with comb {best_comb}')
                #plt.imshow(db.labels_)
                #plt.show()


if __name__ == '__main__':
    s = []
    images = ["C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture1.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture2.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture3.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture4.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture5.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture6.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture7.PNG",
              "C:/Users/annel/Documents/3Bir/P&O/ESAT7A1/testImages/kinectColor/KinectColorPicture8.PNG"]
    nb_clusters = [5, 6, 6, 6, 4, 4, 4]
    for image in images:
        s.extend([image_to_datasetDBSCAN(process_image(image))])
    gridsearch(s, nb_clusters)
