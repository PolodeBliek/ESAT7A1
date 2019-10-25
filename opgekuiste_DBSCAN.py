import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN


matrix = pickle.load(open("kinectfoto_detection_matrix2.pkl", "rb"))
matrix = np.array(matrix)
image = Image.fromarray(matrix)
image = image.resize(size=(int(len(matrix) / 2), int(len(matrix[0]) / 2)))
matrix = np.array(image)
nb_columns = len(matrix[0])
nb_rows = len(matrix)
d = []

# DBSCAN needs a dataset of the coordinates of all the 1's in the matrix
def matrix_to_coordinates():
    for row in range(nb_rows):
        for column in range(nb_columns):
            if matrix[row][column] == 1:
                d.extend(np.array([[row, column]]))


def plot_image():
    for i in range(len(d)):
        row = d[i][0]
        column = d[i][1]
        matrix[row][column] = db.labels_[i]
    plt.imshow(matrix)
    plt.show()


matrix_to_coordinates()
db = DBSCAN(eps=3, min_samples=5).fit(d)
plot_image()
print("NUMBER OF OBJECTS:", max(db.labels_))