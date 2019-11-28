import matplotlib.pyplot as plt
import cv2
import pickle
import os
import numpy as np


currentDir = os.path.dirname(os.path.abspath(__file__))


groundMatrix = pickle.load(open("C:/Users/Polo/Documents/GitHub/ESAT7A1"+ "/DepthDiff.pkl", "rb"))[50:250,200:450]
groundMatrix = np.where(groundMatrix < 20, 0, groundMatrix)
plt.imshow(groundMatrix)
plt.show()
new = []
for col in range(0, len(groundMatrix)):
    for row in range(0, len(groundMatrix[0])):
        new.append(groundMatrix[col, row])

different_values = list(set(new))
print(different_values)
different_values.remove(0)
corresponding_value = []
for element in different_values:
    y = new.count(element)
    print(y, element)
    if y < 5000:
        corresponding_value.append(y)
    else:
        corresponding_value.append(5000)

print(max(different_values))

plt.plot(corresponding_value, label='y = x')
plt.title('Depth values and numbers')
plt.ylabel('Y Axis')
plt.yticks(different_values)
plt.show()
