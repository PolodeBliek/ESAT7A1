import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import peakutils

currentDir = os.path.dirname(os.path.abspath(__file__))
groundMatrix = pickle.load(open(currentDir + "/DepthDiff.pkl", "rb"))[50:250,200:450]

new = []
for col in range(0, len(groundMatrix)):
    for row in range(0, len(groundMatrix[0])):
        if groundMatrix[col, row] != 0:
            new.append(groundMatrix[col, row])

different_values = list(set(new))
corresponding_value = []
for element in different_values:
    y = new.count(element)
    corresponding_value.append(y)


time_series = corresponding_value

indices = peakutils.indexes(np.array(corresponding_value), min_dist=15)
print("peaks: ", indices)
print("number of peaks: ", len(indices))


plt.plot(corresponding_value, label='y = x')
plt.title('Depth values and numbers')
plt.ylabel('Y Axis')
plt.yticks(different_values)
plt.show()


