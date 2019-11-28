import pickle
import matplotlib.pyplot as plt
import numpy as np

pkl = "DepthDiff.pkl"
Depth = np.array(pickle.load(open("C:/Users/Polo/Documents/GitHub/ESAT7A1/" + pkl, "rb")))
Depth = np.where(Depth < 35, 0, Depth)
DepthSumHor = []
print(len(Depth))
for x in range(len(Depth)):
    DepthSumHor.append(sum(Depth[x]))
DepthSumHor[0] = 0
DepthSumHor[-1] = 0
plt.plot(DepthSumHor)
plt.show()
