import os
from PIL import Image
import numpy as np

name_image = "1_rechthoeken.png"
currentDir      = os.path.dirname(os.path.abspath(__file__)).replace("\\code\\Testing", "")
directory       = currentDir + "\\testImages\\"
img = np.array(Image.open(directory + name_image))
