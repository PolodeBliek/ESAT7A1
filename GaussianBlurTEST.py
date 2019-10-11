import numpy as np
from PIL import Image
from hulpfunctie_sobel import *


name_image = 'HighRes.jpg'
img = np.array(Image.open("C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\" + name_image))#.astype(np.uint8)
img = grayscale(img)
h, w = img.shape
GaussianKernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
print(GaussianKernel)
NewImage = np.zeros((h,w))
for i in range(1, h - 1):
    for j in range(1, w - 1):
        verticalGrad = (GaussianKernel[0, 0] * img[i - 1, j - 1]) + \
                       (GaussianKernel[0, 1] * img[i - 1, j]) + \
                       (GaussianKernel[0, 2] * img[i - 1, j + 1]) + \
                       (GaussianKernel[1, 0] * img[i, j - 1]) + \
                       (GaussianKernel[1, 1] * img[i, j]) + \
                       (GaussianKernel[1, 2] * img[i, j + 1]) + \
                       (GaussianKernel[2, 0] * img[i + 1, j - 1]) + \
                       (GaussianKernel[2, 1] * img[i + 1, j]) + \
                       (GaussianKernel[2, 2] * img[i + 1, j + 1])

        NewImage[i - 1, j - 1] = abs(verticalGrad)
plt.imsave('C:\\Users\\Polo\\Documents\\GitHub\\ESAT7A1\\GaussianPhoto.jpg', NewImage, cmap='gray', format='jpg')
