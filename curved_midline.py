# importing libraries

import numpy as np
import cv2
from matplotlib import pyplot
import matplotlib.pyplot as plt

# loading and processing the image

img = cv2.imread('/kaggle/input/image-curved-lines/IMPORTANT.png')
inverted = cv2.bitwise_not(img)
gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
a1 = contours[0][:, 0]
a2 = contours[1][:, 0]

# fitting a polynomial to the points on the first curve and uniformly picking 1000 points on the curve

min_a1_x, max_a1_x = min(a1[:,0]), max(a1[:,0])
new_a1_x = np.linspace(min_a1_x, max_a1_x, 1000)
a1_coefs = np.polyfit(a1[:,0],a1[:,1], 20)
new_a1_y = np.polyval(a1_coefs, new_a1_x)

# fitting a polynomial to the points on the second curve and uniformly picking 1000 points on the curve

min_a2_x, max_a2_x = min(a2[:,0]), max(a2[:,0])
new_a2_x = np.linspace(min_a2_x, max_a2_x, 1000)
a2_coefs = np.polyfit(a2[:,0],a2[:,1], 20)
new_a2_y = np.polyval(a2_coefs, new_a2_x)

# finding the midpoints of the points detected on each curve

midx = [np.mean([new_a1_x[i], new_a2_x[i]]) for i in range(1000)]
midy = [np.mean([new_a1_y[i], new_a2_y[i]]) for i in range(1000)]

# plotting the curves and the midcurve

plt.plot(a1[:,0], a1[:,1],c='blue')
plt.plot(a2[:,0], a2[:,1],c='blue')
plt.plot(midx, midy, '--', c='purple')

# displaying all the curves

pyplot.imshow(img)
pyplot.show()
