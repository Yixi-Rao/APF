
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import math


def U_all(x, y, obstacle_List):
    U_rep = 0
    for ob in obstacle_List:
        ob_dis = np.sqrt((x - 3) ** 2 + (y - 3) ** 2)
        if (ob_dis <= 0.6):
            U_rep += 0.5 * 20 * (1 / (ob_dis) - (1 / 0.6)) ** 2
    U_att = 0.5 * 1 * ((x - 5) ** 2 + (y - 5) ** 2)
    return U_att + U_rep


fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(0, 10, 0.1)
y = np.arange(0, 10, 0.1)
X, Y = np.meshgrid(x, y)
# U    = U_all(X,Y,[[2,2]])
r = (X - 5) ** 2 + (Y - 5) ** 2
Z2 = np.sqrt(r)
Z1 = (r / 2) * 1
plt.xlabel('x')
plt.ylabel('y')
#ax.plot_surface(X,Y,Z2, rstride=1, cstride=1,cmap='rainbow')
ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow')

plt.show()
