import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(111)
# p = np.linspace(0, 1, 100)
# x = np.linspace(-12, 12, 100)
# y = np.linspace(-17, 17, 100)
# X, Y = np.meshgrid(x, y)
# gt = np.exp(-(X * X / (2 * 25/6 * 25/6) + Y * Y / (2 * 35/6 * 35/6)))
#
# # y_p = np.ones_like(gt) * np.power(np.ones_like(p) - x, 2) * np.log(p)
# # y_n = np.ones_like(gt) * np.power(p, 2) * np.log(np.ones_like(x) - p)
# p = 0.5
# loss_p = np.ones_like(gt) * np.power(np.ones_like(gt)*p - x, 2) * np.log(p)
# # loss_p = y_p
# ax1.plot_surface(X, Y, gt, facecolors=cm.Rainbow(loss_p))
# plt.show()
p = np.linspace(0, 1, 100)
y = np.exp(-np.power(p, 2) / 0.4)
ax1.plot(p, y)
plt.show()
print('done')