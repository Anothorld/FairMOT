import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-12, 12, 1000)
x2 = np.linspace(-17, 17, 1000)
fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
mu = 0
sigmax = 25/5
sigmay = 25/6
pdf1 = np.exp(-((x1 - mu)**2)/(2*sigmax**2)) / (sigmax * np.sqrt(2*np.pi))
pdf2 = np.exp(-((x1 - mu)**2)/(2*sigmay**2)) / (sigmay * np.sqrt(2*np.pi))
def gaussian2D_mod(shape, sigmaW=1, sigmaH=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigmaW * sigmaW) + y * y / (2 * sigmaH * sigmaH)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
nmsW= 51
nmsH = 61
gausss= gaussian2D_mod((nmsH, nmsW), nmsW/5, nmsH/5)
ax1.imshow(gausss, cmap='plasma_r')
ax2.plot(x1, pdf1, label='/5')
ax2.plot(x1, pdf2, label='/6')
ax2.legend()
plt.show()
print('done')