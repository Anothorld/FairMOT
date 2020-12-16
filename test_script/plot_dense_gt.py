import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

bg = np.zeros([256, 256], dtype=np.float32)
num = random.randint(500, 1000)
cnt = 0
while cnt < num:
    x = random.randint(0, 255)
    y = random.randint(0, 255)
    if bg[x, y] != 1:
        bg[x, y] = 1
        cnt += 1
    else:
        continue

 