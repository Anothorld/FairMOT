import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious

PI = 3.14159265359
# from opts import opts
def gaussian_radius_mod(det_size, center_area=0.3):
    height, width = det_size
    ratio = height / width

    # r repeasent x axies offset
    r = np.sqrt(pow(width, 2) * center_area)

    return (r, ratio * r)

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D_mod(shape, sigmaW=1, sigmaH=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigmaW * sigmaW) + y * y / (2 * sigmaH * sigmaH)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian_mod(heatmap, center, rw, rh, k=1):
    diameterW = 2 * rw + 1
    diameterH = 2 * rh + 1
    # gaussian = gaussian2D((int(1.4*diameter), diameter), sigma=diameter / 6)
    gaussian = gaussian2D_mod((diameterH, diameterW), sigmaW=diameterW / 6, sigmaH=diameterH / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = max(0, x - rw), min(width, x + rw +1)
    top, bottom = max(0, y - rh), min(height, y + rh +1)

    masked_heatmap = heatmap[top:bottom, left:right]
    masked_gaussian = gaussian[top - (y - rh): bottom - (y - rh), left - (x - rw):right - (x - rw)]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # try:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # except:
        #     print('erro')
    return heatmap

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

# from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
# from utils.density import *
img_path = '/media/data/DataSet/DETRAC_TRACK/images/train/MVI_20064-img00489.jpg'
label_path = img_path.replace('images', 'labels_with_ids').replace('png', 'txt').replace('jpg', 'txt')

img = cv2.imread(img_path).astype(np.float32) / 255
labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
hm = np.zeros([img.shape[0], img.shape[1]])
# Normalized xywh to pixel xyxy format
labels = labels0[:, 2:].copy()
for label in labels:
    w = label[2] * img.shape[1] / 2
    h = label[3] * img.shape[0] / 2
    radiusW, radiusH = gaussian_radius_mod((math.ceil(h), math.ceil(w)))
    radiusW = max(0, int(radiusW))
    radiusH = max(0, int(radiusH))
    ct = np.array([label[0] * img.shape[1], label[1] * img.shape[0]], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    draw_umich_gaussian_mod(hm, ct_int, max(0, int(math.ceil(w))), max(0, int(math.ceil(h))))

img_plt = plt.imread(img_path)
plt.imshow(img_plt)
plt.imshow(hm, alpha=0.2, cmap='rainbow')
plt.savefig('hm.png')
print(np.sum(hm))
pass