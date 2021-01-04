import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math


def gaussian2D_mod(shape, sigmaW=1, sigmaH=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigmaW * sigmaW) + y * y / (2 * sigmaH * sigmaH)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius_mod(det_size, center_area=0.1):
    height, width = det_size
    ratio = max(height, width) / min(height, width)

    # r repeasent x axies offset
    r = np.sqrt((center_area * width * height / ratio))

    return (r, ratio * r) if height > width else ( ratio * r, r)

def draw_umich_gaussian_mod(heatmap, center, rw, rh, k=1):
    diameterW = 2 * rw + 1
    diameterH = 2 * rh + 1
    # gaussian = gaussian2D((int(1.4*diameter), diameter), sigma=diameter / 6)
    gaussian = gaussian2D_mod((diameterH, diameterW), sigmaW=diameterW / 6, sigmaH=diameterH / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = x - rw, x + rw
    top, bottom = y - rh, y + rh

    masked_heatmap = heatmap[y - rh:y + rh+1, x - rw:x + rw+1]
    masked_gaussian = gaussian
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        try:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        except:
            print('erro')
    return heatmap



img_path = "/media/data/DataSet/DETRAC_TRACK/images/train/MVI_20011-img00008.jpg"
label_path = img_path.replace('images', 'labels_with_ids').replace('jpg', 'txt')
ori_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
ori_w, ori_h = ori_img.shape[1], ori_img.shape[0]
hm_w = ori_w // 4
hm_h = ori_h // 4
hm = np.zeros([hm_h, hm_w])

anno = open(label_path, 'r')
for line in anno.readlines():
    label = line.strip().split(' ')
    xc = float(label[2]) * hm_w
    yc = float(label[3]) * hm_h
    w = float(label[4]) * hm_w
    h = float(label[5]) * hm_h



    radiusW, radiusH = gaussian_radius_mod((math.ceil(h), math.ceil(w)))
    radiusW = max(0, int(radiusW))
    radiusH = max(0, int(radiusH))


    ct = np.array(
        [xc, yc], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    # draw_gaussian(hm[cls_id], ct_int, radius)
    draw_umich_gaussian_mod(hm, ct_int, radiusW, radiusH)

hm = cv2.resize(hm, (ori_w, ori_h))
plt.imshow(ori_img)
plt.imshow(hm, alpha=0.4, cmap="jet")
plt.savefig('gauss_hm.png')

pass