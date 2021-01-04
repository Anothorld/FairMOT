import cv2
import numpy as np

test_img = cv2.imread('/media/data/DataSet/VisDrone/VisDrone2019-MOT-train/images/uav0000243_00001_v/0000696.jpg')
my_anno = open('/media/data/DataSet/VisDrone/VisDrone2019-MOT-train/labels_with_ids/uav0000243_00001_v/0000696.txt', "r")
anno_orignal = open('/media/data/DataSet/VisDrone/VisDrone2019-MOT-train/annotations/uav0000243_00001_v.txt', "r")
WIDTH = test_img.shape[1]
HEIGHT = test_img.shape[0]
paint = test_img.copy()
for line in my_anno.readlines():
    cls, id, xc, yc, w, h = line.split(' ')
    xc = float(xc) * WIDTH
    yc = float(yc) * HEIGHT
    w = float(w) * WIDTH
    h = float(h) * HEIGHT
    x1 = xc - w / 2
    x2 = x1 + w
    y1 = yc - h / 2
    y2 = y1 + h
    paint = cv2.rectangle(paint, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

for line in anno_orignal.readlines():
    
    frame_id, id, x1, y1, w, h, sco, cls, trun, occ = list(map(int, line.split(',')))
    if frame_id != 696 or cls not in [4, 5, 6, 9]:
        continue
    x2 = x1 + w
    y2 = y1 + h
    paint = cv2.rectangle(paint, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))

cv2.imwrite('out_img.png', paint)