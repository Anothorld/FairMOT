import
import cv2
import numpy as np
import os

root = '/media/arnold/新加卷/Data'
f = open("/media/arnold/新加卷/Data/COCO/COCO.val")
for line in f.readlines():
    anno = line.replace('images', 'labels_with_ids').replace('jpg', 'txt').strip()
    img = cv2.imread(os.path.join(root, line.strip()))
    f_anno = open(os.path.join(root, anno))
    for bboxes in f_anno.readlines():
        box = list(map(float, bboxes.strip().split(' ')[-4:]))
        img = cv2.rectangle(img, (int(img.shape[1]*(box[0] - 0.5 * box[2])), int(img.shape[0]*(box[1] - 0.5 * box[3]))), (int(img.shape[1]*(box[0] + 0.5 * box[2])), int(img.shape[0]*(box[1] + 0.5 * box[3]))), (0, 0, 255))
        img = cv2.circle(img, (int(box[0]*img.shape[1]), int(box[1]*img.shape[0])), radius=5, color=(0, 0, 255), thickness=2)
    print(img.shape)
    cv2.imshow('val', img)
    cv2.waitKey(0)