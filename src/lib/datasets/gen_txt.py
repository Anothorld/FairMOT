import os
from glob import glob

WIDTH = 1904
HEIGHT = 1071
path = "/media/data/DataSet/VisDrone/VisDrone2019-MOT-test-dev"
out_path = "/home/zlz/PycharmProjects/FairMOT/src/data"
annos = os.listdir(os.path.join(path, 'annotations'))
f_train = open(os.path.join(out_path, 'visdrone.test'), 'w')
for anno_file in annos:
    with open(os.path.join(path, 'annotations', anno_file), 'r') as fr:
        for line in fr.readlines():
            frame_id, id, x, y, w, h, score, ctgry, trunc, occ = list(map(int, line.split(',')))
            if ctgry not in [4, 5, 6, 9] or score == 0:
                continue
            if not os.path.exists(os.path.join(path, 'labels_with_ids', anno_file.split('.')[0])):
                os.makedirs(os.path.join(path, 'labels_with_ids', anno_file.split('.')[0]))
                print(os.path.join('VisDrone2019-MOT-test-dev', 'images', anno_file.split('.')[0], f'{frame_id:07d}.jpg'), file=f_train)
            with open(os.path.join(path, 'labels_with_ids', anno_file.split('.')[0], f'{frame_id:07d}.txt'), 'a') as fw:
                x_center = (x + w / 2) / WIDTH
                y_center = (y + h / 2) / HEIGHT
                w /= WIDTH
                h /= HEIGHT
                print(f'0 {id} {x_center} {y_center} {w} {h}', file=fw)
