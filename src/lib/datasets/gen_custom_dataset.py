import cv2
import os

txt_path = '/home/arnold/PycharmProjects/nanonets_object_tracking/det/mmdet/faster_rcnn_r50_fpn_1x'
# image_path = ''
out_path = '/media/arnold/新加卷/Data/custom_dataset/'
data_file = '/home/{}/PycharmProjects/FairMOT/src/data/custom_dataset.train'.format(os.getlogin())
df = open(data_file, 'w')
text_list = os.listdir(txt_path)
text_list = ['Track4.txt', 'Track9.txt']
for txt in text_list:
    with open(os.path.join(txt_path, txt)) as t:
        buff = []
        if not os.path.exists(os.path.join(out_path, txt.split('.')[0], 'labels_with_ids')):
            os.makedirs(os.path.join(out_path, txt.split('.')[0], 'labels_with_ids'))
        last_id = '3'
        if txt.split('.')[0] == 'Track1':
            weight, height = 1550, 734
        if txt.split('.')[0] == 'Track4':
            height, weight = 980, 1920
        if txt.split('.')[0] == 'Track5':
            height, weight = 559, 1400
        if txt.split('.')[0] == 'Track9':
            height, weight = 874, 1116
        if txt.split('.')[0] == 'Track10':
            height, weight = 593, 615
        id_set = set()
        for line in t.readlines():
            frame_id = line.split(',')[0]
            id = line.split(',')[1]
            [x, y, w, h] = list(map(float, line.split(',')[2:6]))
            if frame_id == last_id and float(line.split(',')[6]) > 0.75:
                id_set.add(id)
                cx = (x + 0.5*w)/weight
                cy = (y+0.5*h)/height
                buff.append([0, -1, cx, cy, w/weight, h/height])
                # print('{} {} {} {} {} {}'.format(0, -1, cx, cy, w/weight, h/height), file=f)
            elif frame_id != last_id and int(frame_id) % 6 == 0 and float(line.split(',')[6]) > 0.75:
                if len(buff) != 0:
                    f = open(os.path.join(out_path, txt.split('.')[0], 'labels_with_ids', '{}.txt'.format(last_id)), 'w')
                    print(os.path.join('custom_dataset', txt.split('.')[0], 'images', str(last_id) + '.jpg'), file=df)
                    for bbox in buff:
                        print(' '.join(list(map(str, bbox))), file=f)
                    f.close()
                buff = []
                cx = (x + 0.5 * w) / weight
                cy = (y + 0.5 * h) / height
                buff.append([0, -1, cx, cy, w / weight, h / height])
                last_id = frame_id


