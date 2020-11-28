import os
import numpy as np
import cv2
import json
from tqdm import tqdm

image_path = '/home/zlz/DataSets/CrowdHuman/images'
f = open('/home/zlz/DataSets/CrowdHuman/annotation_train.odgt', 'r')
f_train = open('/home/zlz/DataSets/CrowdHuman/corwdhuman.train', 'w')
if not os.path.exists('/home/zlz/DataSets/CrowdHuman/labels_with_ids'):
    os.makedirs('/home/zlz/DataSets/CrowdHuman/labels_with_ids')
for line in tqdm(f.readlines()):
    anno = json.loads(line)
    id = anno['ID']
    if 'd30ae000416b4f14' in id:
        print(1)
    print(os.path.join(image_path, id+'.jpg'), file=f_train)
    img = cv2.imread(os.path.join(image_path, id+'.jpg'))
    (height, width) = img.shape[0:2]
    fw = open('/home/zlz/DataSets/CrowdHuman/labels_with_ids/{}.txt'.format(id), 'w')
    for inst in anno['gtboxes']:
        if inst['tag'] != 'person':
            continue
        vbox = [inst['vbox'][0] / width + 0.5 * inst['vbox'][2] / width,
                inst['vbox'][1] / height + 0.5 * inst['vbox'][3] / height,
                inst['vbox'][2] / width,
                inst['vbox'][3] / height]
        if 'ignore' in inst['head_attr'].keys():
            if inst['head_attr']['ignore'] == 1:
                hbox = [-1, -1, -1, -1]
            else:
                hbox = [inst['hbox'][0] / width + 0.5 * inst['hbox'][2] / width if inst['hbox'][0] / width > 0 else 0 + 0.5 * inst['hbox'][2] / width,
                        inst['hbox'][1] / height + 0.5 * inst['hbox'][3] / height if inst['hbox'][1] / height > 0 else 0 + 0.5 * inst['hbox'][3] / height,
                        inst['hbox'][2] / width,
                        inst['hbox'][3] / height]
        print('0 -1 {} {}'.format(' '.join(map(str, vbox)), ' '.join(map(str, hbox))), file=fw)

