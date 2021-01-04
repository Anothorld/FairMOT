import xml.dom.minidom
import os
from glob import glob
from tqdm import tqdm
import shutil
import cv2

WIDTH = 352
HEIGHT = 240

img_list = glob('/home/zlz/DataSets/CityCam_MOT/CityCam/*/*/*.jpg')
out_path = '/home/zlz/DataSets/CityCam_MOT'

f_train = open('citycam.train', 'w')

for img_path in tqdm(img_list):
    mask_path = os.path.dirname(img_path)+'_msk.png'
    xml_path = img_path.replace('.jpg', '.xml')
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    masked_img = cv2.bitwise_and(img, mask)
    img_save_path = os.path.join(os.path.dirname(img_path).replace('CityCam/', 'images/'), os.path.basename(img_path)).replace('jpg', 'png')
    txt_save_path = os.path.join(os.path.dirname(img_path).replace('CityCam/', 'labels_with_ids/'), os.path.basename(img_path)).replace('jpg', 'txt')
    if not os.path.exists(os.path.dirname(img_path).replace('CityCam/', 'images/')):
        os.makedirs(os.path.dirname(img_path).replace('CityCam/', 'images/'))
    if not os.path.exists(os.path.dirname(img_path).replace('CityCam/', 'labels_with_ids/')):
        os.makedirs(os.path.dirname(img_path).replace('CityCam/', 'labels_with_ids/'))
    cv2.imwrite(img_save_path, masked_img)

    try:
        dom =  xml.dom.minidom.parse(xml_path)
    except:
        continue
    root = dom.documentElement
    with open(txt_save_path, 'w') as anno:
        for vehicle in root.getElementsByTagName('vehicle'):
            x1 = int(vehicle.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            y1 = int(vehicle.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            x2 = int(vehicle.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            y2 = int(vehicle.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            xc = ((x2 + x1) / 2) / WIDTH
            yc = ((y2 + y1) / 2) / HEIGHT
            w = (x2 - x1) / WIDTH
            h = (y2 - y1) / HEIGHT
            print(f'0 -1 {xc} {yc} {w} {h}', file=anno)
    print(img_save_path.replace('/home/zlz/DataSets/', ''), file=f_train)