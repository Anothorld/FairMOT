from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

# the path you want to save your results for coco to voc

# img_dir = os.path.join(savepath)

datasets_list=['train2017', 'val2017']
# datasets_list = ['val2017']

classes_names = ['person']  # coco有80类，这里写要提取类的名字，以person为例
# Store annotations and train2014/val2014/... in this folder
dataDir = '/media/arnold/新加卷/Data/COCO'  # 原coco数据集

# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# mkr(img_dir)
# mkr(anno_dir)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


# def write_xml(anno_path, head, objs, tail):
#     if not os.path.exists(os.path.dirname(anno_path)):
#         os.makedirs(os.path.dirname(anno_path))
#     f = open(anno_path, "w")
#     f.write(head)
#     for obj in objs:
#         f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
#     f.write(tail)
def write_txt(anno_path, objs):
    if not os.path.exists(os.path.dirname(anno_path)):
        os.makedirs(os.path.dirname(anno_path))
    fw = open(anno_path, "w")
    for obj in objs:
        print('0 -1  {} {} {} {}'.format(obj[1], obj[2], obj[3], obj[4]), file=fw)
    fw.close()

#         os.makedirs(os.path.dirname(anno_path))

def save_annotations_and_imgs(coco, dataset, filename, objs, f):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = os.path.join(anno_dir, filename[:-3] + 'txt')
    img_path =os.path.join('COCO', 'images', filename)
    # print(img_path)
    print(img_path, file=f)
    # img = cv2.imread(img_path)
    # # if (img.shape[2] == 1):
    # #    print(filename + " not a RGB image")
    # #   return
    # shutil.copy(img_path, dst_imgpath)

    # head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    # tail = tailstr
    write_txt(anno_path, objs)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/images/%s' % (dataDir, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        if ann["iscrowd"] == True:
            continue
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [0, 0.5 * (xmin + xmax) / I.size[0], 0.5*(ymin+ymax) / I.size[1], bbox[2] / I.size[0], bbox[3] / I.size[1]]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
        plt.pause(0.01)


    return objs


for dataset in datasets_list:
    anno_dir = os.path.join(dataDir, 'labels_with_ids')
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    plt.ion()
    # ./COCO/annotations/instances_train2014.json
    orin_annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    txt_file = '{}/COCO.{}'.format(dataDir, 'train' if 'train' in dataset else 'val')

    # COCO API for initializing annotated data
    coco = COCO(orin_annFile)
    f_index = open(txt_file, 'w')
    # show all classes in coco
    classes = id2name(coco)
    print(classes)
    # [1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    for cls in classes_names:
        # Get ID number of this class
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        print(cls, len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs = showimg(coco, dataset, img, classes, classes_ids, show=True)

            # print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs, f_index)
    f_index.close()

