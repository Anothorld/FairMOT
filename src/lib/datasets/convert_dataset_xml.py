import xml.dom.minidom
import os
from glob import glob
from tqdm import tqdm
import shutil

output_root = '/media/data/DataSet/DETRAC_TRACK'
image_root = '/media/data/DataSet/DETRAC_TRACK/DETRAC-train-data/Insight-MVT_Annotation_Train'

f_tv = open('detrac.train', 'w')
for xml_f in tqdm(glob("/home/zlz/DataSets/DETRAC_TRACK/DETRAC-Train-Annotations-XML-v3/*.xml")):
    file_name = os.path.basename(xml_f)
    video_dir = '_'.join(file_name.split('_')[:2])
    
    dom =  xml.dom.minidom.parse(xml_f)


    root = dom.documentElement

    frame_nodes = root.getElementsByTagName("frame")

    WIDTH = 960
    HEIGHT = 540
    
    for frame_node in frame_nodes:
        frame_id = int(frame_node.getAttribute('num'))
        with open(os.path.join(output_root, 'labels_with_ids', 'train' ,f'{video_dir}-img{frame_id:05d}.txt'), 'w') as f:
            shutil.copy(os.path.join(image_root, video_dir, f'img{frame_id:05d}.jpg'), os.path.join(output_root, 'images', 'train', f'{video_dir}-img{frame_id:05d}.jpg'))
            for car_node in frame_node.getElementsByTagName('target_list')[0].getElementsByTagName("target"):
                id = car_node.getAttribute('id')
                box_node = car_node.getElementsByTagName('box')
                height = float(box_node[0].getAttribute('height'))
                left = float(box_node[0].getAttribute('left'))
                top = float(box_node[0].getAttribute('top'))
                width = float(box_node[0].getAttribute('width'))
                x = (left + 0.5*width) / WIDTH
                y = (top + 0.5*height) / HEIGHT
                w = width / WIDTH
                h = height / HEIGHT
                print(f'0 {id} {x:.3f} {y:.3f} {w:.3f} {h:.3f}', file=f)
            print(os.path.join('images', 'train', f'{video_dir}-img{frame_id:05d}.jpg'), file=f_tv)


# <collection shelf="New Arrivals">
# <movie title="Enemy Behind">
#    <type>War, Thriller</type>
#    <format>DVD</format>
#    <year>2003</year>
#    <rating>PG</rating>
#    <stars>10</stars>
#    <description>Talk about a US-Japan war</description>
# </movie>
# <movie title="Transformers">
#    <type>Anime, Science Fiction</type>
#    <format>DVD</format>
#    <year>1989</year>
#    <rating>R</rating>
#    <stars>8</stars>
#    <description>A schientific fiction</description>
# </movie>
#    <movie title="Trigun">
#    <type>Anime, Action</type>
#    <format>DVD</format>
#    <episodes>4</episodes>
#    <rating>PG</rating>
#    <stars>10</stars>
#    <description>Vash the Stampede!</description>
# </movie>
# <movie title="Ishtar">
#    <type>Comedy</type>
#    <format>VHS</format>
#    <rating>PG</rating>
#    <stars>2</stars>
#    <description>Viewable boredom</description>
# </movie>
# </collection>