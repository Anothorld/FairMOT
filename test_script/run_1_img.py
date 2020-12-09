import torch
import __init__paths
import os
import cv2
import json
from opts import opts
import numpy as np
import torch.nn as nn
import matplotlib.cm as cmx
import matplotlib.colors as colors
from models.model import create_model, load_model
from datasets.dataset.jde import LoadImages, LoadImagesAndLabels, JointDataset
import matplotlib.pyplot as plt
from models.decode import _nms, _fit_nms
from models.decode import mot_decode
from torchvision.transforms import transforms as T
from utils.post_process import ctdet_post_process
from utils.utils import xyxy2xywh
import torch.nn.functional as F

THRESH = 0.3

def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1,.. . N-1 to a distinct
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

sig = nn.Sigmoid()
arch = 'dla_34'
heads = {'hm': 1, 'wh': 2, 'id': 512, 'reg': 2}
# arch = 'hrnet_18'
# heads = {'hm': 1, 'wh': 2, 'id': 128, 'reg': 2}
# model_path = '/media/arnold/新加卷/Data/all_dla34.pth'
# model_path = '/home/arnold/PycharmProjects/FairMOT/exp/mot/dla_tf_custom/model_29.pth'
# model_path = '/home/arnold/PycharmProjects/FairMOT/exp/mot/dla_tf_MOT20/model_1.pth'
# model_path = '/home/arnold/PycharmProjects/FairMOT/exp/mot/dla_tf_MOT20_ft/model_9.pth'
# model_path = '/media/arnold/新加卷/Data/model_10.pth'
model_path = '/media/arnold/新加卷/Data/new_start_5.pth'
model = create_model(arch, heads, 256)
model = load_model(model, model_path)
# model = load_model(model, '/home/arnold/PycharmProjects/FairMOT/exp/mot/all_hrnet_ft_custom_20/model_22.pth')
# model = load_model(model, '/home/arnold/PycharmProjects/FairMOT/exp/mot/dla_tf_custom/model_29.pth')

model = model.to(torch.device('cuda'))
model.eval()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt = opts().init()
img_path = '/media/arnold/新加卷/Data/trackdata/B-folder/B-data/Track2/'
imageLoader = LoadImages(img_path)
f = open('/home/arnold/PycharmProjects/FairMOT/test_script/data.json')
data_config = json.load(f)
trainset_paths = data_config['train']
valset_paths = data_config['test_emb']
dataset_root = data_config['root']
f.close()
transforms = T.Compose([T.ToTensor()])
# imageAnnoLoader = JointDataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)

fig1 = plt.figure(figsize=(16,9), dpi=150)
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax21 = fig1.add_subplot(223)
ax22 = fig1.add_subplot(224)
fig1.tight_layout()
ax11.set_title('hm_')
ax12.set_title('det result')
ax21.set_title('hm_sigmoid')
ax22.set_title('hm_sigmoid_nms')

def fit_nms(heat, kernelw=5, kernelh=5):
    padW = (kernelw - 1) // 2
    padH = (kernelh - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernelh, kernelw), stride=1, padding=(padH, padW))
    keep = (hmax == heat).float()
    return keep

def gaussian2D_mod(shape, sigmaW=1, sigmaH=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigmaW * sigmaW) + y * y / (2 * sigmaH * sigmaH)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian_mod(nms_mask, heatmap, center, rw, rh, part_mask, thresh=0.3, k=1):
    diameterW = 2 * rw + 1
    diameterH = 2 * rh + 1
    # gaussian = gaussian2D((int(1.4*diameter), diameter), sigma=diameter / 6)
    heatmap_pad = F.pad(nms_mask.cpu(), pad=(rw, rw, rh, rh), mode='constant', value=0).numpy()

    x, y = int(center[0]), int(center[1])
    height, width = nms_mask.shape[2], nms_mask.shape[-1]

    left, right = x, x + 2*rw+1
    top, bottom = y, y + 2*rh+1

    masked_heatmap = heatmap_pad[:, :, top:bottom, left:right]
    masked_gaussian = part_mask
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0 and heatmap[0, 0, y, x] >= thresh:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    masked_heatmap[:, :, rh, rw] = 0
        # if heatmap[0, 0, y, x] < 0.3:
        #     masked_heatmap *= (heatmap[0, 0, y, x].cpu().numpy() / 0.3)

    return torch.tensor(heatmap_pad[:, :, rh: height+rh, rw:width+rw])

def soft_nms(heatmap, maxW, maxH, kernelW, kernelH, thresh=0.3):
    rw = int((kernelW - 1) / 2)
    rh = int((kernelH - 1) / 2)
    maximum = fit_nms(heatmap, maxW, maxH)
    part_mask = gaussian2D_mod((kernelH, kernelW), kernelW / 6, kernelH / 6)
    cY, cX = torch.where(maximum.squeeze() != 0)
    for cy, cx in zip(cY, cX):
        maximum = draw_umich_gaussian_mod(maximum, heatmap,torch.tensor([cx, cy]), rw, rh, part_mask, thresh)
        test_m = maximum.squeeze().numpy()
    test = maximum.squeeze().numpy()
    return (1 - maximum) * heatmap


plt.ion()
for path, img, img0 in imageLoader:
    blob = torch.from_numpy(img).cuda().unsqueeze(0)
    output = model(blob)[-1]

    origin_shape = img0.shape[0:2]
    width = origin_shape[1]
    height = origin_shape[0]
    inp_height = 608
    inp_width = 1088
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': inp_height // opt.down_ratio,
            'out_width': inp_width // opt.down_ratio}
    hm = output['hm']
    hm_sig = output['hm'].sigmoid_()
    hm_sig_soft_nms = soft_nms(hm_sig.detach().cpu(), 3, 3, 21, 21, thresh=THRESH)
    wh = output['wh']
    reg = output['reg'] if opt.reg_offset else None
    opt.K = 200
    detections, inds = mot_decode(hm_sig, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, thresh=THRESH, K=opt.K)
    dets = post_process(opt, detections, meta)
    dets = merge_outputs(opt, [dets])[1]
    remain_inds = dets[:, 4] > THRESH
    dets = dets[remain_inds]
    remain_inds = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]) > opt.min_box_area
    dets = dets[remain_inds]
    rects = []
    cmap = get_cmap(dets.shape[0])
    for i, det in enumerate(dets):
        col = cmap(i)
        rect = plt.Rectangle((det[0], det[1]), det[2]-det[0], det[3]-det[1], fill=False, edgecolor=col, linewidth=1)
        ax12.add_patch(rect)

    # hm_nms = (hm_nms - hm_nms.mean())/hm_nms.std()
    im11 = ax11.imshow(hm.detach().cpu().squeeze())
    # ax11.imshow(hm_nms.squeeze(), cmap='plasma')
    ax11.set_title('hm_')
    im12 = ax12.imshow(img0[:, :, [2, 1, 0]])
    ax12.set_title('det result')
    # plt.colorbar(im11, ax=ax11)
    # plt.colorbar(im12, ax=ax12)
    im21 = ax21.imshow(hm_sig.detach().cpu().squeeze(), cmap='plasma_r')
    # plt.colorbar(im21, ax=ax21)
    ax21.set_title('hm_sigmoid')
    im22 = ax22.imshow(hm_sig_soft_nms.squeeze(),
                       cmap='plasma_r')
    # plt.colorbar(im22, ax=ax22)
    ax22.set_title('hm_sigmoid_nms')
    # im22 = ax22.imshow(torch.where(hm_sig_nms > THRESH, torch.tensor(1), torch.tensor(0)).squeeze().squeeze(), cmap='plasma_r')
    # plt.colorbar(im22, ax=ax22)
    # ax22.set_title('hm_sigmoid_nms')
    print(output)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    plt.savefig('{}_{}_{}.png'.format(os.path.basename(model_path).split('.')[0], os.path.basename(img_path).split('.')[0], THRESH), bbox_inches='tight')
    plt.show()
    plt.pause(3)
    print('done')
# plt.ion()
# for ret in imageAnnoLoader:
#
#     img0 = ret['input'].numpy()
#     hm = ret['hm']
#     hm_nms = _nms(torch.tensor(ret['hm']))
#     # hm_nms = (hm_nms - hm_nms.mean())/hm_nms.std()
#     # ax11.imshow(torch.where(hm_nms>0.3, hm_nms, torch.ones_like(hm_nms)*-10).squeeze())
#     ax11.imshow(hm_nms.squeeze(), cmap='plasma_r')
#     ax11.set_title('hm_nms')
#     ax12.imshow(hm.squeeze(), cmap='plasma_r')
#     ax12.set_title('hm')
#     heatmap_img = cv2.applyColorMap((hm.squeeze()*255).astype('uint8'), cv2.COLORMAP_JET)
#     heatmap_img = cv2.resize(heatmap_img, (img0.shape[2], img0.shape[1]))
#     img0 = cv2.cvtColor(img0.transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * (255, 255, 255)
#     img0 = cv2.addWeighted(heatmap_img, 0.3, img0.astype('uint8'), 0.7, 0)
#
#
#
#     cv2.imshow('img', img0)
#     plt.show()
#     plt.pause(0.01)
#     cv2.waitKey(0)