from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _fit_nms(heat, kernelw=5, kernelh=5):
    padW = (kernelw - 1) // 2
    padH = (kernelh - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernelh, kernelw), stride=1, padding=(padH, padW))
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

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
    return torch.tensor(h, device='cuda', dtype=torch.float)

def draw_umich_gaussian_mod(nms_mask, heatmap, center, rw, rh, part_mask, thresh=0.3, k=1):
    diameterW = 2 * rw + 1
    diameterH = 2 * rh + 1
    # gaussian = gaussian2D((int(1.4*diameter), diameter), sigma=diameter / 6)
    heatmap_pad = F.pad(nms_mask, pad=(rw, rw, rh, rh), mode='constant', value=0)

    x, y = int(center[0]), int(center[1])
    height, width = nms_mask.shape[2], nms_mask.shape[-1]

    left, right = x, x + 2*rw+1
    top, bottom = y, y + 2*rh+1

    masked_heatmap = heatmap_pad[:, :, top:bottom, left:right]
    if min(part_mask.shape) > 0 and min(masked_heatmap.shape) > 0 and heatmap[0, 0, y, x] >= thresh:  # TODO debug
        torch.max(masked_heatmap, part_mask * k, out=masked_heatmap)
        # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    masked_heatmap[:, :, rh, rw] = 0
        # if heatmap[0, 0, y, x] < 0.3:
        #     masked_heatmap *= (heatmap[0, 0, y, x].cpu().numpy() / 0.3)
    return heatmap_pad[:, :, rh: height+rh, rw:width+rw]

def soft_nms(heatmap, maxW, maxH, kernelW, kernelH, thresh=0.3):
    rw = int((kernelW - 1) / 2)
    rh = int((kernelH - 1) / 2)
    maximum = fit_nms(heatmap, maxW, maxH)
    part_mask = gaussian2D_mod((kernelH, kernelW), kernelW / 6, kernelH / 6)
    cY, cX = torch.where(maximum.squeeze() != 0)
    for cy, cx in zip(cY, cX):
        maximum = draw_umich_gaussian_mod(maximum.detach(), heatmap.detach(), torch.tensor([cx, cy]), rw, rh, part_mask, thresh)
        # test_m = maximum.squeeze().numpy()
    # test = maximum.squeeze().numpy()
    return ((1 - maximum) * heatmap)


def mot_decode(heat, wh, reg=None, cat_spec_wh=False, thresh=0.3, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # heat = _nms(heat)
    heat = soft_nms(heat, 5, 5, 19, 19, thresh)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
