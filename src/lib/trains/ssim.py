from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
import logging
from math import exp
import numpy as np
import itertools

device = 'cuda'

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, NL='relu',same_padding=True, bn=False, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, dilation=dilation, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, dilation=1, bn=False, bias=True, groups=1):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None

    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    if input_length is None:
        return None
    assert padding == "same"
    output_length = input_length
    return (output_length + stride - 1) // stride

def same_padding_length(input_length, filter_size, stride, dilation=1):
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    output_length = (input_length + stride - 1) // stride
    pad_length = max(0, (output_length - 1) * stride + dilated_filter_size - input_length)
    return pad_length

def compute_same_padding2d(input_shape, kernel_size, strides, dilation):
    space = input_shape[2:]
    assert len(space) == 2, "{}".format(space)
    new_space = []
    new_input = []
    for i in range(len(space)):
        pad_length = same_padding_length(
            space[i],
            kernel_size[i],
            stride=strides[i],
            dilation=dilation[i])
        new_space.append(pad_length)
        new_input.append(pad_length % 2)
    return tuple(new_space), tuple(new_input)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return (window / window.sum()).to(device=device)


def t_ssim(img1, img2, img11, img22, img12, window, channel, dilation=1, size_average=True):
    window_size = window.size()[2]
    input_shape = list(img1.size())

    padding, pad_input = compute_same_padding2d(input_shape, \
                                                kernel_size=(window_size, window_size), \
                                                strides=(1,1), \
                                                dilation=(dilation, dilation))
    if img11 is None:
        img11 = img1 * img1
    if img22 is None:
        img22 = img2 * img2
    if img12 is None:
        img12 = img1 * img2

    if pad_input[0] == 1 or pad_input[1] == 1:
        img1 = F.pad(img1, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img2 = F.pad(img2, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img11 = F.pad(img11, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img22 = F.pad(img22, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img12 = F.pad(img12, [0, int(pad_input[0]), 0, int(pad_input[1])])

    padd = (padding[0] // 2, padding[1] // 2)

    mu1 = F.conv2d(img1, window , padding=padd, dilation=dilation, groups=channel)
    mu2 = F.conv2d(img2, window , padding=padd, dilation=dilation, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    si11 = F.conv2d(img11, window, padding=padd, dilation=dilation, groups=channel)
    si22 = F.conv2d(img22, window, padding=padd, dilation=dilation, groups=channel)
    si12 = F.conv2d(img12, window, padding=padd, dilation=dilation, groups=channel)

    sigma1_sq = si11 - mu1_sq
    sigma2_sq = si22 - mu2_sq
    sigma12 = si12 - mu1_mu2

    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))


    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs

class NORMMSSSIM(torch.nn.Module):

    def __init__(self, sigma=1.0, levels=5, size_average=True, channel=1):
        super(NORMMSSSIM, self).__init__()
        self.sigma = sigma
        self.window_size = 5
        self.levels = levels
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', create_window(self.window_size, self.channel, self.sigma))
        self.register_buffer('weights', torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device=device))


    def forward(self, img1, img2):
        img1 = (img1 + 1e-12) / (img2.max() + 1e-12)
        img2 = (img2 + 1e-12) / (img2.max() + 1e-12)

        img1 = img1 * 255.0
        img2 = img2 * 255.0

        msssim_score = self.msssim(img1, img2)
        return 1 - msssim_score

    def msssim(self, img1, img2):
        levels = self.levels
        mssim = []
        mcs = []

        img1, img2, img11, img22, img12 = img1, img2, None, None, None
        for i in range(levels):
            l, cs = \
                    t_ssim(img1, img2, img11, img22, img12, \
                                Variable(getattr(self, "window"), requires_grad=False),\
                                self.channel, size_average=self.size_average, dilation=(1 + int(i ** 1.5)))

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
            mssim.append(l)
            mcs.append(cs)

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        weights = Variable(self.weights, requires_grad=False)

        return torch.prod(mssim ** weights)