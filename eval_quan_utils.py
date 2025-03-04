import numpy as np
import torch
import torch.nn as nn
import cv2

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)

    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def compute_WE(img_t, img_s_warp, occ_mask, threshold=1):

    valid_pixels = torch.sum(occ_mask == 1)
    mean_error = torch.sum(torch.absolute(occ_mask * img_t - occ_mask * img_s_warp)) / (valid_pixels + 1e-10)
    return mean_error


def load_image(imfile, range=255.0):
    img = cv2.imread(imfile)
    img = torch.from_numpy(img).permute(2, 0, 1).float()[[2, 1, 0], :] / range
    return img[None]


class Metric:

    def __init__(self, name, crop_begin, crop_end):
        self.values_counter = 0
        self.name = name
        self.values = np.zeros(200).astype(float)
        self.crop_begin = crop_begin
        self.crop_end = crop_end

    def add(self, value):
        self.values[self.values_counter] = value
        self.values_counter += 1

    def get_last(self):
        if self.values_counter:
            return self.values[self.values_counter-1]

    def avg(self):
        if self.values_counter:
            return self.values[self.crop_begin:self.values_counter-self.crop_end].mean()
        return 0
