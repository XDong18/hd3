import torch
import torch.nn as nn
from models.hd3net import HD3Net
from hd3losses import *
from utils.visualizer import get_visualization
from utils.utils import *
from models.hd3_ops import *


class HD3Model(nn.Module):

    def __init__(self, task, encoder, decoder, corr_range=None, context=False):
        super(HD3Model, self).__init__()
        self.ds = 6  # default downsample ratio of the coarsest level
        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        self.corr_range = corr_range
        self.context = context
        self.criterion = torch.nn.CrossEntropyLoss()
        self.eval_epe = EndPointError
        self.hd3net = HD3Net(task, encoder, decoder, corr_range, context,
                             self.ds)

    def flow_warp(self, x, flo):
        """
        inverse warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid = torch.stack([
            2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0,
            2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        ],
                            dim=1)

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='nearest', padding_mode='border')

        return output


    def forward(self,
                img_list,
                label_list=None,
                get_vect=True,
                get_prob=False,
                get_loss=False,
                get_epe=False,
                get_vis=False):
        result = {}

        ms_prob, ms_vect = self.hd3net(torch.cat(img_list, 1))
        sur_map, tar_map = label_list

        if get_vect:
            result['vect'] = ms_vect[-1]

        # add flow warp part
        corr_range = [4, 4, 4, 4, 4]
        scale_factor = 1 / 2**(7 - len(corr_range))

        out_vect = resize_dense_vector(result['vect'] * scale_factor,
                                        1280, 720)
        
        warped_map = self.flow_warp(sur_map, out_vect)

        if get_prob:
            result['prob'] = ms_prob[-1]
        if get_loss:
            # new crossentropy loss
            result['loss'] = self.criterion(warped_map, tar_map)
        # if get_epe:
        #     scale_factor = 1 / 2**(self.ds - len(ms_vect) + 1)
        #     result['epe'] = self.eval_epe(ms_vect[-1] * scale_factor,
        #                                   label_list[0])
        # if get_vis:
        #     result['vis'] = get_visualization(img_list, label_list, ms_vect,
        #                                       ms_prob, self.ds)

        return result
