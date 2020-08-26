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
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.eval_epe = EndPointError
        self.hd3net = HD3Net(task, encoder, decoder, corr_range, context,
                             self.ds)
    
    def extend_map(self, label_map, corr_range, size): 
        # TODO constrain origin_map

        resized_label_map = torch.nn.functional.interpolate(label_map, size, mode='bilinear')
        # resized_origin_map = torch.nn.functional.interpolate(origin_map, size, mode='bilinear')

        B, _, H, W = resized_label_map.size()
        resized_label_map = resized_label_map.squeeze(1)

        pad = torch.nn.ConstantPad2d(corr_range, 0)
        resized_label_map = pad(resized_label_map)

        out_list = []
        x_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]
        y_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]

        for dy in y_range:
            for dx in x_range:
                temp_label_map = pad(torch.zeros((B, H, W), device=label_map.device))
                temp_label_map[:, corr_range:-corr_range, corr_range:-corr_range] = resized_label_map[:, corr_range+dy:corr_range+dy+H, corr_range+dx:corr_range+dx+W]
                out_list.append(temp_label_map[:, corr_range:-corr_range, corr_range:-corr_range].eq(1).float().unsqueeze(3).to(label_map.device))

        
        out = torch.cat(out_list, dim=3).permute(0, 3, 1, 2)
        return out

    def forward(self,
                img_list,
                label_list=None,
                get_vect=True,
                get_prob=False,
                get_loss=False,
                get_epe=False,
                get_vis=False,
                get_instance_iou=False):

        result = {}

        ms_prob, ms_vect = self.hd3net(torch.cat(img_list, 1))
        # sur_map, tar_map = label_list
        instance_num = int(len(label_list) / 2)
        # print(instance_num)
        sur_map_list = label_list[:instance_num]
        tar_map_list = label_list[instance_num:]

        # sur_map = sur_map.float().requires_grad_()

        if get_vect:
            result['vect'] = ms_vect[-1]

        if get_prob:
            result['prob'] = ms_prob[-1]

        if get_loss:
            # add flow warp part
            corr_range_list = [4, 4, 4, 4, 4]

            total_loss = None

            for prob_map, corr_range in zip(ms_prob, corr_range_list):
                # each level loss
                for i, tar_map in enumerate(tar_map_list):
                    if tar_map.max()==-1:
                        continue
                    # print('prob', prob_map.size())
                    tar_size = (prob_map.size(2), prob_map.size(3))
                    extended_tar_map = self.extend_map(tar_map.float(), corr_range, tar_size)
                    if total_loss is None:
                        total_loss = self.criterion(prob_map, extended_tar_map)
                    else:
                        total_loss += self.criterion(prob_map, extended_tar_map)
            
            result['loss'] = total_loss
        
        # if get_instance_iou: 
        #     corr_range = [4, 4, 4, 4, 4]
        #     scale_factor = 1 / 2**(7 - len(corr_range))
        #     out_vect = resize_dense_vector(result['vect'] * scale_factor,
        #                                     img_list[0].shape[2], img_list[0].shape[3])
        #     total_loss = None
        #     for sur_map, tar_map in zip(sur_map_list, tar_map_list):
        #         sur_map = sur_map.float().requires_grad_()
        #         warped_map = self.flow_warp(sur_map, out_vect)
        #         if total_loss is None:
        #             total_loss = self.criterion(warped_map, tar_map.float())
        #         else:
        #             total_loss += self.criterion(warped_map, tar_map.float())
            
        #     result['loss'] = total_loss
            


        return result
