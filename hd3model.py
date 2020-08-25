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

        out_list = []
        x_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]
        y_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]

        for dy in range(y_range):
            for dx in range(x_range):
                temp_label_map = torch.zeros((B, H, W), device=label_map.device)
                temp_label_map[:, dy:, dx:] = resized_label_map[:, :-dy, :-dx]
                out_list.append(temp_label_map.eq(1).float().unsqueeze(3).to(label_map.device))
        
        out = torch.cat(out_list, dim=3).premute(0, 3, 1, 2)
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
            scale_factor = 1 / 2**(7 - len(corr_range))
            total_loss = None

            for prob_map, corr_range in zip(ms_prob, corr_range_list):
                # each level loss
                for tar_map in tar_map_list:
                    tar_size = (prob_map.size(2), prob_map.size(3))
                    extended_tar_map = self.extend_map(tar_map.float(), corr_range, tar_size)
                    if total_loss is None:
                        total_loss = self.criterion(prob_map, extended_tar_map)
                    else:
                        total_loss += self.criterion(prob_map, extended_tar_map)
            
            result['loss'] = total_loss
            # if total_loss is None:
            #     result['loss'] = total_loss
            # else:
            #     result['loss'] = total_loss / instance_num 
            # result['num'] = instance_num
        
        if get_instance_iou: 
            corr_range = [4, 4, 4, 4, 4]
            scale_factor = 1 / 2**(7 - len(corr_range))
            out_vect = resize_dense_vector(result['vect'] * scale_factor,
                                            img_list[0].shape[2], img_list[0].shape[3])
            total_loss = None
            for sur_map, tar_map in zip(sur_map_list, tar_map_list):
                sur_map = sur_map.float().requires_grad_()
                warped_map = self.flow_warp(sur_map, out_vect)
                if total_loss is None:
                    total_loss = self.criterion(warped_map, tar_map.float())
                else:
                    total_loss += self.criterion(warped_map, tar_map.float())
            
            result['loss'] = total_loss
            


            # new crossentropy loss
            # B, C, H, W = warped_map.size()
            # warped_map = torch.nn.functional.one_hot(warped_map.long().view(B, H, W))
            # warped_map = warped_map.view(B, warped_map.size()[3], H, W).float()
            # tar_map = tar_map.view(B, H, W)
        # if get_epe:
        #     scale_factor = 1 / 2**(self.ds - len(ms_vect) + 1)
        #     result['epe'] = self.eval_epe(ms_vect[-1] * scale_factor,
        #                                   label_list[0])
        # if get_vis:
        #     result['vis'] = get_visualization(img_list, label_list, ms_vect,
        #                                       ms_prob, self.ds)

        return result
