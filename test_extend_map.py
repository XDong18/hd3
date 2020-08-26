import torch
import torch.nn as nn
from models.hd3net import HD3Net
from hd3losses import *
from utils.visualizer import get_visualization
from utils.utils import *
from models.hd3_ops import *
import cv2
import os


def extend_map(label_map, corr_range, size): 
        # TODO constrain origin_map

        resized_label_map = torch.nn.functional.interpolate(label_map, size, mode='bilinear')
        # resized_origin_map = torch.nn.functional.interpolate(origin_map, size, mode='bilinear')

        B, _, H, W = resized_label_map.size()
        resized_label_map = resized_label_map.squeeze(1)

        out_list = []
        x_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]
        y_range = list(range(corr_range + 1))[::-1] + [-1 - p for p in range(corr_range)]

        for dy in y_range:
            for dx in x_range:
                temp_label_map = torch.zeros((B, H, W), device=label_map.device)
                if dx!=0 and dy!=0:
                    temp_label_map[:, dy:, dx:] = resized_label_map[:, :-dy, :-dx]
                elif dx==0 and dy==0:
                    temp_label_map[:,:,:] = resized_label_map[:,:,:]
                elif dx==0 and dy!=0:
                    temp_label_map[:, dy:, :] = resized_label_map[:, :-dy, :]
                elif dx!=0 and dy==0:
                    temp_label_map[:, :, dx:] = resized_label_map[:, :, :-dx]

                out_list.append(temp_label_map.eq(1).float().unsqueeze(3).to(label_map.device))
        
        out = torch.cat(out_list, dim=3).permute(0, 3, 1, 2)
        return out


def main():
    file_name = '/shared/xudongliu/code/semi-flow/mask/b1c81faa-3df17267/b1c81faa-3df17267-0000001.png'
    np_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # print(np_img.shape)
    tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
    out = extend_map(tensor_img.float(), 1, np_img.shape)
    np_out = out.unsqueeze(0).permute(1,2,0).numpy()
    for i in range(9):
        out_fn = os.path.join('test_imgs', str(i)+'.png')
        cv2.imwrite(out_fn, np_out[:,:,i] * 255)

if __name__ == "__main__":
    main()