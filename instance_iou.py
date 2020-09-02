import os
import numpy as np
import cv2
import torch
import multiprocessing as mp
from torch.nn import functional as F
from pycocotools.coco import COCO
from pycocotools.mask import *
import json

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def flow_warp(x, flo):
    """
    inverse warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device)
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

    return output.squeeze().numpy().astype(np.uint8)

def instance_warp(fn_list):
    flow_fn, img_name_sur, img_name_des = fn_list
    #flow = readFlow(flow_fn)
    flow = cv2.imread(flow_fn, -1)[:, :, ::-1].astype(np.float)
    flow = (flow[:, :, :2] - 2.0 ** 15) / 64.0
    flow = torch.Tensor(flow).permute(2, 0, 1).contiguous().unsqueeze(dim=0)
    img_id = reverse_img_dir[img_name_sur]
    img_id_des = reverse_img_dir[img_name_des]
    annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    annIds_des = coco.getAnnIds(imgIds=[img_id_des], iscrowd=None)
    annos = coco.loadAnns(annIds)
    annos_des = coco.loadAnns(annIds_des)
    instance_ids = [anno['instance_id'] for anno in annos]
    instance_ids_des = [anno['instance_id'] for anno in annos_des]

#     total_iou = 0
#     total_num = 0
    total_ious = np.zeros(8)
    total_nums = np.zeros(8)
    for anno, instance_id in zip(annos, instance_ids):
        if instance_id not in instance_ids_des:
            continue

        idx = instance_ids_des.index(instance_id)
        anno_des = annos_des[idx]
        mask = coco.annToMask(anno)
        mask_des = coco.annToMask(anno_des)
        mask = torch.Tensor(mask).unsqueeze(dim=0).unsqueeze(dim=1)
        # print(flow.shape)
        new_mask = flow_warp(mask, flow)
        e_mask_des = encode(np.asfortranarray(mask_des))
        e_new_mask = encode(np.asfortranarray(new_mask))
        # compute IoU via coco mask API
        instance_iou = iou([e_mask_des], [e_new_mask], [0])
#         total_iou += instance_iou[0][0]
#         total_num += 1
        
        total_ious[anno['category_id'] - 1] += instance_iou[0][0]
        total_nums[anno['category_id'] - 1] += 1
    
    return [total_ious, total_nums]


def main():
    global reverse_img_dir
    global coco
    # global anno_to_instance

    fl_base = '/shared/xudongliu/code/semi-flow/hd3/predictions/seg_track_bdd_1e-3_xia_bce_epoch15/vec'
    json_fn = '/shared/xudongliu/bdd_part/seg_track_val_new.json'
    list_file = '/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt'
    coco = COCO(json_fn)

    with open(json_fn) as f:
        sur_dir = json.load(f)
    
    img_dir_list = sur_dir['images']

    reverse_img_dir = {img_dir['file_name']:img_dir['id'] for img_dir in img_dir_list}
    # anno_to_instance = { for anno_dir in anno_dir_list}

    args = []

    with open(list_file) as f:
        pair_list = f.readlines()
    
    for i, line in enumerate(pair_list):
        flow_name = os.path.join(fl_base, line.strip(' \n').split(' ')[0].split('.')[0] + '.png')
        img_name_sur = os.path.split(line.strip(' \n').split(' ')[1])[-1]
        img_name_des = os.path.split(line.strip(' \n').split(' ')[0])[-1]
        args.append([flow_name, img_name_sur, img_name_des])

    pool = mp.Pool(16)
    results = pool.map(instance_warp, args)
    iou_list = np.sum(np.concatenate([np.expand_dims(result[0], axis=0) for result in results], axis=0), axis=0)
    num_list = np.sum(np.concatenate([np.expand_dims(result[1], axis=0) for result in results], axis=0), axis=0)

    print(iou_list / num_list)
    print(num_list)

if __name__ == "__main__":
    main()
