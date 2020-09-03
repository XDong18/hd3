import os
import numpy as np
import cv2
import torch
import multiprocessing as mp
from torch.nn import functional as F
from pycocotools.coco import COCO
import json

COLORS = np.array([[255,255,255], [0,255,255],[255,0,255],[255,255,0],[255,125,50],[125,255,50],[50,255,125],[255,50,125],[125,50,255],\
        [50, 125, 255]])

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
    sur_color_map = np.zeros((720, 1280, 3))
    tar_color_map = np.zeros((720, 1280, 3))

    color_id = 0
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
        mask = mask.squeeze().numpy().astype(np.uint8)
        print(mask.shape, new_mask.shape)
        sur_color_map[np.where(mask==1)] = COLORS[color_id % len(COLORS)]
        tar_color_map[np.where(new_mask==1)] = COLORS[color_id % len(COLORS)]
        color_id += 1
    
    sur_save_pth = os.path.join(out_sur_file, img_name_sur[:17], img_name_sur)
    tar_save_pth = os.path.join(out_tar_file, img_name_des[:17], img_name_des)
    # if not os.path.exists(os.path.join(out_sur_file, img_name_sur[:17])):
    #     os.makedirs(os.path.join(out_sur_file, img_name_sur[:17]))
    
    if not os.path.exists(os.path.join(out_tar_file, img_name_des[:17])):
        os.makedirs(os.path.join(out_tar_file, img_name_des[:17]))
    
    # ok = cv2.imwrite(sur_save_pth, sur_color_map)
    # print(ok)
    ok = cv2.imwrite(tar_save_pth, tar_color_map)
    print(ok)


def main():
    global reverse_img_dir
    global coco
    global out_sur_file
    global out_tar_file
    # global anno_to_instance

    fl_base = '/shared/xudongliu/code/semi-flow/hd3/predictions/semi_lr_0.001_gap_1_xia_epoch1/vec'
    json_fn = '/shared/xudongliu/bdd_part/seg_track_val_new.json'
    list_file = '/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt'
    out_sur_file = '/shared/xudongliu/code/semi-flow/hd3/generated_color_map/semi_lr_0.001_gap_1_xia_epoch1/frame_1'
    out_tar_file = '/shared/xudongliu/code/semi-flow/hd3/generated_color_map/semi_lr_0.001_gap_1_xia_epoch1/frame_0'
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
    pool.map(instance_warp, args)

if __name__ == "__main__":
    main()