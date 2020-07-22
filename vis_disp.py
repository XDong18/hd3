import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import *

FRAME_NUMS = [201, 201, 202, 202]
VIDEO_NAMES = {'b1c81faa-3df17267': 0, 'b1c81faa-c80764c5': 1, 'b1c9c847-3bda4659': 2, 'b1ca2e5d-84cf9134': 3}
COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100], [200, 100 ,50], [50, 100, 200]]

def map_disp(mask_img, flow_fn):
    flow = cv2.imread(flow_fn, -1)[:,:,::-1].astype(np.float)
    # disp2 = cv2.imread(disp2_fn, -1)
    # new_disp2 = np.zeros(disp2.shape, dtype=disp2.dtype)
    new_mask = np.zeros(mask_img.shape, dtype=mask_img.dtype)

    valid_flow = flow[:,:,2]>0
    # valid_disp2 = disp2>0
    flow = (flow[:,:,:2] - 2.0 ** 15) / 64.0
    # disp2 = np.float(disp2) / 256.0
    for i in np.arange(new_mask.shape[0]):
        for j in np.arange(new_mask.shape[1]):
            if valid_flow[i,j]:
                target_i = int(np.round(i - flow[i, j, 0]))
                target_j = int(np.round(j - flow[i, j, 1]))
                # print(target_i, target_j)
                if target_i>=0 and target_i<new_mask.shape[0] and target_j>=0 and target_j<new_mask.shape[1]:
                    new_mask[i, j] = mask_img[target_i, target_j]
    
    return new_mask

def main():
    num = 0
    total = 0
    coco = COCO('/shared/xudongliu/code/semi-flow/hd3/bdd100k_json/seg_track_val.json')
    with open('/shared/xudongliu/code/semi-flow/hd3/lists/seg_track_val.txt') as f:
        image_list = f.readlines()
    
    image_list = image_list[:806] # TODO
    i = 100
    line = image_list[i]
    # for i, line in enumerate(image_list):
    image_name = line.strip(' \n').split(' ')[0]
    image_name = image_name.split('.')[0] + '.png'
    flow_fn = os.path.join('/shared/xudongliu/code/semi-flow/hd3/predictions/fc_pre_KT_seg_track_val/vec', image_name)
    print(flow_fn)
    video_name = image_name.split('/')[0]
    video_idx = VIDEO_NAMES[video_name]
    image_idx = i + video_idx
    annIds = coco.getAnnIds(imgIds=[image_idx], iscrowd=None)
    annos = coco.loadAnns(annIds)
    vis_image_a = np.zeros((720, 1280, 3))
    vis_image_b = np.zeros((720, 1280, 3))
    for i, anno in enumerate(annos):
        mask = coco.annToMask(anno)
        vis_image_a[np.where(mask==1)] = COLORS[i]
        new_mask = map_disp(mask, flow_fn)
        vis_image_b[np.where(new_mask==1)] = COLORS[i]
        e_mask = encode(np.asfortranarray(mask))
        e_new_mask = encode(np.asfortranarray(new_mask))
        instance_iou = iou([e_mask], [e_new_mask], [0])
        # instance_iou = iou(e_mask, e_new_mask, [np.asfortranarray(np.zeros((1)))])
        print(i, instance_iou)
        total += instance_iou[0][0]
        num += 1

    cv2.imwrite('test_a.png', vis_image_a)
    cv2.imwrite('test_b.png', vis_image_b)


    print('average:', total / num)


    # for f, d in zip(flow_list, disp_list):
    #     new_fn = map_disp(os.path.join(flow_dir, f), os.path.join(disp_dir, d)).astype('uint16')
    #     cv2.imwrite(os.path.join(new_dir, f), new_fn)


if __name__ == "__main__":
    main()