import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import *
import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser(description='segmentation tracking verify protocol')
    parser.add_argument('--json', type=str, help='json file path')
    parser.add_argument('--flow_maps', type=str, help='flow_maps directory')
    parser.add_argument('--pair_list', type=str, help='pair list directory')
    return parser.parse_args()


# FRAME_NUMS = [201, 201, 202, 202]
# VIDEO_NAMES = {'b1c81faa-3df17267': 0, 'b1c81faa-c80764c5': 1, 'b1c9c847-3bda4659': 2, 'b1ca2e5d-84cf9134': 3}

def add_flow(mask_img, flow_fn):
    '''
    mask_img: np.array
    flow_fn: str
    return: np.array
    '''
    # read flow_map
    flow = cv2.imread(flow_fn, -1)[:,:,::-1].astype(np.float)

    # new mask
    new_mask = np.zeros(mask_img.shape, dtype=mask_img.dtype)
    valid_flow = flow[:,:,2]>0

    # convert flow
    flow = (flow[:,:,:2] - 2.0 ** 15) / 64.0
    for i in np.arange(new_mask.shape[0]):
        for j in np.arange(new_mask.shape[1]):
            if valid_flow[i,j]:
                target_i = int(np.round(i - flow[i, j, 0]))
                target_j = int(np.round(j - flow[i, j, 1]))
                if target_i>=0 and target_i<new_mask.shape[0] and target_j>=0 and target_j<new_mask.shape[1]:
                    new_mask[i, j] = mask_img[target_i, target_j]
    
    return new_mask

def main():
    args = get_parser()
    
    
    # coco = COCO('/shared/xudongliu/code/semi-flow/hd3/bdd100k_json/seg_track_val.json')
    # with open('/shared/xudongliu/code/semi-flow/hd3/lists/seg_track_val.txt') as f:
    #     image_list = f.readlines()

    # load json file in coco format
    coco = COCO(args.json)
    with open(args.pair_list) as f:
        image_list = f.readlines()
    
    # load json file and extract video information
    video_dir = {}
    with open(args.json) as f:
        json_dir = json.load(f)

    for video_info in json_dir['videos']:
        video_dir[video_info['name']] = video_info['id']

    num = 0
    total = 0
    # image_list = image_list[:806] # TODO
    for i, line in enumerate(image_list):
        image_name = line.strip(' \n').split(' ')[0]
        image_name = image_name.split('.')[0] + '.png'
        # flow_fn = os.path.join('/shared/xudongliu/code/semi-flow/hd3/predictions/fc_pre_KT_seg_track_val/vec', image_name)
        flow_fn = os.path.join(args.flow_maps, image_name)
        # print(flow_fn)
        video_name = image_name.split('/')[0]
        video_idx = video_dir[video_name]
        image_idx = i + video_idx
        annIds = coco.getAnnIds(imgIds=[image_idx], iscrowd=None)
        annos = coco.loadAnns(annIds)
        for i, anno in enumerate(annos):
            mask = coco.annToMask(anno)
            new_mask = add_flow(mask, flow_fn)
            e_mask = encode(np.asfortranarray(mask))
            e_new_mask = encode(np.asfortranarray(new_mask))
            instance_iou = iou([e_mask], [e_new_mask], [0])
            print(i, instance_iou)
            total += instance_iou[0][0]
            num += 1


        print('average:', total / num)

if __name__ == "__main__":
    main()
