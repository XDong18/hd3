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
    for i, line in enumerate(image_list):
        # for each pair to be evaluate

        # flow_map path
        image_name = line.strip(' \n').split(' ')[0]
        image_name = image_name.split('.')[0] + '.png'
        flow_fn = os.path.join(args.flow_maps, image_name)

        # video name and index
        video_name = image_name.split('/')[0]
        video_idx = video_dir[video_name]

        # image index
        image_idx = i + video_idx

        # load annotation via cocoapi
        annIds = coco.getAnnIds(imgIds=[image_idx], iscrowd=None)
        annos = coco.loadAnns(annIds)
        for i, anno in enumerate(annos):
            # for each instance in a frame
            # load mask
            mask = coco.annToMask(anno)
            # add flow
            new_mask = add_flow(mask, flow_fn)

            # encode mask and new_mask
            e_mask = encode(np.asfortranarray(mask))
            e_new_mask = encode(np.asfortranarray(new_mask))

            # compute IoU via coco mask API
            instance_iou = iou([e_mask], [e_new_mask], [0])
            
            print(i, instance_iou)
            total += instance_iou[0][0]
            num += 1


        print('average:', total / num)

if __name__ == "__main__":
    main()
