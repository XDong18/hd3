import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import *
import argparse
import json
import multiprocessing as mp

# def get_parser():
#     parser = argparse.ArgumentParser(description='segmentation tracking verify protocol')
#     parser.add_argument('--json', type=str, help='json file path')
#     parser.add_argument('--flow_maps', type=str, help='flow_maps directory')
#     parser.add_argument('--pair_list', type=str, help='pair list directory')
#     return parser.parse_args()

def iou(pair):
    fn1, fn2 = pair
    img1 = cv2.imread(fn1)
    img2 = cv2.imread(fn2)
    e_mask = encode(np.asfortranarray(img1))
    e_new_mask = encode(np.asfortranarray(img2))
    map_iou = iou([e_mask], [e_new_mask], [0])
    return map_iou[0][0]

def main():
    pd_base = '/shared/xudongliu/code/semi-flow/hd3/pd_mask/bdd-Sintel-val'
    gt_base = '/shared/xudongliu/code/semi-flow/masks'
    # args = get_parser()

    # load json file in coco format
    with open(args.pair_list) as f:
        image_list = f.readlines()
    
    args = []
    for i, line in enumerate(image_list):
        gt_name = os.path.join(gt_base, line.strip(' \n').split(' ')[1].split('.')[0] + '.png')
        pd_name = os.path.join(pd_base, line.strip(' \n').split(' ')[1].split('.')[0] + '.png')
        args.append([gt_name, pd_name])

    pool = mp.Pool(16)
    iou_list = pool.map(iou, args)
    iou_list = np.array(iou_list)
    print(iou_list.mean())

if __name__ == "__main__":
    main()
