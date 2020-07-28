import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import *
import argparse
import json
import multiprocessing as mp

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)

def get_parser():
    parser = argparse.ArgumentParser(description='segmentation tracking verify protocol')
    parser.add_argument('--json', type=str, help='json file path')
    parser.add_argument('--out', type = str, help='out directory')
    return parser.parse_args()

def mask(pair):
    args = get_parser()
    i, line = pair
    image_name = line.strip(' \n').split(' ')[0]
    image_name = image_name.split('.')[0] + '.png'
    mkdir(os.path.join(args.out, image_name.split('/')[0]))

    # image index
    image_idx = i
    
    # load annotation via cocoapi
    annIds = coco.getAnnIds(imgIds=[image_idx], iscrowd=None)
    annos = coco.loadAnns(annIds)

    for i, anno in enumerate(annos):
        # for each instance in a frame
        # load mask
        if i==0:
            mask = coco.annToMask(anno)
        else:
            mask += coco.annToMask(anno)
    
    mask[np.where(mask>0)] = 1
    cv2.imwrite(image_name, mask)

def get_name_list(fn):
    name_list = []
    with open(fn) as f:
        seg_track_dir = json.load(f)

    image_list = seg_track_dir['images']
    for i, image_info in enumerate(image_list):
        image_fn = os.path.join(image_info['file_name'][:17], image_info['file_name'])
        name_list.append(image_fn)
    return image_list

def main():
    global coco
    args = get_parser()

    # load json file in coco format
    coco = COCO(args.json)
    img_list = get_name_list(args.json)
    pairs = list(enumerate(img_list))
    pool = mp.Pool(16)
    pool.map(mask, pairs)

if __name__ == "__main__":
    main()