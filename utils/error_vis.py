import cv2
import numpy as np 
import os


def main():
gt_fn = '/shared/xudongliu/code/semi-flow/hd3/semi_color_mask_lr_0.001/frame_1/b1c81faa-3df17267/b1c81faa-3df17267-0000002.jpg'
pred_fn = '/shared/xudongliu/code/semi-flow/hd3/generated_color_map/semi_lr_0.001_e_100/frame_0/b1c81faa-3df17267/b1c81faa-3df17267-0000002.jpg'
gt_img = cv2.imread(gt_fn)
pred_img = cv2.imread(pred_fn)
gt_valid = np.zeros((gt_img.shape[1:]))
pred_valid = np.zeros((pred_img.shape[1:]))
gt_valid[np.where(np.sum(gt_img, axis=2)>0)] = 1
pred_valid[np.where(np.sum(pred_img, axis=2)>0)] = 1
error_map = gt_valid - pred_valid

error_img = np.zeros(gt_img.shape)
error_img[np.where(error_map==1)] = [0,0,255]
error_img[np.where(error_map==-1)] = [255,0,0]
out_fn = 'error_map.png'
cv2.imwrite(out_fn, error_img)

