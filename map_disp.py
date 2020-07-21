import os
import numpy as np
import cv2

def map_disp(flow_fn, disp2_fn):
    flow = cv2.imread(flow_fn, -1)[:,:,::-1].astype(np.float)
    disp2 = cv2.imread(disp2_fn, -1)
    new_disp2 = np.zeros(disp2.shape, dtype=disp2.dtype)

    valid_flow = flow[:,:,2]>0
    # valid_disp2 = disp2>0
    flow = (flow[:,:,:2] - 2.0 ** 15) / 64.0
    # disp2 = np.float(disp2) / 256.0
    for i in np.arange(new_disp2.shape[0]):
        for j in np.arange(new_disp2.shape[1]):
            if valid_flow[i,j]:
                target_i = int(np.round(i + flow[i, j, 0]))
                target_j = int(np.round(j + flow[i, j, 1]))
                # print(target_i, target_j)
                if target_i>=0 and target_i<new_disp2.shape[0] and target_j>=0 and target_j<new_disp2.shape[1]:
                    new_disp2[i, j] = disp2[target_i, target_j]
    
    return new_disp2

def main():
    flow_dir = 'prediction/pre_KT_scene/flow'
    disp_dir = 'prediction/pre_KT_scene/disp_1'
    new_dir = 'disp_1_test'

    flow_list = sorted(os.listdir(flow_dir))
    disp_list = sorted(os.listdir(disp_dir))

    for f, d in zip(flow_list, disp_list):
        new_fn = map_disp(os.path.join(flow_dir, f), os.path.join(disp_dir, d)).astype('uint16')
        cv2.imwrite(os.path.join(new_dir, f), new_fn)


if __name__ == "__main__":
    main()
