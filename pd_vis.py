import cv2
import os

sur = 'pd_mask/bdd-Sintel-val-new'
des = 'pd_mask/bdd-Sintel-val-new-vis'

def main():
    for video_fn in os.listdir(sur):
        if '.ipynb' in video_fn:
            continue
        if not os.path.exists(os.path.join(des, video_fn)):
            os.makedirs(os.path.join(des, video_fn))
        video_dir = os.path.join(sur, video_fn)
        des_video_dir = os.path.join(des, video_fn)
        for img_fn in os.listdir(video_dir):
            img = cv2.imread(os.path.join(video_dir, img_fn))
            img *= 255
            cv2.imwrite(os.path.join(des_video_dir, img_fn), img)

if __name__ == "__main__":
    main()
