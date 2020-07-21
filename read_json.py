import json
import os

fn = '/data5/bdd100k/labels/seg_track/seg_track_val.json'
out_fn = 'lists/seg_track_val.txt'

with open(fn) as f:
    seg_track_dir = json.load(f)

image_list = seg_track_dir['images']

with open(out_fn, 'w') as f:
    for i, image_info in enumerate(image_list):
        video_id = image_info['video_id']
        if i==len(image_list) - 1:
            break
        next_video_id = image_list[i + 1]['video_id']
        if video_id == next_video_id:
            image_fn = os.path.join(image_info['file_name'][:17], image_info['file_name'])
            next_image_info = image_list[i + 1]
            next_iamge_fn = os.path.join(next_image_info['file_name'][:17], next_image_info['file_name'])
            new_line = image_fn + ' ' + next_iamge_fn + '\n'
            f.write(new_line)
