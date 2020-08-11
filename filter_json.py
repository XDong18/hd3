import json
import os

fn = '/data5/bdd100k/labels/seg_track/seg_track_train.json'
root_dir = '/data5/bdd100k/images/track/train'
new_fn = 'seg_track_train_new.json'
with open(fn) as f:
    seg_track_dir = json.load(f)

new_track_dir = seg_track_dir
new_track_dir["videos"] = []
new_track_dir["images"] = []
image_list = seg_track_dir["images"]
video_list = seg_track_dir["videos"]
for video in video_list:
    if os.path.exists(os.path.join(root_dir, video["name"])):
        new_track_dir["videos"].append(video)

for image in image_list:
    if os.path.exists(os.path.join(root_dir, image["name"][:17], image["name"])):
        new_track_dir["images"].append(image)

with open(new_fn, 'w') as f:
    json.dump(new_track_dir, f)
