import json
import os

fn = '/data5/bdd100k/labels/seg_track/seg_track_train.json'
root_dir = '/data5/bdd100k/images/track/train'
new_fn = 'seg_track_train_new.json'
with open(fn) as f:
    seg_track_dir = json.load(f)

new_track_dir = seg_track_dir
temp_videos = []
temp_images = []
image_list = seg_track_dir["images"]
video_list = seg_track_dir["videos"]
for video in video_list:
    if os.path.exists(os.path.join(root_dir, video["name"])):
        temp_videos.append(video)
print(len(temp_videos))

for image in image_list:
    if os.path.exists(os.path.join(root_dir, image["file_name"][:17], image["file_name"])):
        temp_images.append(image)
print(len(temp_images))

new_track_dir["videos"] = temp_videos
new_track_dir["images"] = temp_images
with open(new_fn, 'w') as f:
    json.dump(new_track_dir, f)
