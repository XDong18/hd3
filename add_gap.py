import os

sur_fn = '/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt'
out_fn = 'lists/seg_track_val_new_5.txt'

with open(sur_fn) as f:
    lines = f.readlines()

new_lines = lines[::5]

with open(out_fn, 'w') as f:
    for new_line in new_lines:
        f.write(new_line)