export CUDA_VISIBLE_DEVICES=1,2,3,4
python -u train.py \
  --dataset_name=KITTI \
  --train_root=/shared/xudongliu/bdd_part/train \
  --train_list=lists/seg_track_train.txt \
  --train_coco=seg_track_train_new.json \
  --val_coco=/shared/xudongliu/bdd_part/seg_track_val_new.json \
  --val_root=/shared/xudongliu/bdd_part/train \
  --val_list=lists/seg_track_train.txt \
  --task=flow \
  --base_lr=1e-3 \
  --encoder=dlaup \
  --decoder=hda \
  --context \
  --workers=16 \
  --epochs=200 \
  --batch_size=16 \
  --batch_size_val=4 \
  --visual_freq=20 \
  --save_step=50 \
  --save_path=checkpoints/seg_track_bdd_1e-3_me_bce_-1_nopre_second
