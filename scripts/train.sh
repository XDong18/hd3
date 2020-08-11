export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
python -u train.py \
  --dataset_name=KITTI \
  --train_root=/data5/bdd100k/images/track/train \
  --train_list=lists/seg_track_train.txt \
  --val_root=/data5/bdd100k/images/track/train \
  --val_list=lists/seg_track_train.txt \
  --task=flow \
  --base_lr=1e-5 \
  --pretrain=model_zoo/hd3fc_chairs_things-0b92a7f5.pth \
  --encoder=dlaup \
  --decoder=hda \
  --context \
  --workers=16 \
  --epochs=200 \
  --batch_size=32 \
  --batch_size_val=4 \
  --visual_freq=20 \
  --save_step=50 \
  --save_path=checkpoints/seg_track_bdd
