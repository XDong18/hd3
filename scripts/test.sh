export CUDA_VISIBLE_DEVICES=1,2,3,4
python -u inference.py \
  --task=flow \
  --data_root=/data5/bdd100k/images/track/val \
  --data_list=lists/seg_track_val_new.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=24 \
  --workers=16 \
  --flow_format=png \
  --model_path=model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth \
  --save_folder=predictions/fc_pre_KT_seg_track_val
