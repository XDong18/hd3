export CUDA_VISIBLE_DEVICES=5,6,7,8
python -u inference.py \
  --task=flow \
  --data_root=/data5/bdd100k/images/track/val \
  --data_list=lists/seg_track_val_new.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --flow_format=png \
  --model_path=model_zoo/hd3fc_chairs_things_sintel-0be17c83.pth \
  --save_folder=predictions/fc_pre_Sintel_seg_track_val
