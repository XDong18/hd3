export CUDA_VISIBLE_DEVICES=5,6,7,8
python -u inference.py \
  --task=flow \
  --data_root=/data5/bdd100k/images/track/val \
  --data_list=/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --context \
  --flow_format=png \
  --model_path=model_zoo/hd3fc_chairs_things-0b92a7f5.pth \
  --save_folder=predictions/chairs_thngs_gap_1
