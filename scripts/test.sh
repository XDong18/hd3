export CUDA_VISIBLE_DEVICES=4,5,6,7
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
  --model_path=/shared/xudongliu/code/semi-flow/hd3/checkpoints/seg_track_bdd_1e-3_edge/model_latest.pth \
  --save_folder=predictions/semi_lr_0.001_gap_1_edge_epoch6

python instance_iou.py


