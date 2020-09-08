export CUDA_VISIBLE_DEVICES=5,6,7,8
python -u inference.py \
  --task=flow \
  --data_root=/shared/xudongliu/bdd_part/val \
  --data_list=/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --context \
  --flow_format=png \
  --model_path=/shared/xudongliu/code/semi-flow/hd3/checkpoints/seg_track_bdd_1e-3_me_bce_-1_+/model_latest.pth \
  --save_folder=predictions/seg_track_bdd_1e-3_me_bce_-1_+_epoch1

python color_mask.py \
  --fl_base=predictions/seg_track_bdd_1e-3_me_bce_-1_+_epoch1/vec \
  --out_dir=generated_color_map/seg_track_bdd_1e-3_me_bce_-1_+_epoch1/frame_0

python instance_iou.py \
  --fl_base=predictions/seg_track_bdd_1e-3_me_bce_-1_+_epoch1/vec


