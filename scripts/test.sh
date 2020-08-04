export CUDA_VISIBLE_DEVICES=5,6,7,8
python -u inference.py \
  --task=flow \
  --data_root=/data5/bdd100k/images/track/val \
  --data_list=/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --flow_format=png \
  --model_path=/shared/xudongliu/code/hd3/checkpoints/ft_fc_MPI_2_r/model_best.pth \
  --save_folder=predictions/fc_pre_Sintel_seg_track_val_my
