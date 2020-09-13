export CUDA_VISIBLE_DEVICES=1,2,3,4
python -u com_inference.py \
  --task=flow \
  --data_root=/shared/xudongliu/bdd_part/val \
  --data_list=/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --context \
  --flow_format=png \
  --model_path=/shared/xudongliu/code/semi-flow/hd3/checkpoints/seg_track_bdd_1e-3_me_bce_-1_+_nopre/model_latest.pth \
  --save_folder=predictions/seg_track_bdd_1e-3_xia_bce_epoch150predictions/seg_track_bdd_1e-3_me_bce_-1_+_nopre_epoch64