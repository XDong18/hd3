export CUDA_VISIBLE_DEVICES=1,2,3,4
python -u inference.py \
  --task=flow \
  --data_root=/shared/xudongliu/MPI \
  --data_list=lists/MPISintel_test_final_pass.txt \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=16 \
  --workers=16 \
  --context \
  --flow_format=flo \
  --model_path=model_zoo/hd3fc_chairs_things_sintel-0be17c83.pth \
  --save_folder=predictions/fc_pre_Sintel
