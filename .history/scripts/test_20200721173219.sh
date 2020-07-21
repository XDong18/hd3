python -u inference.py \
  --task=flow \
  --data_root=path_to_test_data \
  --data_list=lists/test_data_list \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=png/flo \
  --evaluate \
  --model_path=path_to_trained_model \
  --save_folder=path_to_save_predictions
