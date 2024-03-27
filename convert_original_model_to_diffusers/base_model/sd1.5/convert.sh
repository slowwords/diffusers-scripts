python convert_base_model.py \
     --checkpoint_path [ckpt_path] \
     --original_config_file v1-inference.yaml \
     --dump_path [out_path] \
     --device cuda \
     --num_in_channels 4 \
     --scheduler_type dpm \
     --image_size 512 \
     --prediction_type epsilon \
     --from_safetensors