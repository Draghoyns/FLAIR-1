olive auto-opt \
    --model_name_or_path /media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt \
    --trust_remote_code \
    --output_path /home/ign.fr/SHys/FLAIR-1/0testing_saves/onnx/ao \
    --device gpu \
    --provider CUDAExecutionProvider \
    --precision int8 \
    --log_level 1

# execute
# chmod +x /home/ign.fr/SHys/FLAIR-1/src/zone_detect/test/onnx/olive_test.sh


# actually more like
olive run --config /home/ign.fr/SHys/FLAIR-1/src/zone_detect/test/onnx/olive_config_pt.json