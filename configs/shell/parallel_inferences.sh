
conda activate opti

flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cpu_512.yaml -b -c -m --onnx &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cpu_1024.yaml -b -c -m --onnx &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cuda_512.yaml -b -c -m --onnx &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cuda_1024.yaml -b -c -m --onnx &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cpu_512.yaml -b -c -m &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cpu_1024.yaml -b -c -m &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cuda_512.yaml -b -c -m &
flair-detect --conf=/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_cuda_1024.yaml -b -c -m &
