# let's try quantization with ONNX

# loading an already quantized model is handled using torchao

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

model_fp32 = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/openmmlab/upernet-swin-small_gpu_1x3x512x512.onnx"

out_dir = "/home/ign.fr/SHys/FLAIR-1/0testing_saves/onnx"
model_preprocessed = f"{out_dir}/upernet-swin-small_cpu_1x3x512x512_preprocessed.onnx"
model_quant = f"{out_dir}/upernet-swin-small_cpu_1x3x512x512_exclude.quant.onnx"

quant_pre_process(model_fp32, model_preprocessed)

quantized_model = quantize_dynamic(
    model_preprocessed,
    model_quant,
    op_types_to_quantize=[
        "Attention",
        "Gather",
        "Transpose",
        "EmbedLayerNormalization",
        "ArgMax",
        "Gemm",
        "MatMul",
        "Add",
        "Mul",
        "Relu",
        "Clip",
        "LeakyRelu",
        "Sigmoid",
        "MaxPool",
        "GlobalAveragePool",
        "Split",
        "Pad",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Resize",
        "AveragePool",
        "Concat",
        "Softmax",
        "Where",
    ],
)
