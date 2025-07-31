import torch
from transformers import AutoConfig, AutoModelForSemanticSegmentation
import yaml

# Path to config and checkpoint
CONFIG_PATH = "/home/ign.fr/SHys/FLAIR-1/configs/flair-1-config.yaml"
CKPT_PATH = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt"


def load_and_quantize_with_torch_export():
    """
    Alternative function using torch.export.export to save and load quantized models
    """
    print("=== Using torch.export.export method ===")

    # 1. Load the config file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # 2. Extract HuggingFace model name and number of classes
    hf_model_name = config["model_framework"]["HuggingFace"]["org_model"]
    n_classes = len(config["classes"])

    print(f"Loading model: {hf_model_name}")
    print(f"Number of classes: {n_classes}")

    # 3. Load HuggingFace config and model
    cfg_model = AutoConfig.from_pretrained(
        hf_model_name,
        num_labels=n_classes,
    )
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(
        hf_model_name,
        config=cfg_model,
        ignore_mismatched_sizes=True,
    )

    # 4. Load checkpoint weights
    print(f"Loading checkpoint from: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if it exists
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove 'model.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        seg_model.load_state_dict(new_state_dict, strict=False)
    else:
        seg_model.load_state_dict(checkpoint, strict=False)

    print("Original model loaded successfully!")

    # 5. Quantize to int8 using dynamic quantization
    print("Quantizing model to int8...")
    # quantized_model = torch.quantization.quantize_dynamic(
    #     seg_model,
    #     {torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d},
    #     dtype=torch.qint8
    # )

    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    import copy

    quantized_model = copy.deepcopy(seg_model)
    quantize_(
        quantized_model,
        Int8WeightOnlyConfig(),  # {torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d}, # dtype=torch.qint8
    )  # type: ignore

    print("Model quantized successfully!")
    print(
        f"Quantized model size: {sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024:.2f} MB"
    )
    # 6. Prepare example input for export
    example_input = torch.randn(1, 3, 512, 512)

    # 7. Export the quantized model using torch.export
    print("Exporting quantized model with torch.export...")
    try:
        # Export the model
        exported_model = torch.export.export(
            quantized_model,
            (example_input,),
            dynamic_shapes=None,  # Use static shapes for simplicity
        )

        # Save the exported model
        torch.export.save(exported_model, "quantized_model_exported.pt")
        print("Quantized model exported and saved successfully!")

        # Load the exported model
        print("Loading exported quantized model...")
        loaded_exported_model = torch.export.load(
            "quantized_model_exported.pt"
        ).module()
        print("Exported quantized model loaded successfully!")

        # Test inference with exported model
        print("Testing inference with exported model...")
        # ExportedProgram doesn't have .eval(), just call it directly
        with torch.no_grad():
            exported_output = loaded_exported_model(example_input)
            # Handle the output - ExportedProgram returns the raw output
            if isinstance(exported_output, dict) and "logits" in exported_output:
                exported_output = exported_output["logits"]
            elif hasattr(exported_output, "logits"):
                exported_output = exported_output.logits  # type: ignore

                print(f"Exported model output shape: {exported_output.shape}")

        return loaded_exported_model, exported_output

    except Exception as e:
        print(f"torch.export failed: {e}")
        print("Falling back to regular save/load method...")
        return None, None


def benchmark_model_performance(model, model_name, test_input, num_runs=100):
    """
    Benchmark model performance including timing, memory usage, and throughput
    """
    print(f"\n=== Performance Benchmark for {model_name} ===")

    import time
    import psutil
    import gc

    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)

    # Clear cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Measure memory before inference
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Benchmark inference time
    print(f"Running {num_runs} inference iterations...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(num_runs):
            output = model(test_input)
            if i % 20 == 0:  # Progress indicator
                print(f"  Progress: {i}/{num_runs}")

    end_time = time.time()

    # Measure memory after inference
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_runs
    throughput = num_runs / total_time  # inferences per second

    # Calculate model size
    if hasattr(model, "parameters"):
        model_size = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        )
    else:
        model_size = "N/A (ExportedProgram)"

    print(f"\n--- Performance Results ---")
    print(
        f"Model size: {model_size:.2f} MB"
        if isinstance(model_size, float)
        else f"Model size: {model_size}"
    )
    print(f"Total inference time ({num_runs} runs): {total_time:.3f} seconds")
    print(f"Average time per inference: {avg_time_per_inference*1000:.2f} ms")
    print(f"Throughput: {throughput:.1f} inferences/second")
    print(f"Memory usage during inference: {memory_used:.2f} MB")
    print(f"Memory before: {memory_before:.2f} MB")
    print(f"Memory after: {memory_after:.2f} MB")

    return {
        "model_size": model_size,
        "avg_time_ms": avg_time_per_inference * 1000,
        "throughput": throughput,
        "memory_used": memory_used,
    }


def get_model_size_mb(model):
    """
    Get the actual model size in MB (size of parameters)
    """
    if hasattr(model, "parameters"):
        total_params = 0
        total_size = 0
        for param in model.parameters():
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
        return total_size / 1024 / 1024, total_params
    else:
        # For ExportedProgram, we can't easily get parameter size
        return "N/A (ExportedProgram)", "N/A"


def compare_quantized_vs_original(
    original_model, quantized_model, num_samples=20, input_size=(1, 3, 512, 512)
):
    """
    Compare performance and accuracy between original and quantized models
    """
    print(f"\n=== Quantized vs Original Model Comparison ===")
    print(f"Testing on {num_samples} random samples...")

    import time
    import psutil
    import gc
    import numpy as np

    # Get model sizes first
    print("Calculating model sizes...")
    orig_size_mb, orig_params = get_model_size_mb(original_model)
    quant_size_mb, quant_params = get_model_size_mb(quantized_model)

    print(
        f"Original model size: {orig_size_mb:.2f} MB ({orig_params:,} parameters)"
        if isinstance(orig_size_mb, float)
        else f"Original model size: {orig_size_mb}"
    )
    print(
        f"Quantized model size: {quant_size_mb:.2f} MB ({quant_params:,} parameters)"
        if isinstance(quant_size_mb, float)
        else f"Quantized model size: {quant_size_mb}"
    )

    if isinstance(orig_size_mb, float) and isinstance(quant_size_mb, float):
        model_size_reduction = (orig_size_mb - quant_size_mb) / orig_size_mb * 100
        print(f"Model size reduction: {model_size_reduction:.1f}%")
    else:
        model_size_reduction = "N/A (one of the models is not quantized)"
        print("Model size reduction: N/A (one of the models is not quantized)")

    # Generate test samples
    test_samples = [torch.randn(*input_size) for _ in range(num_samples)]

    # Warm up both models
    print("Warming up models...")
    with torch.no_grad():
        for _ in range(5):
            _ = original_model(test_samples[0])
            _ = quantized_model(test_samples[0])

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Test original model
    print("Testing original model...")
    process = psutil.Process()
    memory_before_orig = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    original_outputs = []
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            output = original_model(sample)
            if hasattr(output, "logits"):
                output = output.logits
            original_outputs.append(output)
            if i % 5 == 0:
                print(f"  Original model progress: {i+1}/{num_samples}")

    orig_time = time.time() - start_time
    memory_after_orig = process.memory_info().rss / 1024 / 1024
    orig_memory = memory_after_orig - memory_before_orig

    # Test quantized model
    print("Testing quantized model...")
    memory_before_quant = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    quantized_outputs = []
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            output = quantized_model(sample)
            if hasattr(output, "logits"):
                output = output.logits
            quantized_outputs.append(output)
            if i % 5 == 0:
                print(f"  Quantized model progress: {i+1}/{num_samples}")

    quant_time = time.time() - start_time
    memory_after_quant = process.memory_info().rss / 1024 / 1024
    quant_memory = memory_after_quant - memory_before_quant

    # Calculate accuracy metrics
    print("Calculating accuracy metrics...")
    max_diffs = []
    mean_diffs = []
    cosine_similarities = []

    for orig_out, quant_out in zip(original_outputs, quantized_outputs):
        # Ensure same shape
        if orig_out.shape != quant_out.shape:
            print(
                f"Warning: Shape mismatch - Original: {orig_out.shape}, Quantized: {quant_out.shape}"
            )
            continue

        # Calculate differences
        diff = torch.abs(orig_out - quant_out)
        max_diffs.append(torch.max(diff).item())
        mean_diffs.append(torch.mean(diff).item())

        # Calculate cosine similarity
        orig_flat = orig_out.flatten()
        quant_flat = quant_out.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
        ).item()
        cosine_similarities.append(cos_sim)

    # Calculate statistics
    avg_max_diff = np.mean(max_diffs)
    avg_mean_diff = np.mean(mean_diffs)
    avg_cosine_sim = np.mean(cosine_similarities)

    # Performance metrics
    orig_avg_time = orig_time / num_samples * 1000  # ms
    quant_avg_time = quant_time / num_samples * 1000  # ms
    speedup = orig_avg_time / quant_avg_time if quant_avg_time > 0 else float("inf")

    # Memory efficiency (runtime memory)
    runtime_memory_reduction = (
        ((orig_memory - quant_memory) / orig_memory * 100) if orig_memory > 0 else 0
    )

    print(f"\n--- Comparison Results ---")
    print(f"Model Size:")
    print(
        f"  Original model: {orig_size_mb:.2f} MB"
        if isinstance(orig_size_mb, float)
        else f"  Original model: {orig_size_mb}"
    )
    print(
        f"  Quantized model: {quant_size_mb:.2f} MB"
        if isinstance(quant_size_mb, float)
        else f"  Quantized model: {quant_size_mb}"
    )
    if isinstance(orig_size_mb, float) and isinstance(quant_size_mb, float):
        print(f"  Model size reduction: {model_size_reduction:.1f}%")

    print(f"\nPerformance:")
    print(f"  Original model avg time: {orig_avg_time:.2f} ms")
    print(f"  Quantized model avg time: {quant_avg_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Original model runtime memory: {orig_memory:.2f} MB")
    print(f"  Quantized model runtime memory: {quant_memory:.2f} MB")
    print(f"  Runtime memory reduction: {runtime_memory_reduction:.1f}%")

    print(f"\nAccuracy:")
    print(f"  Average max difference: {avg_max_diff:.6f}")
    print(f"  Average mean difference: {avg_mean_diff:.6f}")
    print(f"  Average cosine similarity: {avg_cosine_sim:.6f}")
    print(f"  Max difference range: {min(max_diffs):.6f} - {max(max_diffs):.6f}")
    print(
        f"  Cosine similarity range: {min(cosine_similarities):.6f} - {max(cosine_similarities):.6f}"
    )

    return {
        "speedup": speedup,
        "runtime_memory_reduction": runtime_memory_reduction,
        "model_size_reduction": (
            model_size_reduction
            if isinstance(orig_size_mb, float) and isinstance(quant_size_mb, float)
            else "N/A"
        ),
        "avg_max_diff": avg_max_diff,
        "avg_mean_diff": avg_mean_diff,
        "avg_cosine_sim": avg_cosine_sim,
        "orig_time_ms": orig_avg_time,
        "quant_time_ms": quant_avg_time,
        "orig_memory": orig_memory,
        "quant_memory": quant_memory,
        "orig_size_mb": orig_size_mb,
        "quant_size_mb": quant_size_mb,
    }


def load_default_model():
    """
    Load the default model from checkpoint
    """
    print("=== Loading Default Model ===")

    # 1. Load the config file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # 2. Extract HuggingFace model name and number of classes
    hf_model_name = config["model_framework"]["HuggingFace"]["org_model"]
    n_classes = len(config["classes"])

    print(f"Loading model: {hf_model_name}")
    print(f"Number of classes: {n_classes}")

    # 3. Load HuggingFace config and model
    cfg_model = AutoConfig.from_pretrained(
        hf_model_name,
        num_labels=n_classes,
    )
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(
        hf_model_name,
        config=cfg_model,
        ignore_mismatched_sizes=True,
    )

    # 4. Load checkpoint weights
    print(f"Loading checkpoint from: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if it exists
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove 'model.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        seg_model.load_state_dict(new_state_dict, strict=False)
    else:
        seg_model.load_state_dict(checkpoint, strict=False)

    seg_model.eval()
    print("Default model loaded successfully!")

    return seg_model


def load_and_compile_model():
    """
    Load model and compile it with torch.compile for performance optimization
    """
    print("=== Loading and Compiling Model ===")

    # Load the default model first
    original_model = load_default_model()

    # Compile the model
    print("Compiling model with torch.compile...")
    try:
        compiled_model = torch.compile(original_model, mode="reduce-overhead")
        print("Model compiled successfully!")

        # Warm up the compiled model
        print("Warming up compiled model...")
        test_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            for _ in range(10):
                _ = compiled_model(test_input)

        return original_model, compiled_model

    except Exception as e:
        print(f"torch.compile failed: {e}")
        print("Falling back to original model...")
        return original_model, original_model


def compare_compiled_vs_original(
    original_model, compiled_model, num_samples=20, input_size=(1, 3, 512, 512)
):
    """
    Compare performance between original and compiled models
    """
    print(f"\n=== Compiled vs Original Model Comparison ===")
    print(f"Testing on {num_samples} random samples...")

    import time
    import psutil
    import gc
    import numpy as np

    # Get model sizes first
    print("Calculating model sizes...")
    orig_size_mb, orig_params = get_model_size_mb(original_model)
    comp_size_mb, comp_params = get_model_size_mb(compiled_model)

    print(
        f"Original model size: {orig_size_mb:.2f} MB ({orig_params:,} parameters)"
        if isinstance(orig_size_mb, float)
        else f"Original model size: {orig_size_mb}"
    )
    print(
        f"Compiled model size: {comp_size_mb:.2f} MB ({comp_params:,} parameters)"
        if isinstance(comp_size_mb, float)
        else f"Compiled model size: {comp_size_mb}"
    )

    # Generate test samples
    test_samples = [torch.randn(*input_size) for _ in range(num_samples)]

    # Warm up both models
    print("Warming up models...")
    with torch.no_grad():
        for _ in range(5):
            _ = original_model(test_samples[0])
            _ = compiled_model(test_samples[0])

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Test original model
    print("Testing original model...")
    process = psutil.Process()
    memory_before_orig = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    original_outputs = []
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            output = original_model(sample)
            if hasattr(output, "logits"):
                output = output.logits
            original_outputs.append(output)
            if i % 5 == 0:
                print(f"  Original model progress: {i+1}/{num_samples}")

    orig_time = time.time() - start_time
    memory_after_orig = process.memory_info().rss / 1024 / 1024
    orig_memory = memory_after_orig - memory_before_orig

    # Test compiled model
    print("Testing compiled model...")
    memory_before_comp = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    compiled_outputs = []
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            output = compiled_model(sample)
            if hasattr(output, "logits"):
                output = output.logits
            compiled_outputs.append(output)
            if i % 5 == 0:
                print(f"  Compiled model progress: {i+1}/{num_samples}")

    comp_time = time.time() - start_time
    memory_after_comp = process.memory_info().rss / 1024 / 1024
    comp_memory = memory_after_comp - memory_before_comp

    # Calculate accuracy metrics
    print("Calculating accuracy metrics...")
    max_diffs = []
    mean_diffs = []
    cosine_similarities = []

    for orig_out, comp_out in zip(original_outputs, compiled_outputs):
        # Ensure same shape
        if orig_out.shape != comp_out.shape:
            print(
                f"Warning: Shape mismatch - Original: {orig_out.shape}, Compiled: {comp_out.shape}"
            )
            continue

        # Calculate differences
        diff = torch.abs(orig_out - comp_out)
        max_diffs.append(torch.max(diff).item())
        mean_diffs.append(torch.mean(diff).item())

        # Calculate cosine similarity
        orig_flat = orig_out.flatten()
        comp_flat = comp_out.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), comp_flat.unsqueeze(0)
        ).item()
        cosine_similarities.append(cos_sim)

    # Calculate statistics
    avg_max_diff = np.mean(max_diffs)
    avg_mean_diff = np.mean(mean_diffs)
    avg_cosine_sim = np.mean(cosine_similarities)

    # Performance metrics
    orig_avg_time = orig_time / num_samples * 1000  # ms
    comp_avg_time = comp_time / num_samples * 1000  # ms
    speedup = orig_avg_time / comp_avg_time if comp_avg_time > 0 else float("inf")

    # Memory efficiency (runtime memory)
    runtime_memory_change = (
        ((comp_memory - orig_memory) / orig_memory * 100) if orig_memory > 0 else 0
    )

    print(f"\n--- Comparison Results ---")
    print(f"Model Size:")
    print(
        f"  Original model: {orig_size_mb:.2f} MB"
        if isinstance(orig_size_mb, float)
        else f"  Original model: {orig_size_mb}"
    )
    print(
        f"  Compiled model: {comp_size_mb:.2f} MB"
        if isinstance(comp_size_mb, float)
        else f"  Compiled model: {comp_size_mb}"
    )

    print(f"\nPerformance:")
    print(f"  Original model avg time: {orig_avg_time:.2f} ms")
    print(f"  Compiled model avg time: {comp_avg_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Original model runtime memory: {orig_memory:.2f} MB")
    print(f"  Compiled model runtime memory: {comp_memory:.2f} MB")
    print(f"  Runtime memory change: {runtime_memory_change:+.1f}%")

    print(f"\nAccuracy:")
    print(f"  Average max difference: {avg_max_diff:.6f}")
    print(f"  Average mean difference: {avg_mean_diff:.6f}")
    print(f"  Average cosine similarity: {avg_cosine_sim:.6f}")
    print(f"  Max difference range: {min(max_diffs):.6f} - {max(max_diffs):.6f}")
    print(
        f"  Cosine similarity range: {min(cosine_similarities):.6f} - {max(cosine_similarities):.6f}"
    )

    return {
        "speedup": speedup,
        "runtime_memory_change": runtime_memory_change,
        "avg_max_diff": avg_max_diff,
        "avg_mean_diff": avg_mean_diff,
        "avg_cosine_sim": avg_cosine_sim,
        "orig_time_ms": orig_avg_time,
        "comp_time_ms": comp_avg_time,
        "orig_memory": orig_memory,
        "comp_memory": comp_memory,
        "orig_size_mb": orig_size_mb,
        "comp_size_mb": comp_size_mb,
    }


# def load_and_quantize_regular():
#     """
#     Original function using regular torch.save/load
#     """
#     print("=== Using regular torch.save/load method ===")

#     # 1. Load the config file
#     with open(CONFIG_PATH, "r") as f:
#         config = yaml.safe_load(f)

#     # 2. Extract HuggingFace model name and number of classes
#     hf_model_name = config['model_framework']['HuggingFace']['org_model']
#     n_classes = len(config['classes'])

#     print(f"Loading model: {hf_model_name}")
#     print(f"Number of classes: {n_classes}")

#     # 3. Load HuggingFace config and model
#     cfg_model = AutoConfig.from_pretrained(
#         hf_model_name,
#         num_labels=n_classes,
#     )
#     seg_model = AutoModelForSemanticSegmentation.from_pretrained(
#         hf_model_name,
#         config=cfg_model,
#         ignore_mismatched_sizes=True,
#     )

#     # 4. Load checkpoint weights
#     print(f"Loading checkpoint from: {CKPT_PATH}")
#     checkpoint = torch.load(CKPT_PATH, map_location="cpu")
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#         # Remove 'model.' prefix if it exists
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('model.'):
#                 new_key = key[6:]  # Remove 'model.' prefix
#             else:
#                 new_key = key
#             new_state_dict[new_key] = value
#         seg_model.load_state_dict(new_state_dict, strict=False)
#     else:
#         seg_model.load_state_dict(checkpoint, strict=False)

#     print("Original model loaded successfully!")

#     # 5. Quantize to int8 using dynamic quantization
#     print("Quantizing model to int8...")
#     quantized_model = torch.quantization.quantize_dynamic(
#         seg_model,
#         {torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d},
#         dtype=torch.qint8
#     )

#     print("Model quantized successfully!")

#     # 6. Save quantized model using torch.jit (better for quantized models)
#     print("Saving quantized model...")
#     try:
#         # Use torch.jit.save for quantized models
#         torch.jit.save(torch.jit.script(quantized_model), "quantized_hf_seg_model_int8.pt")
#         print("Quantized model saved successfully!")
#     except Exception as e:
#         print(f"Failed to save with torch.jit: {e}")
#         # Fallback to regular save
#         torch.save(quantized_model, "quantized_hf_seg_model_int8.pt")
#         print("Quantized model saved with regular torch.save")

#     # 7. Load quantized model back
#     print("Loading quantized model back...")
#     try:
#         # Try loading with torch.jit first
#         loaded_quantized_model = torch.jit.load("quantized_hf_seg_model_int8.pt")
#         print("Quantized model loaded successfully with torch.jit!")
#     except Exception as e:
#         print(f"Failed to load with torch.jit: {e}")
#         # Fallback to regular load
#         loaded_quantized_model = torch.load("quantized_hf_seg_model_int8.pt", map_location="cpu")
#         print("Quantized model loaded with regular torch.load")

#     # 8. Test inference with both models
#     print("Testing inference...")
#     test_input = torch.randn(1, 3, 512, 512)  # Adjust size based on your model's expected input

#     # Original model
#     seg_model.eval()
#     with torch.no_grad():
#         original_output = seg_model(test_input)
#         if hasattr(original_output, 'logits'):
#             original_output = original_output.logits

#     # Quantized model
#     loaded_quantized_model.eval()
#     with torch.no_grad():
#         quantized_output = loaded_quantized_model(test_input)
#         if hasattr(quantized_output, 'logits'):
#             quantized_output = quantized_output.logits

#     # 9. Compare outputs
#     print(f"Original output shape: {original_output.shape}")
#     print(f"Quantized output shape: {quantized_output.shape}")

#     # Calculate difference
#     diff = torch.abs(original_output - quantized_output)
#     max_diff = torch.max(diff).item()
#     mean_diff = torch.mean(diff).item()

#     print(f"Max difference: {max_diff:.6f}")
#     print(f"Mean difference: {mean_diff:.6f}")

#     # 10. Show model sizes
#     original_size = sum(p.numel() * p.element_size() for p in seg_model.parameters()) / 1024 / 1024
#     quantized_size = sum(p.numel() * p.element_size() for p in loaded_quantized_model.parameters()) / 1024 / 1024

#     print(f"Original model size: {original_size:.2f} MB")
#     print(f"Quantized model size: {quantized_size:.2f} MB")
#     print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

#     return loaded_quantized_model, quantized_output

if __name__ == "__main__":
    print("=== Model Optimization Comparison ===")
    print("Choose optimization method:")
    print("1. Quantization (int8)")
    print("2. Compilation (torch.compile)")
    print("3. Both")

    # For now, let's run both to demonstrate
    choice = "3"  # You can change this to "1", "2", or "3"

    if choice in ["1", "3"]:
        # Try torch.export method first
        print("\n" + "=" * 50)
        print("=== QUANTIZATION TEST ===")
        exported_model, exported_output = load_and_quantize_with_torch_export()

        # Benchmark the exported model performance
        if exported_model is not None:
            test_input = torch.randn(1, 3, 512, 512)
            performance_metrics = benchmark_model_performance(
                exported_model, "Exported Quantized Model", test_input, num_runs=50
            )

            print(f"\n=== Quantization Performance Summary ===")
            print(f"✓ Exported model benchmark completed")
            print(
                f"  - Average inference time: {performance_metrics['avg_time_ms']:.2f} ms"
            )
            print(
                f"  - Throughput: {performance_metrics['throughput']:.1f} inferences/sec"
            )
            print(f"  - Memory usage: {performance_metrics['memory_used']:.2f} MB")

            # Compare with original model (reload it for fair comparison)
            print("\n" + "=" * 50)
            print("Loading original model for quantization comparison...")

            # Reload original model
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)

            hf_model_name = config["model_framework"]["HuggingFace"]["org_model"]
            n_classes = len(config["classes"])

            cfg_model = AutoConfig.from_pretrained(hf_model_name, num_labels=n_classes)
            original_model = AutoModelForSemanticSegmentation.from_pretrained(
                hf_model_name, config=cfg_model, ignore_mismatched_sizes=True
            )

            # Load checkpoint weights
            checkpoint = torch.load(CKPT_PATH, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        new_key = key[6:]
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                original_model.load_state_dict(new_state_dict, strict=False)
            else:
                original_model.load_state_dict(checkpoint, strict=False)

            original_model.eval()
            print("Original model loaded for quantization comparison")

            # Compare quantized vs original
            comparison_metrics = compare_quantized_vs_original(
                original_model,
                exported_model,
                num_samples=15,  # Reduced for faster execution
                input_size=(1, 3, 512, 512),
            )

            print(f"\n=== Quantization Final Summary ===")
            print(f"✓ Quantization successful with torch.export")
            print(f"✓ Performance comparison completed")
            print(f"  - Speedup: {comparison_metrics['speedup']:.2f}x")
            print(
                f"  - Model size reduction: {comparison_metrics['model_size_reduction']:.1f}%"
                if isinstance(comparison_metrics["model_size_reduction"], float)
                else f"  - Model size reduction: {comparison_metrics['model_size_reduction']}"
            )
            print(
                f"  - Runtime memory reduction: {comparison_metrics['runtime_memory_reduction']:.1f}%"
            )
            print(
                f"  - Accuracy preserved: {comparison_metrics['avg_cosine_sim']:.4f} cosine similarity"
            )

    if choice in ["2", "3"]:
        # Try compilation method
        print("\n" + "=" * 50)
        print("=== COMPILATION TEST ===")
        original_model, compiled_model = load_and_compile_model()

        if compiled_model is not original_model:  # Compilation was successful
            # Compare compiled vs original
            compilation_metrics = compare_compiled_vs_original(
                original_model,
                compiled_model,
                num_samples=15,  # Reduced for faster execution
                input_size=(1, 3, 512, 512),
            )

            print(f"\n=== Compilation Final Summary ===")
            print(f"✓ Compilation successful with torch.compile")
            print(f"✓ Performance comparison completed")
            print(f"  - Speedup: {compilation_metrics['speedup']:.2f}x")
            print(
                f"  - Runtime memory change: {compilation_metrics['runtime_memory_change']:+.1f}%"
            )
            print(
                f"  - Accuracy preserved: {compilation_metrics['avg_cosine_sim']:.4f} cosine similarity"
            )
        else:
            print("✗ Compilation failed, using original model")

    print("\n" + "=" * 50)
    print("=== OVERALL SUMMARY ===")
    print("Model optimization testing complete!")
    print("Check the results above for detailed performance metrics.")
