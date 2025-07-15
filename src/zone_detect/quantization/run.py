from copy import deepcopy
import yaml

from src.zone_detect.main import prepare_model
from src.zone_detect.utils import setup


def get_configs():
    """Load and return the configurations for methods and pipeline.
    The configurations are loaded from YAML files in th e'configs' folder, the paths are hardcoded.
    """
    methods_cfg_path = "src/zone_detect/quantization/configs/methods_config.yaml"
    with open(methods_cfg_path, "r") as file:
        methods_config = yaml.safe_load(file)

    pipeline_cfg_path = "src/zone_detect/quantization/configs/pipeline_config.yaml"
    pipeline_config, device, _ = setup({"conf": pipeline_cfg_path})

    return methods_config, pipeline_config, device


def dry_run():
    print("No actions will be performed.")

    methods_config, pipeline_config, device = get_configs()

    # print each method
    for method_name, method_params in methods_config.items():
        if method_params.get("enabled", True):
            print(f"\nMethod: {method_name}")
            for param, value in method_params.items():
                if param != "enabled":
                    print(f"  {param}: {value}")

            # load model and print specs
            config = deepcopy(pipeline_config)
            flag = method_params.get("flag", "")
            config[flag] = True
            if "pruned" in method_name:
                config["sparse"] = method_params.get("sparse", 0.005)
            config = prepare_model(config, device)

            # Print model details
            print(f"\nModel type: {config.get('model_type', 'unknown')}")
            model = config.get("model_args", {}).get("model", None)
            if model:
                print(f"\nModel precision: {model.dtype}")
            else:
                print(f"\nModel arguments: {config.get('model_args', {})}")

    print(f"\nOutputs are saved to: {pipeline_config.get('output_path', 'outputs/')}")

    print("\nDry run complete.")


def run_pipeline():

    methods_config, pipeline_config, device = get_configs()

    raise NotImplementedError("Pipeline execution logic is not implemented yet.")
    # TODO
