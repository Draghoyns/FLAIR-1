from copy import deepcopy
import os
import sys
import yaml

from src.zone_detect.main import prepare_model
from src.zone_detect.utils import Logger, setup


def get_configs():
    """Load and return the configurations for methods and pipeline.
    The configurations are loaded from YAML files in the 'configs' folder, the paths are hardcoded.
    """
    methods_cfg_path = (
        "src/zone_detect/optimization/quantization/configs/methods_config.yaml"
    )
    with open(methods_cfg_path, "r") as file:
        methods_config = yaml.safe_load(file)

    pipeline_cfg_path = (
        "src/zone_detect/optimization/quantization/configs/pipeline_config.yaml"
    )
    pipeline_config, device, _ = setup({"conf": pipeline_cfg_path})

    return methods_config, pipeline_config, device


def log_init(out: str):
    """Initialize logging."""
    out_path = "src/zone_detect/quantization/outputs"
    os.makedirs(out_path, exist_ok=True)
    sys.stdout = Logger(filename=f"{out_path}/{out}.log")
    sys.stderr = sys.stdout
    print(f"    [LOGGER] Writing logs to: {out_path}/{out}.log")


def dry_run():
    log_init("dry_run")
    print("No actions will be performed.")

    methods_config, pipeline_config, device = get_configs()

    # print each method
    for method_name, method_params in methods_config.items():
        if method_params.get("enable", True):

            config = deepcopy(pipeline_config)
            flag = method_params.get("flag", "")
            config[flag] = True

            config.update(
                {
                    f"{flag}_args": method_params,
                }
            )

            # print parameters
            print(f"\nMethod: {method_name}")
            for param, value in method_params.items():
                if param != "enable":
                    print(f"  {param}: {value}")

            # load model and print specs
            config = prepare_model(config, device)

            # print model details
            print(f"Model type: {config.get('model_type', 'unknown')}")
            model = config.get("model_args", {}).get("model", None)
            if model:
                print(f"Model precision: {model.dtype}")
            else:
                print(f"Model arguments: {config.get('model_args', {})}")

    print(f"\nOutputs are saved to: {pipeline_config.get('output_path', 'outputs/')}")

    print("\nDry run complete.")

    sys.stdout = sys.__stdout__


def run_pipeline():

    methods_config, pipeline_config, device = get_configs()

    for method_name, method_params in methods_config.items():
        pass

    raise NotImplementedError("Pipeline execution logic is not implemented yet.")
    # TODO
