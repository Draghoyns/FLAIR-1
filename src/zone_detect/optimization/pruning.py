from torch import nn
import numpy as np
import pandas as pd
from pruna import SmashConfig, smash


#### OPTIMIZATION ####
def opti_pruna(model: nn.Module, params: dict) -> nn.Module:
    """
    Apply pruna algorithms.
    """

    smash_config = SmashConfig()  # see pruna documentation for details

    for key, value in params.get("methods", {}).items():
        smash_config[str(key)] = str(value)

    sparse = params.get("sparse", 0)
    if sparse != 0:
        print(f"Applying pruna with sparsity: {sparse:.2%}")

        smash_config["pruner"] = "torch_unstructured"
        smash_config["torch_unstructured_sparsity"] = sparse
        # smash_config["torch_unstructured_pruning_method"] = "random"

    # smash_config["quantizer"] = "half"
    # smash_config["quantizer"] = "torch_dynamic"

    model = smash(model=model, smash_config=smash_config)

    return model


def analyze_model_weights(
    model: nn.Module,
    include_bias: bool = True,
    epsilon: float = 1e-3,
    save: bool = False,
) -> list:
    report = []
    group_stats = {}
    total_near_zeros = 0
    total_elements = 0

    for name, param in model.named_parameters():
        if not include_bias and "bias" in name:
            continue
        if param.requires_grad:
            data = param.data.cpu().numpy()
            abs_data = np.abs(data)
            max_val = abs_data.max()

            if max_val == 0:
                near_zero_mask = abs_data == 0
            else:
                threshold = epsilon * max_val
                near_zero_mask = abs_data < threshold

            near_zero_count = near_zero_mask.sum()
            element_count = data.size

            total_near_zeros += near_zero_count
            total_elements += element_count

            layer_type = name.split(".")[0] if "." in name else name
            if layer_type not in group_stats:
                group_stats[layer_type] = {"near_zeros": 0, "elements": 0}
            group_stats[layer_type]["near_zeros"] += near_zero_count
            group_stats[layer_type]["elements"] += element_count

            if save:
                report.append(
                    {
                        "Layer.Parameter": name,
                        "Shape": list(data.shape),
                        "Mean": data.mean(),
                        "Std": data.std(),
                        "Min": data.min(),
                        "Max": data.max(),
                        "Near-Zeros (<{:.0e} * max)".format(epsilon): near_zero_count,
                        "Near-Zero Ratio": near_zero_count / element_count,
                    }
                )

    overall_sparsity = total_near_zeros / total_elements if total_elements > 0 else 0
    print(
        f"\n🔍 Overall near-zero sparsity (< {epsilon:.0e} * max): {overall_sparsity:.4%}"
    )

    print("\n📊 Sparsity by Principal Layer:")
    for group_name, stats in group_stats.items():
        group_sparsity = (
            stats["near_zeros"] / stats["elements"] if stats["elements"] > 0 else 0
        )
        print(f"{group_name}: {group_sparsity:.4%}")

    return report


#### SPARSITY ####
def sparsity(model: nn.Module, save: str = "") -> None:
    """Compute the sparsity of the model from the configuration file.
    Args:
        model (nn.Module): The model to analyze.
    """

    if model is None:
        raise ValueError("Please provide a model.")

    report = analyze_model_weights(model, save=bool(save))
    report_df = pd.DataFrame(report)

    if save:
        out_path = save if save.endswith(".csv") else save + ".csv"
        report_df.to_csv(out_path, index=False)
        print(f"Model weight report saved to {out_path}")
