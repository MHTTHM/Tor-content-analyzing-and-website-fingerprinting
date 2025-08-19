import torch
import utils
import os
from glob import glob

def load_model_and_print_params(model_path: str):
    """
    Load the model from the given path and print its parameters.

    Parameters:
    - model_path: str, the path to the model's saved state dictionary.
    - config_path: str, the path to the configuration file required to initialize the model structure.
    """
    # Assuming your utils.get_model_from_pretrain function requires a config dict to initialize the model,
    # and the config can be loaded with a utility function similar to utils.parse_config.
    #config = torch.load(config_path, map_location='cpu')
    config = torch.load(
            glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')

    # Load model structure based on config
    model, _, _ = utils.get_model_from_pretrain(
        model_path=model_path,
        config=config,
        load_level='all',  # Assuming this argument loads the entire model including its state
        resume=True  # Assuming this argument indicates loading weights
    )
    
    # Print model's state_dict keys and their respective shapes
    print("Model's parameters:")
    for param_tensor in model.state_dict():
        print(f"{param_tensor} \t {model.state_dict()[param_tensor].size()}")

# Example usage
model_path = r"D:\博士研究\论文写作\202410类别分类论文\Suervised-Contrasive-Learning\MulSupCon\codes\experiment\yeast\Pretrain\2024-11-15_18-23-05-303208\finetune\2024-12-02_21-15-43-092781"

load_model_and_print_params(model_path)

