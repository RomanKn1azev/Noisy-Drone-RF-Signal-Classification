import torch


from numpy.random import seed as np_seed
from random import seed as rand_seed
from torch import manual_seed
from torch.cuda import manual_seed_all


def setting_seed(seed: int):
    rand_seed(seed)
    np_seed(seed)
    manual_seed(seed)
    manual_seed_all(seed)


def setting_device(device_name: str):
    if device_name == "cuda":
        if torch.cuda.is_available():
            print(
                "Setting device cuda."
            )
        else:
            print(
                f"CUDA is not available. Setting device to cpu."
            )
            device_name = "cpu"
    elif device_name == "cpu":
        print(
            "Setting device cpu."
        )
    else:
        print(
            f"Unsupported device: {device_name}. Setting device cpu."
        )
    
    return torch.device(device_name)