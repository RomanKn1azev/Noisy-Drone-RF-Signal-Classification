import torch


from torch.utils.data import TensorDataset


def load_dataset(path: str):
    return load_torch_data(path) # TensorDataset(load_torch_data(path))


def load_torch_data(path: str):
    return torch.load(path)