import os
import torch


from torch.utils.data import Dataset

from pathlib import Path
from typing import Union

from src.utils.file import load_yml_file, create_data_list
from src.utils.data import extract_data_from_avg_power_spectrum_file

from typing import Callable, Optional, Tuple


class DatasetBilder:
    def __call__(self, config_path: Union[Path, str]) -> Dataset:
        config = load_yml_file(config_path)
        class_name = config.get('class_name')
        params = config.get('params')

        match class_name:
            case "SpectrumSignalTargetDataset":
                return SpectrumSignalTargetDataset(**params)

            case _:
                raise ValueError(
                     f"Unsupported Dataset class: {class_name}"
                )
            
class DroneSignalsDatasetIQ(Dataset):
    def __init__(self, x_iq, y) -> None:
        self.data = x_iq
        self.targets = y

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, index) -> Tuple:
        return self.data[index], self.targets[index]


class DroneSignalsDatasetSpec(Dataset):
    def __init__(self, x_spec, y) -> None:
        self.data = x_spec
        self.targets = y

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, index) -> Tuple:
        return self.data[index], self.targets[index]
    

class DroneSignalsDatasetIQandSpec(Dataset):
    """
    Class for custom dataset of drone data comprised of
    x_iq (torch.tensor.float): signals iq data(n_samples x 2 x input_vec_length)
    x_spec (torch.tensor.float): signals spectogram (n_samples x 2 x num_segments x num_segments)
    y (torch.tensor.long): targets (n_samples)
    snrs (torch.tensor.int): SNRs per sample (n_samples) 
    duty_cycle (torch.tensor.float): duty cycle length per sample (n_samples) 
    Args:
        Dataset (torch tensor): 
    """
    def __init__(self, x_iq_tensor, x_spec_tensor, y_tensor, snr_tensor, duty_cycle_tensor):
        self.x_iq = x_iq_tensor
        self.x_spec = x_spec_tensor
        self.y = y_tensor
        self.snr = snr_tensor
        self.dury_cyle = duty_cycle_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_iq[idx], self.x_spec[idx], self.y[idx], self.snr[idx], self.dury_cyle[idx]   

    def targets(self):
        return self.y 

    def snrs(self):
        return self.snr

    def duty_cycle(self):
        return self.duty_cycle
            

class DataTargetDataset(Dataset):
    def __init__(
            self,
            root_path: Union[Path, str],
            type_extract_data_func: str,
            data_transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None 
            ):
        self.data_list = create_data_list(root_path)
        self.extraction_data = self.extract_data_func(type_extract_data_func)
        self.data_transform = data_transform
        self.target_transform = target_transform
    
    def extract_data_func(self, type_name: str):
        match type_name:
            case "extract_data_from_avg_power_spectrum_file":
                return extract_data_from_avg_power_spectrum_file

            case _:
                raise ValueError(f"Unknown type extract func: {type_name}")

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index: int) -> Tuple:
        # Extract data, target from file
        data, target = self.extraction_data(self.data_list[index])

        # Apply the data transform to the input data
        if self.data_transform:
            data = self.data_transform(data)

        # Apply the target transform to the target data
        if self.target_transform:
            target = self.target_transform(target)

        return data, target 


class SpectrumSignalTargetDataset(DataTargetDataset):
    def __init__(
            self, 
            root_path: Union[Path, str],
            type_extract_data_func: str, 
            data_transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None
            ):
        super().__init__(root_path, type_extract_data_func, data_transform, target_transform)