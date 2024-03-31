import torch
import os


from src.utils.file import create_dir, join_path_file_name, save_torch_data, save_dict_csv, save_dict_yml
from src.utils.logger import LOGGER


def save_metrcis(
        metrics: dict,
        path: str,
        name: str,
        type: str
        ):
    create_dir(path)
    file_path = join_path_file_name(path, name, type)

    match type:
        case "csv":
            save_dict_csv(metrics, file_path)
            LOGGER.info(f"Save metrics in csv. Path: {file_path}")
            
        case "yml":
            save_dict_yml(metrics, file_path)
            LOGGER.info(f"Save metrics in yml. Path: {file_path}")

        case "pth":
            save_torch_data(metrics, file_path)
            LOGGER.info(f"Save metrics in pth. Path: {file_path}")
        
        case _:
            raise ValueError(f"Unknow save metrics type: {type}")


def save_dataset(
        dataset: torch.utils.data.Dataset,
        path: str,
        name: str,
        type: str="pt"
        ):
    create_dir(path)
    file_path = join_path_file_name(path, name, type)

    match type:
        case "pt":
            save_torch_data(dataset, file_path)
            LOGGER.info(f"Save dataset in pt. Path: {file_path}")
        
        case _:
            raise ValueError(f"Unknow save dataset type: {type}")