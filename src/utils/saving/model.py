import torch
import os

from src.utils.file import create_dir, join_path_file_name, save_torch_data
from src.utils.logger import LOGGER


def export_model_onnx(
            model: torch.nn.Module,
            input_tensor_size: tuple,
            path: str,
            name: str,
            export_params: bool,
            opset_version: int,
            input_names: list,
            output_names: list
            ):
        model.eval()

        dummy_input = torch.rand(*input_tensor_size)
        
        create_dir(path)
        file_path = join_path_file_name(path, name, "onxx")

        torch.onnx.export(
            model,
            dummy_input,
            file_path,
            export_params=export_params,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names
        )

        LOGGER.info(f"Export PyTorch model to ONNX is completed. Path: {file_path}")


def save_model_in_pt(
        model: torch.nn.Module,
        path: str,
        name: str
        ):
    """
        The reason for this is because pickle does not save the model class itself. 
    Rather, it saves a path to the file containing the class, which is used during load time.
    Because of this, 
    your code can break in various ways when used in other projects or after refactors.
    """

    create_dir(path)
    file_path = join_path_file_name(path, name, "pt")

    save_torch_data(model, file_path)

    LOGGER.info(f"Export PyTorch model to pt is completed. Path: {file_path}")


def save_model_state(
        model: torch.nn.Module,
        path: str,
        name: str
        ):

    create_dir(path)
    file_path = join_path_file_name(path, name, "pt")

    save_torch_data(model.state_dict(), file_path)

    LOGGER.info(f"Export PyTorch model state to pt is completed. Path: {file_path}")


def save_model_in_pt_jit(
        model: torch.nn.Module,
        path: str,
        name: str        
        ):
    """
    Using the TorchScript format, 
    you will be able to load the exported model and run inference without defining the model class.
    """
    create_dir(path)
    file_path = join_path_file_name(path, name, "pt")

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(file_path) # Save

    LOGGER.info(f"Export PyTorch model with jit to pt is completed. Path: {file_path}")


def build_checkpoint(
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict,
        path: str,
        name: str,
        valid_metrics: dict = None
        ):
    checkpoint = {
        'epoch': epoch,  # количество выполненных эпох
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if valid_metrics is not None:
         checkpoint["valid_metrics"] = valid_metrics

    create_dir(path)
    file_path = join_path_file_name(path, name, "pt")

    save_torch_data(checkpoint, file_path)

    LOGGER.info(f"Export chekpoint model and metrics params to pt is completed. Path: {file_path}")