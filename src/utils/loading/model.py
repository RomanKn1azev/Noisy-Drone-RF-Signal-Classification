import torch
import onnxruntime


def load_jit_model(path: str):
    return torch.jit.load(path)


def load_state_model(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))


def load_model(path: str):
    return torch.load(path)


def load_inference_onnx_model(path: str):
    return onnxruntime.InferenceSession(path)