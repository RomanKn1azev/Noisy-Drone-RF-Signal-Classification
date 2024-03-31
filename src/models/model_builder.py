import torch.nn as nn


from src.utils.logger import LOGGER

from collections import OrderedDict
from src.models.classes.cnn import SpectrogramCNN


class LayerBuilder:
    def __init__(self) -> None:
        self._dict_layers = {
            "Linear": self.build_linear,
            "Conv1d": self.build_conv1d,
            "Conv2d": self.build_conv2d,
            "BatchNorm1d": self.build_batch_norm1d,
            "BatchNorm2d": self.build_batch_norm2d,
            "ReLU": self.build_relu,
            "Sigmoid": self.build_sigmoid,
            "MaxPool1d": self.build_maxpool1d,
            "MaxPool2d": self.build_maxpool2d,
            "Dropout": self.build_dropout,
            "Flatten": self.build_flatten,
            "LogSoftmax": self.build_log_softmax
        }

    def build_conv1d(self, params):
        return nn.Conv1d(**params)

    def build_conv2d(self, params):
        return nn.Conv2d(**params)
    
    def build_batch_norm1d(self, params):
        return nn.BatchNorm1d(**params)
    
    def build_batch_norm2d(self, params):
        return nn.BatchNorm2d(**params)
    
    def build_sigmoid(self, params):
        return nn.Sigmoid()
    
    def build_relu(self, params):
        return nn.ReLU()

    def build_linear(self, params):
        return nn.Linear(**params)
    
    def build_maxpool1d(self, params):
        return nn.MaxPool1d(**params)

    def build_maxpool2d(self, params):
        return nn.MaxPool2d(**params)
    
    def build_dropout(self, params):
        return nn.Dropout(**params)
    
    def build_flatten(self, params):
        return nn.Flatten()
    
    def build_log_softmax(self, params):
        return nn.LogSoftmax()
    
    def build_layer_from_type(self, layer_type: str, params: dict):
        return self._dict_layers[layer_type](params)


def build_class_model(class_name: str):
    match class_name:
        case "SpectrogramCNN":
            return SpectrogramCNN

        case _:
            raise ValueError(f"Unknown model {class_name}")
        

def build_model(model_params: dict):
    blocks = []
    layer_builder = LayerBuilder()

    class_name = model_params['class_name']
    class_model = build_class_model(class_name)

    for block_params in model_params['blocks']:
        block_name = block_params['name']
        layers = []

        for layer_param in block_params['layers']:
            type = layer_param['type']
            name = layer_param['name']
            params = layer_param.get('params', {})

            layer = layer_builder.build_layer_from_type(type, params)

            layers.append((name, layer))

        blocks.append(
            (block_name, 
            nn.Sequential(OrderedDict(layers)))
        )    

    model = class_model(nn.Sequential(OrderedDict(blocks)))

    LOGGER.info(f"Class model: {class_name}\nModel architecture:\n{str(model)}")

    # save_params = model_params["save"]

    return model