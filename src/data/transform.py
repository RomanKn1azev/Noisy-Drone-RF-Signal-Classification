import numpy as np
import torch


from sklearn import preprocessing
from src.utils.data import get_tensor_type
from src.utils.file import load_yml_file


def build_transform(type: str, params: dict):
    match type:
        case "SpectrumSignalTransform":
            return SpectrumSignalTransform(**params)
        
        case "DataTransform":
            return DataTransform(**params)
        
        case "TargetTransform":
            return TargetTransform(**params)

        case _:
            raise ValueError(f"Unknown transformer type {type}")


class SpectrumSignalTransform:
    def __init__(self, normalize_params: dict, interp_params: dict, tensor_type: str) -> None:
        self.scaler = self._build_scaler(normalize_params)
        self.interp = self._build_interp()
        
        self.interp_params = interp_params

        self.tensor_type = get_tensor_type(tensor_type)

    def _build_scaler(self, params: dict):
        type = params.pop('type')
        
        _range = params.pop('range')

        self.left = _range.get('left')
        self.right = _range.get('right')

        match type:
            case "MinMaxScaler":
                return preprocessing.MinMaxScaler(
                    feature_range=(self.left, self.right)
                    )
            # case "Standart":
            #     return preprocessing.normalize
            case _:
                raise ValueError(
                    f"Unsupported model type: {type}"
                )
    
    def _build_interp(self):
        return np.interp
    
    def __call__(self, original_array: list) -> torch.tensor:
        size_window = self.interp_params.get('size_window')
        x_inter = np.linspace(self.left, self.right, size_window)

        original_array_numpy = np.array(original_array).reshape(-1, 1)

        self.scaler.fit(original_array_numpy)

        scaled_data = self.scaler.transform(original_array_numpy).flatten()

        original_array_numpy_x = np.linspace(self.left, self.right, len(original_array_numpy))

        return torch.tensor(self.interp(x_inter, original_array_numpy_x, scaled_data), dtype=self.tensor_type)
    

class DataTransform:
    def __init__(self, params: dict) -> None:
        self.params = params

    def __call__(self) -> torch.tensor:
        if self.params is None:
            return None


class TargetTransform:
    def __init__(self, tensor_type: str, dictionary_path: str=None) -> None:
        self.tensor_type = get_tensor_type(tensor_type)

        if dictionary_path is not None:
            self.dictionary = load_yml_file(dictionary_path)
        else:
            self.dictionary = None

    def __call__(self, y) -> torch.tensor:
        if self.dictionary is not None:
            y = self.dictionary[y]
        
        return torch.tensor(y, dtype=self.tensor_type)
        
        