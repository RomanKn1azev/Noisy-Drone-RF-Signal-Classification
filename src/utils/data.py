from torch import float64, float32, float16, long as torch_long


def append_list_value_in_dict(orig_dict: dict, add_dict: dict):
    for key in orig_dict.keys():
        orig_dict[key].append(add_dict[key])


def extend_list_value_in_dict(orig_dict: dict, add_dict: dict):
    for key in orig_dict.keys():
        if len(add_dict[key]) == 1:
            orig_dict[key] += add_dict[key]
        else:
            orig_dict[key].extend(add_dict[key])


def extract_data_from_avg_power_spectrum_file(file_path: str) -> tuple:
    with open(file_path, 'r') as file:
        label = file.readline().strip()
        data = [float(x.strip()) for x in file.readlines()]

    return data, label


def get_tensor_type(type_name: str):
    match type_name:
        case "float64":
            return float64
        
        case "float32":
            return float32
        
        case "float16":
            return float16
        
        case "long":
            return torch_long
        
        case _:
            raise ValueError(f"Unknown tensor type {type_name}")