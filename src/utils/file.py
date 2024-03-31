import csv
import yaml
import os


from torch import save

def create_dir(path: str):
    os.makedirs(path, exist_ok=True)


def join_path_file_name(path: str, name: str, type: str):
    return os.path.join(path, f"{name}.{type}")


def create_data_list(path: str):
    return [
        join_path_file_name(path, file_name) for file_name in os.listdir(path)
    ]


def save_dict_csv(data: dict, path: str):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, data.keys())
        writer.writeheader()
        writer.writerows(data)


def save_torch_data(data, path: str):
    save(data, path)


def save_dict_yml(data: dict, path: str):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yml_file(file_path: str) -> dict:
    """
    Load the content of a YAML file and return it as a Python dictionary.

    :param file_path: The path to the YAML file.
    :return: The content of the YAML file as a Python dictionary.
    """
     
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)