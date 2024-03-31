from src.utils.setting import setting_seed
from torch.utils.data import random_split

from src.utils.saving.data import save_dataset

from src.data.dataset import DatasetBilder


def creating_datasets(params: dict):
    setting_seed(params["random_state"])

    dataset_bilder = DatasetBilder()
    
    dataset = dataset_bilder(params["datasets"]["config_path"])

    split = params["datasets"].get("split", None)

    if split is not None:
        split_size = split["size"]
        dataset_list = random_split(dataset, split_size)

        for idx, save_param in enumerate(split["save"]):
            save_dataset(
                dataset_list[idx]
                **save_param
            )
    else:        
        save_dataset(
            dataset=dataset, 
            **params["dataset"].get("save")
        )